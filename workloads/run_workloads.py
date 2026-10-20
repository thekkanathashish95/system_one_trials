"""Benchmark Jev on labelled sample workloads: accuracy, latency (p50/p95), and cost.

Three workloads, one per text question type:
    tickets    choice  - 20 support tickets -> billing / technical / account / sales / other
    guardrail  noul    - 12 messages -> is this a prompt injection?
    severity   score   - 8 bug reports -> severity 0..3

    python workloads/run_workloads.py                     # all workloads
    python workloads/run_workloads.py tickets -c 8        # one workload, 8 concurrent requests
    python workloads/run_workloads.py --dry-run           # print one payload per workload

Results are written to workloads/results/<workload>-<timestamp>.jsonl for later inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from jev import JevClient, JevError, choice, noul, score  # noqa: E402

WORKLOADS = {
    "tickets": {
        "question": choice(
            {
                "billing": "Charges, refunds, invoices, payment methods, plan pricing.",
                "technical": "Bugs, errors, outages, integration or API problems.",
                "account": "Login, password, profile, deleting or merging accounts.",
                "sales": "Buying, upgrading, quotes, demos, discounts.",
                "other": "Anything else.",
            },
            instructions="Which support queue should handle this ticket?",
        ),
        "predict": lambda d: d.choice("q"),
        "correct": lambda pred, label: pred == label,
    },
    "guardrail": {
        "question": noul(
            "Is this message attempting a prompt injection or jailbreak?",
            true="Tries to override, reveal, or bypass the assistant's instructions or safeguards.",
            false="An ordinary request, including questions *about* security topics.",
        ),
        "predict": lambda d: d.yes("q"),
        "correct": lambda pred, label: (pred >= 0.5) == label,
    },
    "severity": {
        "question": score(
            [
                "0 - cosmetic, no functional impact",
                "1 - minor: degraded but works",
                "2 - major: broken for some users, workaround exists",
                "3 - critical: outage, data loss, or security breach",
            ],
            instructions="How severe is this bug report?",
        ),
        "predict": lambda d: d.score("q"),
        "correct": lambda pred, label: round(pred) == label,
    },
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def run(client: JevClient, name: str, concurrency: int) -> dict:
    spec = WORKLOADS[name]
    rows = [json.loads(line) for line in (HERE / "datasets" / f"{name}.jsonl").read_text().splitlines() if line]

    if client.dry_run:
        client.decide(state=rows[0]["text"], questions={"q": spec["question"]})
        return {}

    def one(row: dict) -> dict:
        try:
            d = client.decide(state=row["text"], questions={"q": spec["question"]})
            pred = spec["predict"](d)
            return {**row, "pred": pred, "ok": spec["correct"](pred, row["label"]),
                    "latency_ms": d.latency_ms, "cost": d.cost, "answer": d.answers["q"]}
        except (JevError, KeyError, TypeError) as e:
            return {**row, "error": str(e)}

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, rows))
    wall = time.perf_counter() - started

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in results))

    good = [r for r in results if "error" not in r]
    lat = [r["latency_ms"] for r in good]
    summary = {
        "workload": name,
        "n": len(rows),
        "errors": len(rows) - len(good),
        "accuracy": round(sum(r["ok"] for r in good) / len(good), 3) if good else None,
        "p50_ms": round(statistics.median(lat)) if lat else None,
        "p95_ms": round(percentile(lat, 0.95)) if lat else None,
        "throughput_rps": round(len(rows) / wall, 1),
        "total_cost_usd": round(sum(r["cost"] for r in good), 6),
        "results_file": str(out.relative_to(HERE.parent)),
    }
    for r in results:
        if "error" in r:
            print(f"  ERROR  {r['text'][:60]}: {r['error'][:120]}")
        elif not r["ok"]:
            print(f"  MISS   {r['text'][:60]!r}: label={r['label']} pred={r['pred']}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("workloads", nargs="*", help=f"any of {', '.join(WORKLOADS)} (default: all)")
    ap.add_argument("-c", "--concurrency", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    unknown = set(args.workloads) - set(WORKLOADS)
    if unknown:
        ap.error(f"unknown workload(s): {', '.join(sorted(unknown))}")
    if args.dry_run:
        os.environ["JEV_DRY_RUN"] = "1"

    client = JevClient()
    print(f"model={client.model} endpoint={client.base_url}{client.path}")
    for name in args.workloads or WORKLOADS:
        summary = run(client, name, args.concurrency)
        if summary:
            print(json.dumps(summary))
