# System One Trials: Jev (TypeSafe AI) templates for OpenRouter

> **GitHub template repo.** Click **Use this template** to start a new project from it.

Ready-to-copy templates for **Jev**, TypeSafe AI's *System One* model, called through OpenRouter.

Jev does not generate text. You send it **state** (your app's content: a string, JSON object,
or array) and named, **typed questions**. It returns **one typed answer per question, with
probabilities**, so you get no JSON prompting, no parsing, and no output validation. OpenRouter
lists input at $0.042 per million tokens with free output. TypeSafe says answers take 70–500 ms
(TypeSafe's own figure).

## Working modes

| # | Mode | Answer you get | Typical use | Template |
|---|------|----------------|-------------|----------|
| 1 | **choice** | one label + probability for every label | routing, intent, classification, tool picking | [`01_choice_ticket_triage.py`](templates/01_choice_ticket_triage.py) |
| 2 | **noul** (yes/no) | P(yes), 0–1 | guardrails, policy gates, "is the task done?" | [`02_noul_guardrail.py`](templates/02_noul_guardrail.py) |
| 3 | **score** | expected score on an ordered rubric + distribution | severity, lead scoring, grading, ranking | [`03_score_lead_qualification.py`](templates/03_score_lead_qualification.py) |
| 4 | **box** (vision) | `{top,bottom,left,right}` as 0–1 fractions + confidence | locating objects/regions in images | [`04_box_locate.py`](templates/04_box_locate.py) |
| 5 | **multi-question** | several typed answers from one call | an agent's per-step decisions | [`05_multi_question_agent_step.py`](templates/05_multi_question_agent_step.py) |
| 6 | **Jev Router** | a normal chat completion, routed to the model Jev picks | cost/quality-aware model routing | [`06_jev_router_chat.py`](templates/06_jev_router_chat.py) |
| 7 | **official TypeSafe SDK** | typed `Choice/Noul/Score/Box` objects | if you prefer the vendor SDK | [`07_official_typesafe_sdk.py`](templates/07_official_typesafe_sdk.py) |

## Quick start

```bash
git clone https://github.com/<you>/<your-repo>.git && cd <your-repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your OpenRouter key in OPENROUTER_API_KEY

python templates/01_choice_ticket_triage.py --dry-run   # prints the request, no key needed
python templates/01_choice_ticket_triage.py             # real call
python workloads/run_workloads.py                       # benchmark on sample workloads
pytest -q                                               # offline tests (local mock server)
```

## The API in one screen

`POST https://openrouter.ai/api/v1/systemone`. The same request also works at
`POST https://openrouter.ai/api/alpha/decisions`; set `JEV_ENDPOINT=decisions` to use it.
Send `Authorization: Bearer <OPENROUTER_API_KEY>`.

```json
{
  "model": "typesafe/jev-latest",
  "state": {"subject": "Duplicate charge", "message": "I was charged twice."},
  "questions": {
    "queue":   {"type": "choice", "instructions": "Which queue?", "criteria": {"billing": "Money issues", "technical": null}},
    "urgent":  {"type": "noul",   "instructions": "Reply needed today?", "criteria": {"true": "...", "false": "..."}},
    "anger":   {"type": "score",  "instructions": "How upset?", "criteria": ["calm", "annoyed", "furious"]},
    "logo":    {"type": "box",    "instructions": "The company logo"}
  },
  "assets":     {"photo": {"type": "image", "format": "png", "data_base64": "..."}},
  "session_id": "optional, groups requests in OpenRouter observability"
}
```

The response has the shape below. These are illustrative values, not real Jev output:

```json
{
  "id": "gen-...", "model": "typesafe/jev-...", "provider": "TypeSafe",
  "answers": {
    "queue":  {"type": "choice", "choice": "billing", "confidence": 0.97, "probabilities": {"billing": 0.97, "technical": 0.03}},
    "urgent": {"type": "noul",   "noul": 0.81},
    "anger":  {"type": "score",  "score": 1.3, "confidence": 0.7, "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4}, "legend": {"0": "calm"}},
    "logo":   {"type": "box",    "box": {"top": 0.1, "bottom": 0.3, "left": 0.6, "right": 0.9}, "confidence": 0.88}
  },
  "usage": {"input_tokens": 123, "output_tokens": 0, "cost": 0.0000052}
}
```

Notes:
- Everything that holds content (`state`, `instructions`, each criterion) can be a string, JSON object, or array.
- A `choice` criterion can be `null` when the label explains itself. In `score`, `criteria[i]` describes level *i* (from 0).
- `noul` needs `instructions`, a described outcome (`criteria.true` / `criteria.false`), or both.
- `box` and image state need a vision-capable Jev variant and an `assets` entry. Put the asset key in
  `state` (or anywhere in the questions), and the API swaps the key for the image.
- Model IDs: `typesafe/jev-latest` (alias), or pin a release such as `typesafe/jev-1.14`. Plain IDs
  like `jev-latest` are mapped to the `typesafe/` namespace. The router is `typesafe/jev-router`.

These schemas come from the official SDKs: `openrouter` 1.3.68 (`DecisionsRequest` /
`DecisionsResponse`) and `typesafe-sdk` 0.7.5. When `openrouter` is installed, the offline tests
check every payload against them.

## Using the client in your own code

```python
from jev import JevClient, choice, noul, score

jev = JevClient()                       # reads OPENROUTER_API_KEY / JEV_MODEL from .env
d = jev.decide(
    state={"message": user_msg, "history": last_turns},
    questions={
        "intent":   choice(["question", "complaint", "cancel", "other"]),
        "is_spam":  noul("Is this spam or a scam?"),
        "priority": score(["low", "normal", "high", "critical"]),
    },
)
if d.yes("is_spam") > 0.9: drop()
route(d.choice("intent"), confidence=d.probabilities("intent")[d.choice("intent")])
print(d.score("priority"), d.latency_ms, d.cost)
```

## Patterns that work well

- **Gate before you generate.** Ask cheap `noul`/`choice` questions first, and call a large LLM only when needed (template 2, 5).
- **Use the probabilities.** Set thresholds per risk level, and send low-confidence cases to a human (template 1).
- **Batch your questions.** Ask several questions about one state in a single call. You pay input tokens once (template 5).
- **Prefer structured state.** Passing JSON (order, tier, last messages) works better than one long text blob.
- **Pin a model version in production.** `jev-latest` moves when a new release ships.

## Sample workloads

`workloads/run_workloads.py` runs three labelled datasets from `workloads/datasets/`:

| Workload | Mode | Size | Measures |
|----------|------|------|----------|
| `tickets`   | choice | 20 | accuracy across 5 queues |
| `guardrail` | noul   | 12 | prompt-injection detection at P ≥ 0.5 |
| `severity`  | score  | 8  | rounded expected score == label |

For each workload it prints accuracy, p50/p95 latency, throughput, total cost, and every miss.
Per-item results go to `workloads/results/` (gitignored). Add your own `.jsonl` file with
`{"text", "label"}` rows and an entry in `WORKLOADS` to benchmark your use case.

## Layout

```
jev/client.py          minimal client (requests): question builders, Decision accessors, retries, dry-run
templates/             one runnable example per working mode (all accept --dry-run)
workloads/             benchmark runner + labelled sample datasets
tests/test_offline.py  request/response tests against a local mock (no network, no key)
assets/red_square.png  sample image for the box template
.env.example           every setting, documented
```

## References

- OpenRouter Jev guide: https://openrouter.ai/docs/guides/community/jev
- First decision call tutorial: https://openrouter.ai/docs/guides/community/jev-tutorial
- Jev Router: https://openrouter.ai/typesafe/jev-router
- TypeSafe docs: https://docs.typesafe.ai/
