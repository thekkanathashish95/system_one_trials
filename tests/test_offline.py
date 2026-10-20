"""Offline tests: run the client against a local mock of the System One endpoint.

These check request shape and response parsing only - they say nothing about Jev's answer
quality. For real numbers run workloads/run_workloads.py with an OpenRouter key.

If the official `openrouter` SDK is installed, payloads are also validated against its
`DecisionsRequest` model and mock responses against `DecisionsResponse`.
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jev import JevClient, JevError, box, choice, image_asset, noul, score  # noqa: E402

try:
    from openrouter.components import DecisionsRequest, DecisionsResponse
except ImportError:
    DecisionsRequest = DecisionsResponse = None


def fake_answer(q: dict) -> dict:
    if q["type"] == "choice":
        labels = list(q["criteria"])
        probs = {label: (0.9 if i == 0 else 0.1 / max(len(labels) - 1, 1)) for i, label in enumerate(labels)}
        return {"type": "choice", "choice": labels[0], "probabilities": probs, "confidence": 0.9}
    if q["type"] == "noul":
        return {"type": "noul", "noul": 0.83}
    if q["type"] == "score":
        n = len(q["criteria"])
        return {"type": "score", "score": n - 1.4, "confidence": 0.6, "probabilities": {str(i): 1 / n for i in range(n)},
                "legend": {str(i): c for i, c in enumerate(q["criteria"])}}
    if q["type"] == "box":
        return {"type": "box", "box": {"top": 0.25, "bottom": 0.625, "left": 0.375, "right": 0.75}, "confidence": 0.7}
    raise ValueError(q["type"])


class Handler(BaseHTTPRequestHandler):
    requests: list = []
    fail_next: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
        if Handler.fail_next:
            status = Handler.fail_next.pop(0)
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"nope"}}')
            return
        resp = {
            "id": "gen-test",
            "model": body["model"],
            "provider": "TypeSafe",
            "answers": {name: fake_answer(q) for name, q in body["questions"].items()},
            "usage": {"input_tokens": 42, "output_tokens": 0, "cost": 0.0000018},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def client():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    Handler.requests.clear()
    Handler.fail_next.clear()
    yield JevClient("sk-test", base_url=f"http://127.0.0.1:{server.server_port}/api", model="typesafe/jev-latest",
                    dry_run=False)
    server.shutdown()


def last_body():
    return Handler.requests[-1]["body"]


def test_all_question_types_roundtrip(client):
    d = client.decide(
        state={"ticket": "charged twice"},
        questions={
            "queue": choice({"billing": "money", "other": None}, instructions="Which queue?"),
            "urgent": noul("Is it urgent?", true="needs reply today", false="can wait"),
            "sev": score(["low", "mid", "high"], instructions="Severity?"),
        },
        session_id="s-1",
    )
    req = Handler.requests[-1]
    assert req["path"] == "/api/v1/systemone"
    assert req["headers"]["Authorization"] == "Bearer sk-test"
    body = req["body"]
    assert body["model"] == "typesafe/jev-latest"
    assert body["session_id"] == "s-1"
    assert body["questions"]["urgent"] == {"type": "noul", "instructions": "Is it urgent?",
                                           "criteria": {"true": "needs reply today", "false": "can wait"}}
    assert body["questions"]["sev"]["criteria"] == ["low", "mid", "high"]

    assert d.choice("queue") == "billing"
    assert d.probabilities("queue")["billing"] == pytest.approx(0.9)
    assert d.yes("urgent") == pytest.approx(0.83)
    assert d.score("sev") == pytest.approx(1.6)
    assert d.cost > 0 and d.latency_ms > 0 and d.id == "gen-test"
    with pytest.raises(TypeError):
        d.yes("queue")

    if DecisionsRequest is not None:
        DecisionsRequest.model_validate(body)
        DecisionsResponse.model_validate(d.raw)


def test_box_with_image_asset(client):
    d = client.decide(state="photo", assets={"photo": image_asset(ROOT / "assets" / "red_square.png")},
                      questions={"target": box("The red square")})
    asset = last_body()["assets"]["photo"]
    assert asset["type"] == "image" and asset["format"] == "png" and len(asset["data_base64"]) > 100
    assert d.box("target")["top"] == pytest.approx(0.25)


def test_decisions_endpoint_and_retry(client):
    client.path = "/alpha/decisions"
    Handler.fail_next.extend([503])
    d = client.decide(state="hi", questions={"q": noul("Greeting?")})
    assert [r["path"] for r in Handler.requests] == ["/api/alpha/decisions"] * 2
    assert d.yes("q") > 0


def test_client_error_not_retried(client):
    Handler.fail_next.extend([400])
    with pytest.raises(JevError) as e:
        client.decide(state="hi", questions={"q": noul("Greeting?")})
    assert e.value.status == 400 and len(Handler.requests) == 1


def test_builders_validate():
    assert choice(["a", "b"])["criteria"] == {"a": None, "b": None}
    with pytest.raises(ValueError):
        choice({})
    with pytest.raises(ValueError):
        noul()
    with pytest.raises(ValueError):
        score([])
    assert noul(true="yes means spam") == {"type": "noul", "criteria": {"true": "yes means spam", "false": None}}


def test_workload_runner_against_mock(client, monkeypatch):
    sys.path.insert(0, str(ROOT / "workloads"))
    import run_workloads

    monkeypatch.setattr(run_workloads, "HERE", ROOT / "workloads")
    summary = run_workloads.run(client, "tickets", concurrency=4)
    assert summary["n"] == 20 and summary["errors"] == 0
    Path(ROOT / summary["results_file"]).unlink()
