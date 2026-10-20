"""Minimal client for Jev (TypeSafe's System One model) on OpenRouter.

Jev does not generate text. You send it `state` (the content to evaluate) plus named, typed
`questions`, and it returns one typed answer per question, with probabilities attached:

    choice  -> picks one label from `criteria`, with a probability per label
    noul    -> yes/no; the answer is the probability of "yes" (0..1)
    score   -> position on an ordered rubric; the answer is the expected score (0..len-1)
    box     -> bounding box (fractions of image height/width) of something in an image

Wire format taken from the official OpenRouter SDK (`DecisionsRequest`/`DecisionsResponse`) and
the TypeSafe SDK (`SystemOneRequest`).
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

DEFAULT_BASE_URL = "https://openrouter.ai/api"
DEFAULT_MODEL = "typesafe/jev-latest"

# OpenRouter exposes Jev on two equivalent endpoints (relative to the base URL):
#   /v1/systemone     - System One endpoint, compatible with the TypeSafe SDKs
#   /alpha/decisions  - Decisions API (alpha)
ENDPOINTS = {"systemone": "/v1/systemone", "decisions": "/alpha/decisions"}


# --------------------------------------------------------------------------------------------
# Question builders. Each returns the plain JSON dict the API expects.
# "Content" anywhere below may be a string, a JSON object, or a JSON array.
# --------------------------------------------------------------------------------------------


def choice(criteria: dict[str, Any] | list[str], instructions: Any = None) -> dict[str, Any]:
    """Pick exactly one label. `criteria` maps label -> description (or None for no description)."""
    if isinstance(criteria, list):
        criteria = {label: None for label in criteria}
    if not criteria:
        raise ValueError("choice needs at least one label")
    q: dict[str, Any] = {"type": "choice", "criteria": criteria}
    if instructions is not None:
        q["instructions"] = instructions
    return q


def noul(instructions: Any = None, *, true: Any = None, false: Any = None) -> dict[str, Any]:
    """Yes/no question. Optionally describe what counts as yes (`true`) and no (`false`)."""
    if instructions is None and true is None and false is None:
        raise ValueError("noul needs instructions or a described outcome")
    q: dict[str, Any] = {"type": "noul"}
    if instructions is not None:
        q["instructions"] = instructions
    if true is not None or false is not None:
        q["criteria"] = {"true": true, "false": false}
    return q


def score(criteria: list[Any], instructions: Any = None) -> dict[str, Any]:
    """Ordered rubric. `criteria[i]` describes score i (0 = lowest)."""
    if not criteria:
        raise ValueError("score needs at least one rubric level")
    q: dict[str, Any] = {"type": "score", "criteria": list(criteria)}
    if instructions is not None:
        q["instructions"] = instructions
    return q


def box(instructions: Any) -> dict[str, Any]:
    """Locate what `instructions` describes in the request's image (needs a vision model)."""
    return {"type": "box", "instructions": instructions}


def image_asset(path: str | Path) -> dict[str, Any]:
    """Inline image asset (png/jpg). Reference it by its key anywhere in state or questions."""
    path = Path(path)
    fmt = path.suffix.lower().lstrip(".")
    if fmt not in {"png", "jpg", "jpeg"}:
        raise ValueError(f"unsupported image format: {fmt}")
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image", "format": fmt, "data_base64": data}


# --------------------------------------------------------------------------------------------
# Response
# --------------------------------------------------------------------------------------------


@dataclass
class Decision:
    answers: dict[str, dict[str, Any]]
    model: str
    usage: dict[str, Any]
    latency_ms: float
    id: str | None = None
    provider: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def choice(self, name: str) -> str:
        return self._get(name, "choice")["choice"]

    def probabilities(self, name: str) -> dict[str, float]:
        return self.answers[name].get("probabilities") or {}

    def confidence(self, name: str) -> float | None:
        return self.answers[name].get("confidence")

    def yes(self, name: str) -> float:
        """Probability (0..1) that the answer to a noul question is yes."""
        return float(self._get(name, "noul")["noul"])

    def score(self, name: str) -> float:
        """Expected score (probability-weighted) on the rubric, 0..len(criteria)-1."""
        return float(self._get(name, "score")["score"])

    def box(self, name: str) -> dict[str, float]:
        return self._get(name, "box")["box"]

    @property
    def cost(self) -> float:
        return float(self.usage.get("cost") or 0.0)

    def _get(self, name: str, kind: str) -> dict[str, Any]:
        answer = self.answers[name]
        if answer.get("type") != kind:
            raise TypeError(f"answer {name!r} is {answer.get('type')!r}, not {kind!r}")
        return answer


class JevError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


# --------------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------------


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 2,
        dry_run: bool | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.getenv("JEV_MODEL") or DEFAULT_MODEL
        endpoint = endpoint or os.getenv("JEV_ENDPOINT") or "systemone"
        self.path = ENDPOINTS.get(endpoint, endpoint)
        self.timeout = timeout
        self.max_retries = max_retries
        self.dry_run = os.getenv("JEV_DRY_RUN") == "1" if dry_run is None else dry_run
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if self.api_key:
            self.session.headers["Authorization"] = f"Bearer {self.api_key}"
        # Optional app attribution headers shown on your OpenRouter dashboard.
        if os.getenv("OPENROUTER_APP_URL"):
            self.session.headers["HTTP-Referer"] = os.environ["OPENROUTER_APP_URL"]
        if os.getenv("OPENROUTER_APP_TITLE"):
            self.session.headers["X-OpenRouter-Title"] = os.environ["OPENROUTER_APP_TITLE"]
        if not self.api_key and not self.dry_run:
            raise RuntimeError("Set OPENROUTER_API_KEY (copy .env.example to .env) or pass api_key=...")

    def build_payload(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        *,
        assets: dict[str, dict[str, Any]] | None = None,
        model: str | None = None,
        session_id: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        if not questions:
            raise ValueError("at least one question is required")
        payload: dict[str, Any] = {"model": model or self.model, "state": state, "questions": questions}
        if assets:
            payload["assets"] = assets
        if session_id:
            payload["session_id"] = session_id
        payload.update(extra)
        return payload

    def decide(self, state: Any, questions: dict[str, dict[str, Any]], **kwargs: Any) -> Decision:
        """Send one System One request and return the typed answers."""
        payload = self.build_payload(state, questions, **kwargs)
        if self.dry_run:
            printable = json.loads(json.dumps(payload))
            for asset in (printable.get("assets") or {}).values():
                if "data_base64" in asset:
                    asset["data_base64"] = f"<{len(asset['data_base64'])} base64 chars>"
            print(f"[dry-run] POST {self.base_url}{self.path}\n{json.dumps(printable, indent=2)}")
            return Decision(answers={}, model=payload["model"], usage={}, latency_ms=0.0)

        url = self.base_url + self.path
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            latency_ms = (time.perf_counter() - start) * 1000
            retryable = resp.status_code == 429 or resp.status_code >= 500
            if resp.ok or not retryable or attempt == self.max_retries:
                break
            time.sleep(float(resp.headers.get("retry-after") or 0.5 * 2**attempt))
        if not resp.ok:
            raise JevError(resp.status_code, resp.text)
        body = resp.json()
        return Decision(
            answers=body.get("answers", {}),
            model=body.get("model", payload["model"]),
            usage=body.get("usage", {}),
            latency_ms=latency_ms,
            id=body.get("id"),
            provider=body.get("provider"),
            raw=body,
        )
