"""Template 6 - JEV ROUTER: let Jev pick the model + reasoning effort for each chat request.

`typesafe/jev-router` is a regular OpenRouter chat model ID. Before each turn Jev scores the
prompt (difficulty, precision, whether a cheaper model suffices) and forwards it to the
chosen model; the response's `model` field shows which one answered. Optionally restrict the
pool with the `jev-router` plugin (`models` / `excluded_models`, wildcards allowed).

    python templates/06_jev_router_chat.py [--dry-run]
"""

import json
import os

import _common  # noqa: F401
import requests

BASE = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api").rstrip("/")


def chat(prompt: str, allowed: list[str] | None = None) -> dict:
    payload = {
        "model": os.getenv("JEV_ROUTER_MODEL", "typesafe/jev-router"),
        "messages": [{"role": "user", "content": prompt}],
    }
    if allowed:
        payload["plugins"] = [{"id": "jev-router", "models": allowed}]
    if os.getenv("JEV_DRY_RUN") == "1":
        print(f"[dry-run] POST {BASE}/v1/chat/completions\n{json.dumps(payload, indent=2)}")
        return {}
    resp = requests.post(
        f"{BASE}/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    prompts = [
        "What is the capital of Kerala?",  # trivial: expect a small, cheap model
        "Prove that there are infinitely many primes of the form 4k+3.",  # harder: expect a stronger model
    ]
    for p in prompts:
        r = chat(p, allowed=["anthropic/*", "openai/*", "google/*", "meta-llama/*"])
        if r:
            usage = r.get("usage", {})
            print(f"routed to {r.get('model')}  tokens={usage.get('total_tokens')}  cost={usage.get('cost')}")
            print("  " + r["choices"][0]["message"]["content"][:200].replace("\n", " "))
