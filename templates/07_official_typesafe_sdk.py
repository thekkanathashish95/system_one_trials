"""Template 7 - the official TypeSafe Python SDK, pointed at OpenRouter.

OpenRouter's /api/v1/systemone endpoint is compatible with the TypeSafe SDKs, so you can use
typed question objects (Choice / Noul / Score / Box) and typed answers without this repo's
client. Configure via env (see .env.example):

    TYPESAFE_API_KEY=<your OpenRouter key>
    TYPESAFE_BASE_URL=https://openrouter.ai/api      # SDK appends /v1/systemone
    TYPESAFE_DEFAULT_MODEL=typesafe/jev-latest

    pip install typesafe-sdk
    python templates/07_official_typesafe_sdk.py
"""

import _common  # noqa: F401  (loads .env)

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient() as client:
    response = client.system_one(
        state={"subject": "Duplicate charge", "message": "I was charged twice. Please fix this ASAP."},
        questions={
            "category": Choice(
                instructions="What is this ticket about?",
                criteria={"billing": None, "technical": None, "other": None},
            ),
            "urgent": Noul(instructions="Does the customer need a reply today?"),
            "sentiment": Score(
                instructions="How negative is the customer?",
                criteria=["calm", "mildly annoyed", "frustrated", "furious"],
            ),
        },
    )

print("category :", response.choices["category"].choice, response.choices["category"].probabilities)
print("urgent   :", response.nouls["urgent"].noul)
print("sentiment:", response.scores["sentiment"].score, response.scores["sentiment"].probabilities)
print("usage    :", response.usage)
