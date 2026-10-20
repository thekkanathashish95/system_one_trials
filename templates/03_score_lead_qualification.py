"""Template 3 - SCORE: grade something on an ordered rubric.

Use score when the labels are ordered (severity, quality, urgency, fit). `criteria[i]`
describes level i. The answer is the *expected* score (probability-weighted, so 2.6 means
"between 2 and 3, leaning 3") plus the full distribution, which is far more useful for
ranking and thresholds than a single integer.

    python templates/03_score_lead_qualification.py [--dry-run]
"""

import _common  # noqa: F401

from jev import JevClient, score

RUBRIC = [
    "0 - Not a fit: student, competitor, spam, or no business need.",
    "1 - Weak: vague interest, no budget or timeline mentioned.",
    "2 - Moderate: real problem we solve, but small team or no timeline.",
    "3 - Strong: clear need, decision-maker or budget mentioned.",
    "4 - Hot: budget, timeline within a quarter, and decision-maker all present.",
]


def qualify(client: JevClient, lead: dict) -> tuple[float, dict]:
    d = client.decide(
        state=lead,
        questions={"fit": score(RUBRIC, instructions="How qualified is this inbound B2B lead for our analytics SaaS?")},
    )
    return (d.score("fit"), d.probabilities("fit")) if d.answers else (0.0, {})


if __name__ == "__main__":
    client = JevClient()
    leads = [
        {"title": "VP Data, 2,000-person retailer", "note": "Budget approved, need a tool live before Q1 close."},
        {"title": "Student", "note": "Doing a thesis on dashboards, can I get a free licence?"},
        {"title": "Ops lead, 30-person startup", "note": "Spreadsheets are breaking, exploring options for next year."},
    ]
    ranked = sorted(((qualify(client, l), l) for l in leads), key=lambda x: -x[0][0])
    for (expected, dist), lead in ranked:
        print(f"{expected:4.2f}/4  {lead['title']:<34} dist={dist}")
