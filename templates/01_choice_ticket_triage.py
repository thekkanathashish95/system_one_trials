"""Template 1 - CHOICE: route a support ticket to exactly one queue.

Use choice when the answer is one label out of a fixed set (routing, intent detection,
classification, picking a tool). You get the label plus a probability for every label, so you
can fall back to a human when the top probability is low.

    python templates/01_choice_ticket_triage.py [--dry-run]
"""

import _common  # noqa: F401  (path + --dry-run setup)

from jev import JevClient, choice

QUEUES = {
    "billing": "Charges, refunds, invoices, payment methods, plan pricing.",
    "technical": "Bugs, errors, outages, integration or API problems.",
    "account": "Login, password reset, profile, deleting or merging accounts.",
    "sales": "Wants to buy, upgrade, get a quote, or talk to sales.",
    "other": "Anything that fits none of the above.",
}

CONFIDENCE_FLOOR = 0.70  # below this, send to a human triager


def triage(client: JevClient, ticket: dict) -> tuple[str, float]:
    decision = client.decide(
        state=ticket,  # structured state works: Jev reads the JSON as context
        questions={"queue": choice(QUEUES, instructions="Which support queue should handle this ticket?")},
    )
    if not decision.answers:  # dry run
        return "(dry-run)", 0.0
    label = decision.choice("queue")
    return label, decision.probabilities("queue").get(label, 0.0)


if __name__ == "__main__":
    client = JevClient()
    tickets = [
        {"subject": "Charged twice", "body": "My card was billed two times this month. Please refund one."},
        {"subject": "500 on /export", "body": "The CSV export endpoint returns HTTP 500 since this morning."},
        {"subject": "Can't sign in", "body": "Password reset email never arrives."},
        {"subject": "Team plan", "body": "We have 40 seats, can we get annual pricing?"},
    ]
    for t in tickets:
        label, p = triage(client, t)
        route = label if p >= CONFIDENCE_FLOOR else f"HUMAN (best guess {label})"
        print(f"{t['subject']:<16} -> {route:<28} p={_common.pct(p)}")
