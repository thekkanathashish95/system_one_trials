"""Template 5 - MULTI-QUESTION: one call, many typed decisions about the same state.

This is the main System One pattern for agents: before (or instead of) asking a big LLM
"what next?", ask Jev several typed questions about the current app state in one request.
All questions are answered together for one input-token bill (output tokens are free).

    python templates/05_multi_question_agent_step.py [--dry-run]
"""

import _common  # noqa: F401

from jev import JevClient, choice, noul, score

QUESTIONS = {
    "next_action": choice(
        {
            "reply": "Answer the customer directly from the knowledge base.",
            "lookup_order": "Need to fetch order details before answering.",
            "refund": "Issue a refund (policy allows it).",
            "escalate": "Hand off to a human agent.",
        },
        instructions="What should the support agent do next?",
    ),
    "angry": noul("Is the customer angry or threatening to leave?"),
    "urgency": score(["0 - can wait days", "1 - today", "2 - within the hour", "3 - immediately"],
                     instructions="How urgent is a response?"),
    "task_done": noul("Has the customer's issue already been fully resolved in this conversation?"),
}


def step(client: JevClient, state: dict) -> dict:
    d = client.decide(state=state, questions=QUESTIONS, session_id=state.get("conversation_id"))
    if not d.answers:
        return {}
    return {
        "next_action": d.choice("next_action"),
        "p_angry": round(d.yes("angry"), 3),
        "urgency": round(d.score("urgency"), 2),
        "p_done": round(d.yes("task_done"), 3),
        "latency_ms": round(d.latency_ms),
        "cost_usd": d.cost,
    }


if __name__ == "__main__":
    client = JevClient()
    state = {
        "conversation_id": "conv-123",
        "customer_tier": "gold",
        "order": {"id": "A-998", "status": "delivered", "days_since_delivery": 3},
        "messages": [
            {"role": "customer", "text": "The blender arrived cracked. Third time ordering from you and this happens."},
            {"role": "agent", "text": "Sorry to hear that! Could you share a photo?"},
            {"role": "customer", "text": "Sent. I want my money back today or I'm disputing the charge."},
        ],
    }
    print(step(client, state))
