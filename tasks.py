"""
Tasks for the Customer Support OpenEnv environment.

Task 1 (Easy):   Classify ticket category + set priority
Task 2 (Medium): Classify + draft a helpful reply
Task 3 (Hard):   Full resolution — classify, prioritise, reply, resolve with summary
"""

from dataclasses import dataclass, field
from typing import Optional
from models import TicketCategory, TicketPriority


@dataclass
class Task:
    task_id: str
    description: str
    difficulty: str
    max_steps: int
    ticket: dict
    expected_category: TicketCategory
    expected_priority: TicketPriority
    good_reply_keywords: list[str] = field(default_factory=list)
    bad_reply_keywords: list[str] = field(default_factory=list)


TASKS: dict[str, Task] = {
    "task1": Task(
        task_id="task1",
        difficulty="easy",
        description=(
            "A customer has submitted a support ticket. "
            "Your job: (1) classify the ticket into the correct category, "
            "and (2) set the appropriate priority level. "
            "Use the 'classify' action followed by the 'set_priority' action."
        ),
        max_steps=4,
        ticket={
            "ticket_id": "TKT-001",
            "customer_message": (
                "Hi, I was charged twice for my subscription this month. "
                "My bank statement shows two identical charges of $29.99 on March 3rd. "
                "Please fix this immediately, this is very frustrating!"
            ),
        },
        expected_category=TicketCategory.BILLING,
        expected_priority=TicketPriority.HIGH,
        good_reply_keywords=[],
        bad_reply_keywords=[],
    ),

    "task2": Task(
        task_id="task2",
        difficulty="medium",
        description=(
            "A customer needs help with their order. "
            "Your job: (1) classify the ticket, (2) set priority, "
            "and (3) draft a clear, empathetic reply that addresses their concern. "
            "The reply must mention an estimated timeline and next steps."
        ),
        max_steps=6,
        ticket={
            "ticket_id": "TKT-002",
            "customer_message": (
                "My order #45892 was supposed to arrive 5 days ago. "
                "The tracking hasn't updated in 3 days and just says 'in transit'. "
                "I need this package for an event this weekend. What's going on?"
            ),
        },
        expected_category=TicketCategory.SHIPPING,
        expected_priority=TicketPriority.HIGH,
        good_reply_keywords=[
            "apologize", "sorry", "investigate", "tracking", "timeline",
            "48 hours", "24 hours", "carrier", "contact", "update"
        ],
        bad_reply_keywords=[
            "not our fault", "nothing we can do", "wait longer", "policy"
        ],
    ),

    "task3": Task(
        task_id="task3",
        difficulty="hard",
        description=(
            "A customer wants to return a defective product. "
            "You must fully resolve this ticket by: "
            "(1) classifying it, (2) setting priority, "
            "(3) drafting an empathetic and actionable reply, "
            "(4) adding an internal note for the warehouse team, "
            "and (5) resolving the ticket with a clear resolution summary. "
            "The resolution summary must mention refund or replacement."
        ),
        max_steps=10,
        ticket={
            "ticket_id": "TKT-003",
            "customer_message": (
                "I received my laptop stand last week but it's completely broken — "
                "one of the legs snapped off right out of the box. "
                "I paid $89 for this. I want either a full refund or a replacement. "
                "I'm attaching photos. This is unacceptable quality."
            ),
        },
        expected_category=TicketCategory.RETURNS,
        expected_priority=TicketPriority.HIGH,
        good_reply_keywords=[
            "apologize", "sorry", "replacement", "refund", "return",
            "defective", "quality", "photo", "prepaid", "label"
        ],
        bad_reply_keywords=[
            "policy", "cannot", "unfortunately", "not eligible", "warranty void"
        ],
    ),
}


# ─── Graders ──────────────────────────────────────────────────────────────────

def grade_task1(state: dict) -> float:
    """
    Score = 0.0–1.0
    - Correct category: +0.5
    - Correct priority: +0.5
    Partial credit: wrong priority but correct category = 0.5
    """
    score = 0.0
    task = TASKS["task1"]

    if state.get("category") == task.expected_category:
        score += 0.5
    if state.get("priority") == task.expected_priority:
        score += 0.5

    return round(score, 2)


def grade_task2(state: dict) -> float:
    """
    Score = 0.0–1.0
    - Correct category: +0.2
    - Correct priority: +0.2
    - Reply drafted: +0.2
    - Reply contains good keywords (each up to 0.2 total): +0.2
    - Reply avoids bad keywords: +0.2
    """
    score = 0.0
    task = TASKS["task2"]

    if state.get("category") == task.expected_category:
        score += 0.2
    if state.get("priority") == task.expected_priority:
        score += 0.2

    reply = (state.get("last_reply") or "").lower()
    if reply:
        score += 0.2
        # Good keywords
        hits = sum(1 for kw in task.good_reply_keywords if kw in reply)
        score += min(0.2, hits * 0.04)
        # Penalise bad keywords
        bad_hits = sum(1 for kw in task.bad_reply_keywords if kw in reply)
        if bad_hits == 0:
            score += 0.2

    return round(min(score, 1.0), 2)


def grade_task3(state: dict) -> float:
    """
    Score = 0.0–1.0
    - Correct category: +0.15
    - Correct priority: +0.15
    - Reply drafted:    +0.15
    - Good keywords in reply: up to +0.15
    - Internal note added: +0.1
    - Resolved with summary: +0.15
    - Summary mentions refund/replacement: +0.15
    """
    score = 0.0
    task = TASKS["task3"]

    if state.get("category") == task.expected_category:
        score += 0.15
    if state.get("priority") == task.expected_priority:
        score += 0.15

    reply = (state.get("last_reply") or "").lower()
    if reply:
        score += 0.15
        hits = sum(1 for kw in task.good_reply_keywords if kw in reply)
        score += min(0.15, hits * 0.025)

    if state.get("internal_note"):
        score += 0.10

    resolution = (state.get("resolution_summary") or "").lower()
    if resolution:
        score += 0.15
        if "refund" in resolution or "replacement" in resolution:
            score += 0.15

    return round(min(score, 1.0), 2)


GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}
