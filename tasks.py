"""
Tasks for the Customer Support OpenEnv environment.

Task 1 (Easy):   Classify ticket category + set priority
Task 2 (Medium): Classify + draft a helpful reply
Task 3 (Hard):   Full resolution — classify, prioritise, reply, note, resolve

Each task has 4 ticket variants — one is chosen randomly on reset(),
making the environment genuinely useful for training and evaluating agents.
"""

import random
from dataclasses import dataclass, field
from models import TicketCategory, TicketPriority


@dataclass
class TicketVariant:
    ticket_id: str
    customer_message: str
    expected_category: TicketCategory
    expected_priority: TicketPriority


@dataclass
class Task:
    task_id: str
    description: str
    difficulty: str
    max_steps: int
    tickets: list[TicketVariant]
    good_reply_keywords: list[str] = field(default_factory=list)
    bad_reply_keywords: list[str] = field(default_factory=list)

    def sample_ticket(self, seed: int | None = None) -> TicketVariant:
        """Return a random ticket variant. Pass seed for reproducibility."""
        rng = random.Random(seed)
        return rng.choice(self.tickets)


# ─── Task 1 — Easy: classify + set priority ───────────────────────────────────

TASK1_TICKETS = [
    TicketVariant(
        ticket_id="TKT-101",
        customer_message=(
            "Hi, I was charged twice for my subscription this month. "
            "My bank statement shows two identical charges of $29.99 on March 3rd. "
            "Please fix this immediately, this is very frustrating!"
        ),
        expected_category=TicketCategory.BILLING,
        expected_priority=TicketPriority.HIGH,
    ),
    TicketVariant(
        ticket_id="TKT-102",
        customer_message=(
            "I cancelled my plan last week but I just got charged again. "
            "My cancellation confirmation number is #CC-88421. "
            "I need a refund right away."
        ),
        expected_category=TicketCategory.BILLING,
        expected_priority=TicketPriority.HIGH,
    ),
    TicketVariant(
        ticket_id="TKT-103",
        customer_message=(
            "I can't log into my account — it says my password is wrong but "
            "I just reset it 10 minutes ago. I've tried 3 times and now "
            "I'm locked out. Please help!"
        ),
        expected_category=TicketCategory.TECHNICAL,
        expected_priority=TicketPriority.HIGH,
    ),
    TicketVariant(
        ticket_id="TKT-104",
        customer_message=(
            "Just wanted to ask — do you guys offer a student discount? "
            "I'm a university student and would love to use your service "
            "but the regular price is a bit high for me."
        ),
        expected_category=TicketCategory.GENERAL,
        expected_priority=TicketPriority.LOW,
    ),
]

# ─── Task 2 — Medium: classify + priority + draft reply ───────────────────────

TASK2_TICKETS = [
    TicketVariant(
        ticket_id="TKT-201",
        customer_message=(
            "My order #45892 was supposed to arrive 5 days ago. "
            "The tracking hasn't updated in 3 days and just says 'in transit'. "
            "I need this package for an event this weekend. What's going on?"
        ),
        expected_category=TicketCategory.SHIPPING,
        expected_priority=TicketPriority.HIGH,
    ),
    TicketVariant(
        ticket_id="TKT-202",
        customer_message=(
            "I ordered the blue version of the jacket but received the black one. "
            "Order #78234. I have an important meeting next Friday and specifically "
            "needed the blue. Can you send the correct item urgently?"
        ),
        expected_category=TicketCategory.SHIPPING,
        expected_priority=TicketPriority.HIGH,
    ),
    TicketVariant(
        ticket_id="TKT-203",
        customer_message=(
            "Your app keeps crashing every time I try to open the dashboard. "
            "I'm on iPhone 15, iOS 17.4. This has been happening for 2 days "
            "and I can't access any of my data. This is affecting my work!"
        ),
        expected_category=TicketCategory.TECHNICAL,
        expected_priority=TicketPriority.HIGH,
    ),
    TicketVariant(
        ticket_id="TKT-204",
        customer_message=(
            "I've been waiting 3 weeks for my refund after returning the headphones. "
            "Your policy says 5-7 business days. I have the return tracking number "
            "showing it was delivered to your warehouse 18 days ago."
        ),
        expected_category=TicketCategory.RETURNS,
        expected_priority=TicketPriority.HIGH,
    ),
]

# ─── Task 3 — Hard: full resolution ───────────────────────────────────────────

TASK3_TICKETS = [
    TicketVariant(
        ticket_id="TKT-301",
        customer_message=(
            "I received my laptop stand last week but it's completely broken — "
            "one of the legs snapped off right out of the box. "
            "I paid $89 for this. I want either a full refund or a replacement. "
            "I'm attaching photos. This is unacceptable quality."
        ),
        expected_category=TicketCategory.RETURNS,
        expected_priority=TicketPriority.HIGH,
    ),
    TicketVariant(
        ticket_id="TKT-302",
        customer_message=(
            "The wireless keyboard I bought 3 weeks ago has stopped working entirely. "
            "The battery is full, I've tried re-pairing it multiple times, nothing works. "
            "Cost me $120. I want a replacement sent before the weekend — "
            "I work from home and can't function without it."
        ),
        expected_category=TicketCategory.RETURNS,
        expected_priority=TicketPriority.HIGH,
    ),
    TicketVariant(
        ticket_id="TKT-303",
        customer_message=(
            "I was double-charged $149.99 on my credit card on the 1st and 3rd of this month "
            "for the same annual subscription. I've already contacted my bank but they "
            "said to resolve it with you first. I need this reversed immediately — "
            "that's $150 I can't afford to lose."
        ),
        expected_category=TicketCategory.BILLING,
        expected_priority=TicketPriority.HIGH,
    ),
    TicketVariant(
        ticket_id="TKT-304",
        customer_message=(
            "My account was hacked — someone changed my email and password and made "
            "two purchases totalling $340 that I did not authorise. "
            "I've been locked out for 6 hours. I need my account secured and "
            "those charges reversed NOW. This is an emergency."
        ),
        expected_category=TicketCategory.TECHNICAL,
        expected_priority=TicketPriority.HIGH,
    ),
]


TASKS: dict[str, Task] = {
    "task1": Task(
        task_id="task1",
        difficulty="easy",
        description=(
            "A customer has submitted a support ticket. "
            "Your job: (1) classify the ticket into the correct category "
            "(billing / technical / shipping / returns / general), "
            "and (2) set the appropriate priority level (low / medium / high). "
            "Use the 'classify' action followed by the 'set_priority' action."
        ),
        max_steps=4,
        tickets=TASK1_TICKETS,
    ),
    "task2": Task(
        task_id="task2",
        difficulty="medium",
        description=(
            "A customer needs help with their issue. "
            "Your job: (1) classify the ticket, (2) set priority, "
            "and (3) draft a clear, empathetic reply that addresses their concern. "
            "The reply must mention an estimated timeline and concrete next steps."
        ),
        max_steps=6,
        tickets=TASK2_TICKETS,
        good_reply_keywords=[
            "apologize", "sorry", "investigate", "look into", "timeline",
            "hours", "business day", "carrier", "contact", "update", "resolve",
            "assist", "help", "immediately", "urgently"
        ],
        bad_reply_keywords=[
            "not our fault", "nothing we can do", "wait longer",
            "read the policy", "not responsible"
        ],
    ),
    "task3": Task(
        task_id="task3",
        difficulty="hard",
        description=(
            "A customer has a serious issue requiring full resolution. "
            "You must: (1) classify the ticket, (2) set priority, "
            "(3) draft an empathetic and actionable reply, "
            "(4) add an internal note for the relevant team, "
            "and (5) resolve the ticket with a clear resolution summary. "
            "The resolution summary must mention the specific outcome "
            "(refund, replacement, account secured, etc.)."
        ),
        max_steps=10,
        tickets=TASK3_TICKETS,
        good_reply_keywords=[
            "apologize", "sorry", "replacement", "refund", "return",
            "resolve", "investigate", "prepaid", "label", "reversed",
            "secured", "priority", "immediately", "urgently", "within"
        ],
        bad_reply_keywords=[
            "not eligible", "warranty void", "not our fault",
            "nothing we can do", "read the policy"
        ],
    ),
}


# ─── Graders ──────────────────────────────────────────────────────────────────

def _reply_score(reply: str, task: Task, max_score: float) -> float:
    """
    Score a reply based on good/bad keyword presence.
    Uses proportion of good keywords hit, not raw count,
    so longer keyword lists don't unfairly penalise.
    """
    if not reply or not task.good_reply_keywords:
        return 0.0
    reply_lower = reply.lower()
    hit_ratio = sum(1 for kw in task.good_reply_keywords if kw in reply_lower) / len(task.good_reply_keywords)
    bad_hits = sum(1 for kw in task.bad_reply_keywords if kw in reply_lower)
    score = hit_ratio * max_score
    score -= bad_hits * 0.05
    return round(max(0.0, min(score, max_score)), 4)


def _category_match(state: dict, ticket: TicketVariant) -> bool:
    return state.get("category") == ticket.expected_category


def _priority_match(state: dict, ticket: TicketVariant) -> bool:
    return state.get("priority") == ticket.expected_priority


def grade_task1(state: dict) -> float:
    """
    Score = 0.0–1.0
    Correct category: +0.5  |  Correct priority: +0.5
    """
    ticket = state.get("_ticket")
    if ticket is None:
        return 0.0
    score = 0.0
    if _category_match(state, ticket): score += 0.5
    if _priority_match(state, ticket): score += 0.5
    return round(score, 2)


def grade_task2(state: dict) -> float:
    """
    Score = 0.0–1.0
    Category: +0.2  |  Priority: +0.2  |  Reply exists: +0.1
    Reply quality (keyword ratio): up to +0.3  |  No bad phrases: +0.2
    """
    ticket = state.get("_ticket")
    if ticket is None:
        return 0.0
    score = 0.0
    task = TASKS["task2"]
    if _category_match(state, ticket): score += 0.2
    if _priority_match(state, ticket): score += 0.2
    reply = (state.get("last_reply") or "").lower()
    if reply:
        score += 0.1
        score += _reply_score(reply, task, max_score=0.3)
        bad_hits = sum(1 for kw in task.bad_reply_keywords if kw in reply)
        if bad_hits == 0: score += 0.2
    return round(min(score, 1.0), 2)


def grade_task3(state: dict) -> float:
    """
    Score = 0.0–1.0
    Category: +0.12  |  Priority: +0.12  |  Reply quality: up to +0.26
    Internal note: +0.1  |  Resolution exists: +0.15  |  Outcome mentioned: +0.25
    """
    ticket = state.get("_ticket")
    if ticket is None:
        return 0.0
    score = 0.0
    task = TASKS["task3"]
    if _category_match(state, ticket): score += 0.12
    if _priority_match(state, ticket): score += 0.12
    reply = (state.get("last_reply") or "").lower()
    if reply:
        score += _reply_score(reply, task, max_score=0.26)
    if state.get("internal_note") and len(state["internal_note"]) > 10:
        score += 0.10
    resolution = (state.get("resolution_summary") or "").lower()
    if resolution:
        score += 0.15
        outcome_words = ["refund", "replacement", "replaced", "refunded",
                         "secured", "reversed", "resolved", "credited"]
        if any(w in resolution for w in outcome_words):
            score += 0.25
    return round(min(score, 1.0), 2)


GRADERS = {
    "task1": grade_task1,
    "task2": grade_task2,
    "task3": grade_task3,
}
