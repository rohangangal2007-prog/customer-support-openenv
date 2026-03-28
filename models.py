from pydantic import BaseModel, Field
from typing import Any, Optional
from enum import Enum


class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SHIPPING = "shipping"
    RETURNS = "returns"
    GENERAL = "general"


class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Observation(BaseModel):
    ticket_id: str = Field(..., description="Unique ticket identifier")
    customer_message: str = Field(..., description="The customer's support message")
    category: Optional[TicketCategory] = Field(None, description="Ticket category if classified")
    priority: Optional[TicketPriority] = Field(None, description="Ticket priority if assessed")
    conversation_history: list[dict] = Field(default_factory=list, description="Prior conversation turns")
    task_id: str = Field(..., description="Which task is active: task1, task2, task3")
    task_description: str = Field(..., description="What the agent must accomplish")
    step_number: int = Field(0, description="Current step in the episode")
    done: bool = Field(False, description="Whether the episode is complete")
    metadata: dict = Field(default_factory=dict, description="Extra context")


class Action(BaseModel):
    action_type: str = Field(
        ...,
        description="One of: classify, set_priority, draft_reply, escalate, resolve, add_note"
    )
    category: Optional[TicketCategory] = Field(None, description="For classify action")
    priority: Optional[TicketPriority] = Field(None, description="For set_priority action")
    reply_text: Optional[str] = Field(None, description="For draft_reply action")
    note: Optional[str] = Field(None, description="For add_note action")
    resolution_summary: Optional[str] = Field(None, description="For resolve action")
    escalation_reason: Optional[str] = Field(None, description="For escalate action")


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(..., ge=-1.0, le=1.0, description="Reward signal for this step")
    done: bool
    info: dict = Field(default_factory=dict, description="Diagnostic info")


class ResetResult(BaseModel):
    observation: Observation


class StateResult(BaseModel):
    observation: Observation
    episode_reward: float
    steps_taken: int
