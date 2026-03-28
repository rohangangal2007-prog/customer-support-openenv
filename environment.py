"""
CustomerSupportEnv — OpenEnv-compliant reinforcement learning environment.

Simulates a real-world customer support agent workflow.
An AI agent processes support tickets by taking structured actions
and receives partial reward signals throughout the episode.
"""

import copy
from typing import Optional
from models import (
    Observation, Action, StepResult, ResetResult, StateResult,
    TicketCategory, TicketPriority
)
from tasks import TASKS, GRADERS


class CustomerSupportEnv:
    """
    OpenEnv-compliant customer support environment.

    The agent must handle a support ticket by taking a sequence of actions:
      classify → set_priority → draft_reply → add_note → resolve / escalate

    Rewards are given at each step for partial progress, not just at the end.
    """

    VALID_ACTIONS = {"classify", "set_priority", "draft_reply", "escalate", "resolve", "add_note"}

    def __init__(self, task_id: str = "task1"):
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS.keys())}")
        self.task_id = task_id
        self._state: dict = {}
        self._episode_reward: float = 0.0
        self._steps: int = 0
        self._done: bool = False
        self._initialized: bool = False

    # ─── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self) -> ResetResult:
        """Reset the environment and return the initial observation."""
        task = TASKS[self.task_id]
        self._state = {
            "task_id": self.task_id,
            "ticket_id": task.ticket["ticket_id"],
            "customer_message": task.ticket["customer_message"],
            "category": None,
            "priority": None,
            "last_reply": None,
            "internal_note": None,
            "resolution_summary": None,
            "conversation_history": [],
            "step_number": 0,
            "done": False,
        }
        self._episode_reward = 0.0
        self._steps = 0
        self._done = False
        self._initialized = True

        obs = self._build_observation()
        return ResetResult(observation=obs)

    def step(self, action: Action) -> StepResult:
        """Apply an action and return (observation, reward, done, info)."""
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        task = TASKS[self.task_id]
        self._steps += 1
        reward = 0.0
        info: dict = {"action_type": action.action_type, "valid": True}

        # ── Validate action type ──
        if action.action_type not in self.VALID_ACTIONS:
            reward = -0.1
            info["valid"] = False
            info["error"] = f"Unknown action '{action.action_type}'"
            obs = self._build_observation()
            return StepResult(observation=obs, reward=reward, done=self._done, info=info)

        # ── Apply action and compute step reward ──
        if action.action_type == "classify":
            if action.category is None:
                reward = -0.05
                info["error"] = "classify action requires 'category' field"
            else:
                self._state["category"] = action.category
                if action.category == task.expected_category:
                    reward = 0.2
                    info["feedback"] = "Correct category!"
                else:
                    reward = -0.05
                    info["feedback"] = "Incorrect category."

        elif action.action_type == "set_priority":
            if action.priority is None:
                reward = -0.05
                info["error"] = "set_priority action requires 'priority' field"
            else:
                self._state["priority"] = action.priority
                if action.priority == task.expected_priority:
                    reward = 0.2
                    info["feedback"] = "Correct priority!"
                else:
                    reward = -0.05
                    info["feedback"] = "Incorrect priority."

        elif action.action_type == "draft_reply":
            if not action.reply_text or len(action.reply_text.strip()) < 20:
                reward = -0.1
                info["error"] = "Reply must be at least 20 characters."
            else:
                self._state["last_reply"] = action.reply_text
                self._state["conversation_history"].append({
                    "role": "agent",
                    "content": action.reply_text
                })
                reward = self._score_reply(action.reply_text, task)
                info["feedback"] = f"Reply scored: {reward:.2f}"

        elif action.action_type == "add_note":
            if not action.note or len(action.note.strip()) < 5:
                reward = -0.05
                info["error"] = "Note is too short."
            else:
                self._state["internal_note"] = action.note
                reward = 0.1
                info["feedback"] = "Internal note saved."

        elif action.action_type == "escalate":
            self._state["done"] = True
            self._done = True
            reason = action.escalation_reason or ""
            reward = 0.05 if reason else -0.05
            info["feedback"] = "Ticket escalated."

        elif action.action_type == "resolve":
            if not action.resolution_summary or len(action.resolution_summary.strip()) < 20:
                reward = -0.1
                info["error"] = "resolution_summary must be at least 20 characters."
            else:
                self._state["resolution_summary"] = action.resolution_summary
                # Final grader score
                final_score = GRADERS[self.task_id](self._state)
                reward = final_score
                self._state["done"] = True
                self._done = True
                info["final_score"] = final_score
                info["feedback"] = f"Ticket resolved. Final score: {final_score:.2f}"

        # ── Max steps guard ──
        if self._steps >= task.max_steps and not self._done:
            self._done = True
            self._state["done"] = True
            reward += -0.1  # small penalty for not finishing in time
            info["feedback"] = info.get("feedback", "") + " | Max steps reached."

        self._episode_reward += reward
        self._state["step_number"] = self._steps
        obs = self._build_observation()
        return StepResult(observation=obs, reward=round(reward, 4), done=self._done, info=info)

    def state(self) -> StateResult:
        """Return the current environment state without advancing it."""
        if not self._initialized:
            raise RuntimeError("Call reset() before state().")
        obs = self._build_observation()
        return StateResult(
            observation=obs,
            episode_reward=round(self._episode_reward, 4),
            steps_taken=self._steps,
        )

    # ─── Internal helpers ─────────────────────────────────────────────────────

    def _build_observation(self) -> Observation:
        s = self._state
        task = TASKS[self.task_id]
        return Observation(
            ticket_id=s["ticket_id"],
            customer_message=s["customer_message"],
            category=s.get("category"),
            priority=s.get("priority"),
            conversation_history=copy.deepcopy(s["conversation_history"]),
            task_id=self.task_id,
            task_description=task.description,
            step_number=s["step_number"],
            done=s["done"],
            metadata={
                "difficulty": task.difficulty,
                "max_steps": task.max_steps,
                "has_reply": s.get("last_reply") is not None,
                "has_note": s.get("internal_note") is not None,
                "is_resolved": s.get("resolution_summary") is not None,
            }
        )

    def _score_reply(self, reply: str, task) -> float:
        """Partial reward for reply quality based on keywords."""
        if not task.good_reply_keywords:
            return 0.15  # Task 1 doesn't need replies
        reply_lower = reply.lower()
        hits = sum(1 for kw in task.good_reply_keywords if kw in reply_lower)
        bad_hits = sum(1 for kw in task.bad_reply_keywords if kw in reply_lower)
        score = min(0.25, hits * 0.04)
        score -= bad_hits * 0.05
        return round(max(0.0, score), 4)
