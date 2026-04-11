#!/usr/bin/env python3
"""
Baseline inference runner for CustomerSupport OpenEnv.

This script intentionally emits only the structured stdout lines required by
the evaluator:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

import requests

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or ""
ENV_URL = os.getenv("ENV_URL") or "http://localhost:7860"
BENCHMARK = "customer-support-openenv"
TASKS = ["task1", "task2", "task3"]
SEED = 42
TEMPERATURE = 0.0
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.50

SYSTEM_PROMPT = """You are an expert customer support agent AI.
Return exactly one JSON object with one of these forms:
{"action_type":"classify","category":"billing|technical|shipping|returns|general"}
{"action_type":"set_priority","priority":"low|medium|high"}
{"action_type":"draft_reply","reply_text":"..."}
{"action_type":"add_note","note":"..."}
{"action_type":"resolve","resolution_summary":"..."}
{"action_type":"escalate","escalation_reason":"..."}

Rules:
- Output only valid JSON, no markdown or explanation.
- For task1: classify then set_priority then resolve.
- For task2: classify then set_priority then draft_reply then resolve.
- For task3: classify then set_priority then draft_reply then add_note then resolve.
- Resolution must mention refund or replacement when relevant.
"""


def _debug(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = (error or "null").replace("\n", " ")
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def call_env(method: str, path: str, payload=None, params=None):
    url = ENV_URL.rstrip("/") + path
    try:
        if method == "POST":
            r = requests.post(url, json=payload, params=params, timeout=15)
        else:
            r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        _debug(f"[DEBUG] ENV ERROR: {e}")
        return None


def _heuristic_action(obs: dict) -> dict:
    """Rule-based fallback — runs when LLM is unavailable."""
    msg = (obs.get("customer_message") or "").lower()
    task_id = obs.get("task_id", "task1")
    metadata = obs.get("metadata", {})

    if any(w in msg for w in ["charge", "charged", "bill", "refund", "payment", "subscription"]):
        category = "billing"
    elif any(w in msg for w in ["ship", "deliver", "tracking", "package", "order"]):
        category = "shipping"
    elif any(w in msg for w in ["return", "broken", "defective", "replace", "damaged"]):
        category = "returns"
    elif any(w in msg for w in ["login", "password", "account", "crash", "error", "app", "hack"]):
        category = "technical"
    else:
        category = "general"

    if obs.get("category") is None:
        return {"action_type": "classify", "category": category}
    if obs.get("priority") is None:
        return {"action_type": "set_priority", "priority": "high"}
    if not metadata.get("has_reply") and task_id in ["task2", "task3"]:
        return {
            "action_type": "draft_reply",
            "reply_text": (
                "We sincerely apologize for the inconvenience you have experienced. "
                "We will investigate this immediately and contact you within 24 hours "
                "with a full resolution including refund or replacement if needed."
            )
        }
    if not metadata.get("has_note") and task_id == "task3":
        return {
            "action_type": "add_note",
            "note": f"Internal: urgent {category} issue, requires immediate follow-up and resolution."
        }
    return {
        "action_type": "resolve",
        "resolution_summary": (
            f"Issue resolved: {category} concern has been fully addressed. "
            "Refund or replacement processed as appropriate. Customer notified of outcome."
        )
    }


def _parse_action(text: str) -> Optional[dict]:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])
        return json.loads(cleaned)
    except Exception:
        return None


def _llm_action(client, obs: dict) -> dict:
    heuristic = _heuristic_action(obs)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Observation:\n{json.dumps(obs, indent=2)}"},
            ],
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _parse_action(content)
        if parsed is None:
            return heuristic
        return parsed
    except Exception as exc:
        _debug(f"[DEBUG] Model request failed: {exc}")
        return heuristic


def run_task(task_id: str, client: Optional[object]) -> float:
    reset_data = call_env("POST", "/reset", params={"task_id": task_id})
    if not reset_data:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        log_end(success=False, steps=0, score=0.001, rewards=[])
        return 0.001

    obs = reset_data["observation"]
    rewards: list[float] = []
    steps_taken = 0
    score = 0.001
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, 16):
            if client is not None:
                action = _llm_action(client, obs)
            else:
                action = _heuristic_action(obs)

            step_data = call_env("POST", "/step", payload=action, params={"task_id": task_id})
            if not step_data:
                break

            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            info = step_data.get("info", {})
            obs = step_data.get("observation", obs)

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action.get("action_type", "unknown"),
                reward=reward,
                done=done,
                error=info.get("error"),
            )

            if done:
                final = info.get("final_score")
                if final is not None:
                    score = float(final)
                else:
                    score = sum(rewards)
                score = max(0.001, min(score, 0.999))
                success = score >= SUCCESS_SCORE_THRESHOLD
                break
    except Exception as exc:
        _debug(f"[DEBUG] Task {task_id} failed: {exc}")
        return score
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    client: Optional[object] = None
    if API_KEY and OpenAI is not None:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    elif API_KEY and OpenAI is None:
        _debug("[DEBUG] openai package unavailable, falling back to heuristic policy.")

    for task_id in TASKS:
        run_task(task_id, client)


if __name__ == "__main__":
    main()
