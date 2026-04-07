"""
inference.py — Baseline agent for CustomerSupport OpenEnv

Prints structured [START]/[STEP]/[END] blocks to stdout as required
by the OpenEnv hackathon validator.

Usage:
  python inference.py --url http://localhost:7860
  python inference.py --url https://codebyrohan-customer-support-openenv.hf.space
"""

import os
import json
import argparse
import requests

DEFAULT_ENV_URL = "http://localhost:7860"
TASKS = ["task1", "task2", "task3"]


def call_env(method, path, url, payload=None, params=None):
    full_url = url.rstrip("/") + path
    try:
        if method == "POST":
            r = requests.post(full_url, json=payload, params=params, timeout=15)
        else:
            r = requests.get(full_url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"ENV ERROR: {e}", flush=True)
        return None


def get_action_for_obs(obs, step_num):
    """
    Rule-based agent — no LLM needed for baseline.
    Deterministic actions based on task and step number.
    """
    task_id = obs.get("task_id", "task1")
    msg = obs.get("customer_message", "").lower()

    # Determine category from message keywords
    if any(w in msg for w in ["charge", "charged", "bill", "refund", "payment", "subscription"]):
        category = "billing"
    elif any(w in msg for w in ["ship", "deliver", "tracking", "package", "order", "arrived"]):
        category = "shipping"
    elif any(w in msg for w in ["return", "broken", "defective", "replace", "damaged"]):
        category = "returns"
    elif any(w in msg for w in ["login", "password", "account", "crash", "error", "bug", "app"]):
        category = "technical"
    else:
        category = "general"

    # All urgent messages get high priority
    priority = "high" if any(w in msg for w in [
        "immediately", "urgent", "asap", "emergency", "frustrating",
        "now", "event", "weekend", "work", "important"
    ]) else "medium"

    metadata = obs.get("metadata", {})
    has_reply = metadata.get("has_reply", False)
    has_note = metadata.get("has_note", False)
    is_resolved = metadata.get("is_resolved", False)

    # Action sequence based on what's been done
    if obs.get("category") is None:
        return {"action_type": "classify", "category": category}
    elif obs.get("priority") is None:
        return {"action_type": "set_priority", "priority": priority}
    elif not has_reply and task_id in ["task2", "task3"]:
        return {
            "action_type": "draft_reply",
            "reply_text": (
                f"Dear Customer, we sincerely apologize for the inconvenience you've experienced. "
                f"We understand how frustrating this situation must be and want to resolve it immediately. "
                f"Our team will investigate this matter and contact you within 24 hours with a full update. "
                f"We appreciate your patience and are committed to making this right for you."
            )
        }
    elif not has_note and task_id == "task3":
        return {
            "action_type": "add_note",
            "note": f"Internal: Customer issue categorized as {category}. Requires urgent follow-up and resolution. Priority: {priority}."
        }
    else:
        return {
            "action_type": "resolve",
            "resolution_summary": (
                f"Issue resolved: Customer concern regarding {category} has been addressed. "
                f"Full refund or replacement has been processed as appropriate. "
                f"Customer has been notified and the ticket is now closed."
            )
        }


def run_episode(env_url, task_id):
    """Run one episode and print structured output blocks."""

    # Reset
    reset_data = call_env("POST", "/reset", env_url, params={"task_id": task_id})
    if not reset_data:
        print(f"[START] task={task_id}", flush=True)
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return 0.0

    obs = reset_data["observation"]
    print(f"[START] task={task_id}", flush=True)

    total_reward = 0.0
    step_num = 0

    while not obs.get("done", False):
        step_num += 1
        action = get_action_for_obs(obs, step_num)

        step_data = call_env(
            "POST", "/step", env_url,
            payload=action,
            params={"task_id": task_id}
        )
        if not step_data:
            break

        reward = step_data.get("reward", 0.0)
        total_reward += reward
        obs = step_data.get("observation", obs)
        info = step_data.get("info", {})
        done = step_data.get("done", False)

        print(f"[STEP] step={step_num} reward={reward} action={action['action_type']}", flush=True)

        if done:
            break

        # Safety limit
        if step_num >= 15:
            break

    final_score = round(total_reward, 4)
    print(f"[END] task={task_id} score={final_score} steps={step_num}", flush=True)
    return final_score


def main():
    parser = argparse.ArgumentParser(description="CustomerSupport OpenEnv baseline inference")
    parser.add_argument("--url", default=DEFAULT_ENV_URL)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    env_url = os.environ.get("ENV_URL", args.url)
    tasks_to_run = [args.task] if args.task else TASKS

    scores = {}
    for task_id in tasks_to_run:
        score = run_episode(env_url, task_id)
        scores[task_id] = score

    # Print summary
    for task_id, score in scores.items():
        print(f"[RESULT] task={task_id} score={score}", flush=True)


if __name__ == "__main__":
    main()
