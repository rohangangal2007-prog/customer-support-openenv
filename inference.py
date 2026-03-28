"""
baseline_inference.py — Baseline agent for CustomerSupport OpenEnv

Runs an LLM against all 3 tasks and reports reproducible scores.
Reads API key from OPENAI_API_KEY environment variable.

Usage:
  OPENAI_API_KEY=sk-... python baseline_inference.py
  OPENAI_API_KEY=sk-... python baseline_inference.py --task task1
  OPENAI_API_KEY=sk-... python baseline_inference.py --url http://localhost:7860
"""

import os
import json
import argparse
import requests
from openai import OpenAI

# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_ENV_URL = "http://localhost:7860"
MODEL = "gpt-4o-mini"
TASKS = ["task1", "task2", "task3"]

SYSTEM_PROMPT = """You are an expert customer support agent AI.
You are interacting with a support ticket environment via structured JSON actions.

At each step you will receive an observation (a JSON object) and must respond
with exactly ONE action as a JSON object. Available actions:

1. classify — set the ticket category
   {"action_type": "classify", "category": "billing|technical|shipping|returns|general"}

2. set_priority — set urgency level
   {"action_type": "set_priority", "priority": "low|medium|high"}

3. draft_reply — write a customer-facing reply
   {"action_type": "draft_reply", "reply_text": "..."}

4. add_note — add an internal note for the team
   {"action_type": "add_note", "note": "..."}

5. resolve — close the ticket with a resolution summary
   {"action_type": "resolve", "resolution_summary": "..."}

6. escalate — escalate to a senior agent
   {"action_type": "escalate", "escalation_reason": "..."}

Rules:
- Always output valid JSON and nothing else.
- Read the task_description carefully — it tells you what actions are required.
- For task1: classify then set_priority, then resolve.
- For task2: classify → set_priority → draft_reply → resolve.
- For task3: classify → set_priority → draft_reply → add_note → resolve.
- Be empathetic and professional in replies.
- Resolution summaries must mention refund or replacement when relevant.
"""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def call_env(method: str, path: str, url: str, payload=None, params=None):
    full_url = url.rstrip("/") + path
    try:
        if method == "POST":
            r = requests.post(full_url, json=payload, params=params, timeout=10)
        else:
            r = requests.get(full_url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"  [ENV ERROR] {e}")
        return None


def ask_llm(client: OpenAI, messages: list) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return None


# ─── Episode runner ───────────────────────────────────────────────────────────

def run_episode(client: OpenAI, env_url: str, task_id: str) -> float:
    print(f"\n{'='*60}")
    print(f"  Task: {task_id}")
    print(f"{'='*60}")

    # Reset
    reset_data = call_env("POST", "/reset", env_url, params={"task_id": task_id})
    if not reset_data:
        return 0.0

    obs = reset_data["observation"]
    print(f"  Ticket: {obs['ticket_id']}")
    print(f"  Message: {obs['customer_message'][:80]}...")
    print(f"  Goal: {obs['task_description'][:100]}...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Observation:\n{json.dumps(obs, indent=2)}"},
    ]

    total_reward = 0.0
    step = 0

    while not obs.get("done", False):
        step += 1
        print(f"\n  Step {step}:")

        action_dict = ask_llm(client, messages)
        if not action_dict:
            print("  LLM failed to produce valid JSON. Skipping step.")
            break

        print(f"  Action: {json.dumps(action_dict)}")

        step_data = call_env(
            "POST", "/step", env_url,
            payload=action_dict,
            params={"task_id": task_id}
        )
        if not step_data:
            break

        reward = step_data["reward"]
        total_reward += reward
        obs = step_data["observation"]
        info = step_data.get("info", {})

        print(f"  Reward: {reward:+.4f}  |  Info: {info.get('feedback', info.get('error', ''))}")

        # Add result to conversation history for context
        messages.append({
            "role": "assistant",
            "content": json.dumps(action_dict)
        })
        messages.append({
            "role": "user",
            "content": (
                f"Step result:\n"
                f"  reward={reward}\n"
                f"  done={step_data['done']}\n"
                f"  info={json.dumps(info)}\n\n"
                f"Next observation:\n{json.dumps(obs, indent=2)}"
            )
        })

        if step_data["done"]:
            final = info.get("final_score")
            if final is not None:
                print(f"\n  Final grader score: {final:.2f}")
            break

    print(f"\n  Episode total reward: {total_reward:.4f}")
    return total_reward


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CustomerSupport OpenEnv baseline agent")
    parser.add_argument("--url", default=DEFAULT_ENV_URL, help="Environment server URL")
    parser.add_argument("--task", default=None, help="Run a specific task only (task1|task2|task3)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI(api_key=api_key)

    # Health check
    health = call_env("GET", "/health", args.url)
    if not health:
        print(f"ERROR: Could not reach environment at {args.url}")
        print("Make sure the server is running: uvicorn main:app --host 0.0.0.0 --port 7860")
        return
    print(f"Environment healthy at {args.url}")

    tasks_to_run = [args.task] if args.task else TASKS
    scores = {}

    for task_id in tasks_to_run:
        score = run_episode(client, args.url, task_id)
        scores[task_id] = round(score, 4)

    print(f"\n{'='*60}")
    print("  BASELINE SCORES")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task_id}: {score:.4f}  {bar}")
    if len(scores) > 1:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  Average: {avg:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
