"""
inference.py — Baseline agent for CustomerSupport OpenEnv

Uses competition's LiteLLM proxy via API_BASE_URL, API_KEY, MODEL env vars.
Prints [START]/[STEP]/[END] structured blocks to stdout.
"""

import os
import json
import argparse
import requests
from openai import OpenAI

DEFAULT_ENV_URL = "http://localhost:7860"
TASKS = ["task1", "task2", "task3"]

SYSTEM_PROMPT = """You are an expert customer support agent AI.
Respond with exactly ONE JSON action object. No markdown, no explanation.

Available actions:
{"action_type": "classify", "category": "billing|technical|shipping|returns|general"}
{"action_type": "set_priority", "priority": "low|medium|high"}
{"action_type": "draft_reply", "reply_text": "..."}
{"action_type": "add_note", "note": "..."}
{"action_type": "resolve", "resolution_summary": "..."}

For task1: classify then set_priority then resolve.
For task2: classify then set_priority then draft_reply then resolve.
For task3: classify then set_priority then draft_reply then add_note then resolve.
Resolution must mention refund or replacement when relevant.
"""


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


def ask_llm(client, messages, model):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        print(f"LLM ERROR: {e}", flush=True)
        return None


def run_episode(client, model, env_url, task_id):
    reset_data = call_env("POST", "/reset", env_url, params={"task_id": task_id})
    if not reset_data:
        print(f"[START] task={task_id}", flush=True)
        print(f"[END] task={task_id} score=0.0 steps=0", flush=True)
        return 0.0

    obs = reset_data["observation"]
    print(f"[START] task={task_id}", flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Observation:\n{json.dumps(obs, indent=2)}"},
    ]

    total_reward = 0.0
    step_num = 0

    while not obs.get("done", False):
        step_num += 1
        action_dict = ask_llm(client, messages, model)
        if not action_dict:
            print(f"[STEP] step={step_num} reward=0.0 action=error", flush=True)
            break

        step_data = call_env("POST", "/step", env_url, payload=action_dict, params={"task_id": task_id})
        if not step_data:
            break

        reward = step_data.get("reward", 0.0)
        total_reward += reward
        obs = step_data.get("observation", obs)
        info = step_data.get("info", {})
        done = step_data.get("done", False)

        print(f"[STEP] step={step_num} reward={reward} action={action_dict.get('action_type', 'unknown')}", flush=True)

        messages.append({"role": "assistant", "content": json.dumps(action_dict)})
        messages.append({
            "role": "user",
            "content": f"reward={reward}, done={done}\nNext obs:\n{json.dumps(obs, indent=2)}"
        })

        if done or step_num >= 15:
            break

    final_score = round(max(0.001, min(total_reward, 0.999)), 4)
    print(f"[END] task={task_id} score={final_score} steps={step_num}", flush=True)
    return final_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.environ.get("ENV_URL", DEFAULT_ENV_URL))
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    # Read ALL config from environment variables injected by the competition
    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("API_KEY", os.environ.get("OPENAI_API_KEY", "dummy"))
    model = os.environ.get("MODEL_NAME", os.environ.get("MODEL", "gpt-4o-mini"))

    print(f"Using API_BASE_URL={api_base_url}", flush=True)
    print(f"Using MODEL={model}", flush=True)

    client = OpenAI(base_url=api_base_url, api_key=api_key)

    tasks_to_run = [args.task] if args.task else TASKS
    scores = {}

    for task_id in tasks_to_run:
        score = run_episode(client, model, args.url, task_id)
        scores[task_id] = score

    for task_id, score in scores.items():
        print(f"[RESULT] task={task_id} score={score}", flush=True)


if __name__ == "__main__":
    main()
