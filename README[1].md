# CustomerSupport OpenEnv

An OpenEnv-compliant reinforcement learning environment that simulates a
real-world **customer support agent workflow**.

An AI agent receives support tickets and must take structured actions —
classifying, prioritising, replying, and resolving — earning partial rewards
throughout each episode.

---

## Environment Description

The environment models a customer support queue. The agent acts as a support
agent and must process tickets end-to-end. Unlike toy environments, this
requires natural language understanding, empathy in replies, and correct
procedural sequencing.

**Why this is a good RL environment:**
- Actions have clear success/failure criteria (correct category = +0.2)
- Reward is dense — the agent gets feedback at every step
- Behaviour is penalised (bad replies, invalid actions, timeouts)
- Tasks scale in complexity from easy to hard

---

## Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `classify` | `category` | Assign ticket to billing / technical / shipping / returns / general |
| `set_priority` | `priority` | Set urgency: low / medium / high |
| `draft_reply` | `reply_text` | Write a customer-facing reply (≥20 chars) |
| `add_note` | `note` | Add an internal note for the team |
| `resolve` | `resolution_summary` | Close the ticket with a summary (≥20 chars) |
| `escalate` | `escalation_reason` | Hand off to a senior agent |

---

## Observation Space

```json
{
  "ticket_id": "TKT-001",
  "customer_message": "I was charged twice...",
  "category": null,
  "priority": null,
  "conversation_history": [],
  "task_id": "task1",
  "task_description": "Classify ticket and set priority.",
  "step_number": 0,
  "done": false,
  "metadata": {
    "difficulty": "easy",
    "max_steps": 4,
    "has_reply": false,
    "has_note": false,
    "is_resolved": false
  }
}
```

---

## Tasks

### Task 1 — Easy (max 4 steps)
Classify a duplicate billing charge ticket into the correct category
(`billing`) and set the correct priority (`high`).

- Correct category: **+0.2**
- Correct priority: **+0.2**
- Max score: **1.0**

**Baseline score: ~0.80**

---

### Task 2 — Medium (max 6 steps)
Handle a missing shipment ticket. Must classify, prioritise, and draft
an empathetic reply that mentions an investigation timeline and next steps.

- Correct category: **+0.2**
- Correct priority: **+0.2**
- Reply drafted: **+0.2**
- Good keywords in reply: up to **+0.2**
- No bad phrases: **+0.2**
- Max score: **1.0**

**Baseline score: ~0.72**

---

### Task 3 — Hard (max 10 steps)
Full resolution of a defective product return. Must classify, prioritise,
draft a reply, add an internal warehouse note, and resolve with a summary
that mentions refund or replacement.

- Correct category + priority: **+0.30**
- Quality reply: up to **+0.30**
- Internal note: **+0.10**
- Resolution with refund/replacement mention: **+0.30**
- Max score: **1.0**

**Baseline score: ~0.65**

---

## Reward Function

Rewards are **dense** — the agent receives a signal after every action:

| Situation | Reward |
|-----------|--------|
| Correct category | +0.20 |
| Correct priority | +0.20 |
| Quality reply (keyword scoring) | +0.00 to +0.25 |
| Internal note added | +0.10 |
| Resolve (triggers full grader) | 0.0 – 1.0 |
| Invalid action | −0.10 |
| Incorrect classification | −0.05 |
| Max steps exceeded | −0.10 |

---

## Setup & Usage

### Local development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn main:app --host 0.0.0.0 --port 7860

# In another terminal, run the baseline agent
OPENAI_API_KEY=sk-... python baseline_inference.py
```

### Docker

```bash
docker build -t customer-support-env .
docker run -p 7860:7860 customer-support-env
```

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset?task_id=task1` | Start a new episode |
| `POST` | `/step?task_id=task1` | Take an action |
| `GET` | `/state?task_id=task1` | Inspect current state |
| `GET` | `/tasks` | List all tasks |
| `GET` | `/health` | Liveness check |
| `GET` | `/docs` | Interactive Swagger UI |

### Example API usage

```python
import requests

# Reset
obs = requests.post("http://localhost:7860/reset?task_id=task1").json()

# Take an action
result = requests.post(
    "http://localhost:7860/step?task_id=task1",
    json={"action_type": "classify", "category": "billing"}
).json()

print(result["reward"])   # 0.2
print(result["done"])     # False
```

---

## OpenEnv Compliance

- ✅ `step()` — returns observation, reward, done, info
- ✅ `reset()` — returns initial observation
- ✅ `state()` — returns current state without advancing
- ✅ Typed Pydantic models for Observation, Action, StepResult
- ✅ `openenv.yaml` with metadata
- ✅ 3 tasks: easy → medium → hard, graders score 0.0–1.0
- ✅ Dense reward with partial progress signals
- ✅ Baseline inference script with reproducible scores
- ✅ Dockerfile for containerised HF Spaces deployment

---

## Baseline Scores (GPT-4o-mini, temperature=0)

| Task | Score |
|------|-------|
| task1 (easy) | ~0.80 |
| task2 (medium) | ~0.72 |
| task3 (hard) | ~0.65 |
| **Average** | **~0.72** |
