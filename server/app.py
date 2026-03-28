"""
FastAPI server for CustomerSupportEnv.

Endpoints:
  POST /reset          — start/restart an episode
  POST /step           — take an action
  GET  /state          — inspect current state
  GET  /tasks          — list available tasks
  GET  /health         — liveness check
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from models import Action, ResetResult, StepResult, StateResult
from environment import CustomerSupportEnv

# One env instance per session (single-user server; extend with session IDs for multi-user)
envs: dict[str, CustomerSupportEnv] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm all three tasks
    for task_id in ["task1", "task2", "task3"]:
        envs[task_id] = CustomerSupportEnv(task_id=task_id)
    yield


app = FastAPI(
    title="CustomerSupport OpenEnv",
    description=(
        "An OpenEnv-compliant reinforcement learning environment simulating "
        "a real-world customer support agent workflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def list_tasks():
    from tasks import TASKS
    return {
        task_id: {
            "difficulty": t.difficulty,
            "description": t.description,
            "max_steps": t.max_steps,
        }
        for task_id, t in TASKS.items()
    }


@app.post("/reset", response_model=ResetResult)
def reset(task_id: str = Query("task1", description="task1 | task2 | task3")):
    if task_id not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    result = envs[task_id].reset()
    return result


@app.post("/step", response_model=StepResult)
def step(action: Action, task_id: str = Query("task1")):
    if task_id not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    env = envs[task_id]
    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/state", response_model=StateResult)
def state(task_id: str = Query("task1")):
    if task_id not in envs:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'")
    env = envs[task_id]
    try:
        result = env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()
