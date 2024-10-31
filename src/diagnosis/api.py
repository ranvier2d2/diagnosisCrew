import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .crew import DiagnosisCrew, TaskOutput

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv_path = find_dotenv(raise_error_if_not_found=False)
if dotenv_path:
    load_dotenv(dotenv_path)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

tasks = {}


class Input(BaseModel):
    chief_complaint: str


class KickoffRequest(BaseModel):
    inputs: Input


class SubtaskStatus(BaseModel):
    status: str
    description: str
    summary: str = None


class TaskStatus(BaseModel):
    status: str
    result: Any = None
    error: str = None
    subtasks: Dict[str, SubtaskStatus] = {}
    tasks_output: List[Dict[str, Any]] = []
    token_usage: Dict[str, Any] = {}


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != "6ce0a076d321":
        logger.warning(f"Invalid token attempt: {credentials.credentials}")
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials


def task_callback(task_output: TaskOutput):
    task_id = task_output.task_id
    if task_id not in tasks:
        tasks[task_id] = TaskStatus(status="running")

    tasks[task_id].subtasks[task_output.description] = SubtaskStatus(
        status="completed",
        description=task_output.description,
        summary=task_output.summary,
    )

    tasks[task_id].tasks_output.append(
        {"description": task_output.description, "output": task_output.raw}
    )

    logger.info(f"Subtask completed for task {task_id}: {task_output.description}")


def run_crew(task_id: str, chief_complaint: str):
    logger.info(f"Starting task {task_id} for chief complaint: {chief_complaint}")
    try:
        crew = DiagnosisCrew()
        result = crew.kickoff(
            inputs={"chief_complaint": chief_complaint}, task_callback=task_callback
        )
        tasks[task_id].status = "completed"
        tasks[task_id].result = result.get("final_output")
        tasks[task_id].tasks_output = result.get("tasks_output", [])
        tasks[task_id].token_usage = result.get("token_usage", {})
        logger.info(f"All tasks for {task_id} completed successfully")
    except Exception as e:
        logger.error(f"Error in task {task_id}: {str(e)}", exc_info=True)
        tasks[task_id].status = "failed"
        tasks[task_id].error = str(e)


@app.post("/kickoff")
async def kickoff(
    request: KickoffRequest,
    background_tasks: BackgroundTasks,
    token: HTTPAuthorizationCredentials = Depends(verify_token),
):
    task_id = str(uuid4())
    tasks[task_id] = TaskStatus(status="running")
    logger.info(
        f"Received kickoff request for chief complaint: {request.inputs.chief_complaint}"
    )
    background_tasks.add_task(run_crew, task_id, request.inputs.chief_complaint)
    logger.info(f"Task {task_id} added to background tasks")
    return {"task_id": task_id}


@app.get("/status/{task_id}")
async def get_status(
    task_id: str, token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    logger.info(f"Received status request for task: {task_id}")
    if task_id not in tasks:
        logger.warning(f"Task not found: {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Returning response: {response.status_code}")
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {"detail": "An unexpected error occurred"}
