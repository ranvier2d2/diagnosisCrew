from typing import Any, Callable, Dict, Optional

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


class TaskOutput:
    def __init__(self, task_id: str, description: str, summary: str, raw: Any):
        self.task_id = task_id
        self.description = description
        self.summary = summary
        self.raw = raw


@CrewBase
class DiagnosisCrew:
    """Diagnosis crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def history_taker(self) -> Agent:
        return Agent(
            config=self.agents_config["history_taker"],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def examiner(self) -> Agent:
        return Agent(
            config=self.agents_config["examiner"],
            verbose=True,
            allow_delegation=False,
        )

    @agent
    def diagnostician(self) -> Agent:
        return Agent(
            config=self.agents_config["diagnostician"],
            verbose=True,
            allow_delegation=False,
        )

    def create_task(
        self, task_config: dict, task_callback: Optional[Callable] = None
    ) -> Task:
        task = Task(config=task_config)
        if task_callback:
            task.callback = lambda output: task_callback(
                TaskOutput(
                    task_id=task.id,
                    description=task.description,
                    summary=task.expected_output[
                        :100
                    ],  # First 100 characters as summary
                    raw=output,
                )
            )
        return task

    @task
    def gather_history_task(self, task_callback: Optional[Callable] = None) -> Task:
        return self.create_task(self.tasks_config["gather_history_task"], task_callback)

    @task
    def perform_examination_task(
        self, task_callback: Optional[Callable] = None
    ) -> Task:
        return self.create_task(
            self.tasks_config["perform_examination_task"], task_callback
        )

    @task
    def generate_differential_diagnosis_task(
        self, task_callback: Optional[Callable] = None
    ) -> Task:
        return self.create_task(
            self.tasks_config["generate_differential_diagnosis_task"], task_callback
        )

    @task
    def bayesian_reasoning_task(self, task_callback: Optional[Callable] = None) -> Task:
        return self.create_task(
            self.tasks_config["bayesian_reasoning_task"], task_callback
        )

    @task
    def synthesize_diagnostic_framework_task(
        self, task_callback: Optional[Callable] = None
    ) -> Task:
        return self.create_task(
            self.tasks_config["synthesize_diagnostic_framework_task"], task_callback
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Diagnosis crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            full_output=True,
        )

    def kickoff(
        self, inputs: dict, task_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        crew = self.crew()
        for task in crew.tasks:
            if task_callback:
                task.callback = lambda output, task=task: task_callback(
                    TaskOutput(
                        task_id=task.id,
                        description=task.description,
                        summary=task.expected_output[:100],
                        raw=output,
                    )
                )
        result = crew.kickoff(inputs=inputs)
        return {
            "final_output": result.raw,
            "tasks_output": [
                {
                    "description": task_output.description,
                    "output": task_output.raw,
                }
                for task_output in result.tasks_output
            ],
            "token_usage": result.token_usage,
        }
