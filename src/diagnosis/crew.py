from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task


@CrewBase
class DiagnosisCrew:
    """Diagnosis crew"""

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

    @task
    def gather_history_task(self) -> Task:
        return Task(
            config=self.tasks_config["gather_history_task"],
        )

    @task
    def perform_examination_task(self) -> Task:
        return Task(
            config=self.tasks_config["perform_examination_task"],
        )

    @task
    def generate_differential_diagnosis_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_differential_diagnosis_task"],
        )

    @task
    def bayesian_reasoning_task(self) -> Task:
        return Task(
            config=self.tasks_config["bayesian_reasoning_task"],
        )

    @task
    def synthesize_diagnostic_framework_task(self) -> Task:
        return Task(
            config=self.tasks_config["synthesize_diagnostic_framework_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Diagnosis crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
