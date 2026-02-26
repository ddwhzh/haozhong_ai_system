"""Experiment Runner — executes experiment code in a sandbox.

Input: Experiment with code and dependencies
Output: Experiment with results populated
"""

from __future__ import annotations

from typing import Any, Dict

from app.core.config import settings
from app.core.logging import logger
from app.core.types.research_types import Experiment, ExperimentStatus, JournalEntry
from app.agents.research.sandbox import SandboxEngine


class ExperimentRunner:
    """Execute experiments in an isolated sandbox environment."""

    def __init__(self) -> None:
        self.sandbox = SandboxEngine(
            timeout_seconds=settings.RESEARCH_SANDBOX_TIMEOUT,
            memory_limit=settings.RESEARCH_SANDBOX_MEMORY,
            cpu_count=settings.RESEARCH_SANDBOX_CPU,
            docker_image=settings.RESEARCH_SANDBOX_IMAGE,
        )

    async def run(self, experiment: Experiment) -> Dict[str, Any]:
        """Execute an experiment and return updated experiment with results.

        Returns:
            dict with keys: experiment, journal_entry
        """
        logger.info(
            "experiment_run_start",
            experiment_name=experiment.name,
            code_length=len(experiment.code),
            dependencies=experiment.dependencies,
        )

        experiment.status = ExperimentStatus.RUNNING

        result = await self.sandbox.execute(
            code=experiment.code,
            dependencies=experiment.dependencies,
        )

        if result.exit_code == 124:
            experiment.status = ExperimentStatus.TIMEOUT
        elif result.exit_code != 0:
            experiment.status = ExperimentStatus.FAILED
        else:
            experiment.status = ExperimentStatus.COMPLETED

        experiment.result = result

        journal = JournalEntry(
            phase="experiment",
            summary=(
                f"Experiment '{experiment.name}' {experiment.status.value}: "
                f"exit_code={result.exit_code}, elapsed={result.elapsed_seconds}s"
            ),
            details={
                "experiment_name": experiment.name,
                "status": experiment.status.value,
                "exit_code": result.exit_code,
                "elapsed_seconds": result.elapsed_seconds,
                "metrics": result.metrics,
                "has_stderr": bool(result.stderr),
            },
        )

        logger.info(
            "experiment_run_complete",
            experiment_name=experiment.name,
            status=experiment.status.value,
            exit_code=result.exit_code,
            elapsed=result.elapsed_seconds,
        )

        return {
            "experiment": experiment,
            "journal_entry": journal,
        }
