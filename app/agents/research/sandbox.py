"""Sandbox engine — isolated code execution for research experiments.

Executes experiment code in a Docker container with resource limits:
- CPU: configurable (default 1 core)
- Memory: configurable (default 2GB)
- Timeout: configurable (default 300s)
- Network: disabled by default

Falls back to subprocess execution (with timeout) if Docker is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.core.types.research_types import ExperimentResult


class SandboxEngine:
    """Execute experiment code in an isolated environment."""

    def __init__(
        self,
        timeout_seconds: int = 300,
        memory_limit: str = "2g",
        cpu_count: int = 1,
        docker_image: str = "python:3.11-slim",
        network_disabled: bool = True,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.cpu_count = cpu_count
        self.docker_image = docker_image
        self.network_disabled = network_disabled

    async def execute(
        self,
        code: str,
        dependencies: Optional[List[str]] = None,
        env_vars: Optional[Dict[str, str]] = None,
    ) -> ExperimentResult:
        """Execute Python code in a sandbox.

        Tries Docker first; falls back to subprocess if Docker is unavailable.
        """
        if not code or not code.strip():
            return ExperimentResult(
                stdout="",
                stderr="error: empty code",
                exit_code=1,
            )

        docker_available = await self._check_docker()
        if docker_available:
            logger.info("sandbox_using_docker", image=self.docker_image)
            return await self._execute_docker(code, dependencies or [], env_vars or {})

        logger.warning("sandbox_docker_unavailable_fallback_subprocess")
        return await self._execute_subprocess(code, dependencies or [])

    async def _check_docker(self) -> bool:
        """Check if Docker daemon is accessible."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "info",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=5)
            return proc.returncode == 0
        except (FileNotFoundError, asyncio.TimeoutError, OSError):
            return False

    async def _execute_docker(
        self,
        code: str,
        dependencies: List[str],
        env_vars: Dict[str, str],
    ) -> ExperimentResult:
        """Execute code inside a Docker container."""
        with tempfile.TemporaryDirectory(prefix="research_sandbox_") as tmpdir:
            script_path = Path(tmpdir) / "experiment.py"
            script_path.write_text(code, encoding="utf-8")

            setup_script = self._build_setup_script(dependencies)
            setup_path = Path(tmpdir) / "setup.sh"
            setup_path.write_text(setup_script, encoding="utf-8")

            cmd = [
                "docker", "run", "--rm",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_count}",
                "-v", f"{tmpdir}:/workspace:rw",
                "-w", "/workspace",
            ]

            if self.network_disabled:
                cmd.append("--network=none")

            for k, v in env_vars.items():
                cmd.extend(["-e", f"{k}={v}"])

            cmd.extend([
                self.docker_image,
                "bash", "-c", "bash /workspace/setup.sh && python /workspace/experiment.py",
            ])

            return await self._run_process(cmd)

    async def _execute_subprocess(
        self,
        code: str,
        dependencies: List[str],
    ) -> ExperimentResult:
        """Fallback: execute code as a subprocess with timeout."""
        python_exe = sys.executable

        with tempfile.TemporaryDirectory(prefix="research_sandbox_") as tmpdir:
            script_path = Path(tmpdir) / "experiment.py"
            script_path.write_text(code, encoding="utf-8")

            if dependencies:
                install_cmd = await self._find_install_cmd(python_exe, dependencies)
                if install_cmd:
                    try:
                        dep_proc = await asyncio.create_subprocess_exec(
                            *install_cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        stdout_b, stderr_b = await asyncio.wait_for(
                            dep_proc.communicate(), timeout=120
                        )
                        if dep_proc.returncode != 0:
                            logger.warning(
                                "sandbox_dependency_install_failed",
                                dependencies=dependencies,
                                stderr=stderr_b.decode("utf-8", errors="replace")[:500],
                            )
                        else:
                            logger.info(
                                "sandbox_dependency_install_success",
                                dependencies=dependencies,
                            )
                    except asyncio.TimeoutError:
                        logger.warning("sandbox_dependency_install_timeout", dependencies=dependencies)
                    except FileNotFoundError:
                        logger.warning("sandbox_installer_not_found")

            cmd = [python_exe, str(script_path)]
            return await self._run_process(cmd)

    async def _run_process(self, cmd: List[str]) -> ExperimentResult:
        """Run a command with timeout, capturing output."""
        import time

        start = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout_seconds
            )
            elapsed = time.monotonic() - start
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")

            output_files = self._try_parse_output_files(stdout)
            metrics = self._try_parse_metrics(stdout)

            logger.info(
                "sandbox_execution_complete",
                exit_code=proc.returncode,
                elapsed=round(elapsed, 2),
                stdout_len=len(stdout),
            )

            return ExperimentResult(
                stdout=stdout[-10000:],
                stderr=stderr[-5000:],
                exit_code=proc.returncode or 0,
                output_files=output_files,
                metrics=metrics,
                elapsed_seconds=round(elapsed, 2),
            )

        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            logger.warning(
                "sandbox_execution_timeout",
                timeout=self.timeout_seconds,
                elapsed=round(elapsed, 2),
            )
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass

            return ExperimentResult(
                stdout="",
                stderr=f"Experiment timed out after {self.timeout_seconds}s",
                exit_code=124,
                elapsed_seconds=round(elapsed, 2),
            )

        except FileNotFoundError as exc:
            return ExperimentResult(
                stdout="",
                stderr=f"Command not found: {exc}",
                exit_code=127,
            )

        except OSError as exc:
            logger.exception("sandbox_execution_os_error", error=str(exc))
            return ExperimentResult(
                stdout="",
                stderr=f"OS error: {exc}",
                exit_code=1,
            )

    def _build_setup_script(self, dependencies: List[str]) -> str:
        """Build a bash setup script for dependency installation."""
        lines = ["#!/bin/bash", "set -e"]
        if dependencies:
            deps = " ".join(dependencies)
            lines.append(f"pip install --quiet {deps}")
        return "\n".join(lines)

    async def _find_install_cmd(
        self, python_exe: str, dependencies: List[str]
    ) -> Optional[List[str]]:
        """Find the best package installer available (uv > pip)."""
        for installer in ["uv", "pip"]:
            try:
                probe = await asyncio.create_subprocess_exec(
                    installer, "--version",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(probe.wait(), timeout=5)
                if probe.returncode == 0:
                    if installer == "uv":
                        return ["uv", "pip", "install", "--python", python_exe, "--quiet"] + dependencies
                    return [python_exe, "-m", "pip", "install", "--quiet"] + dependencies
            except (FileNotFoundError, asyncio.TimeoutError):
                continue

        logger.warning("sandbox_no_installer_found")
        return None

    def _try_parse_metrics(self, stdout: str) -> Dict[str, Any]:
        """Attempt to parse JSON metrics from the last JSON object in stdout."""
        for line in reversed(stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        return {}

    def _try_parse_output_files(self, stdout: str) -> Dict[str, str]:
        """No-op in current implementation; reserved for file extraction."""
        return {}
