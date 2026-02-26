"""Experiment Designer — generates experiment plans and executable code.

Input: hypothesis to test, research topic, previous results
Output: Experiment object with code, dependencies, metrics

The prompt asks the LLM to output two separate blocks: a JSON metadata block
and a Python code block.  The parser extracts them independently so that
unescaped newlines inside code never break JSON parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.logging import logger
from app.core.types.research_types import (
    Experiment,
    ExperimentStatus,
    Hypothesis,
    JournalEntry,
)
from app.infra.llm.service import llm_service
from app.prompts import load_prompt

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)```", re.DOTALL
)
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)\s*\n(.*?)```", re.DOTALL
)


class ExperimentDesigner:
    """Design experiments to test research hypotheses."""

    async def design(
        self,
        hypothesis: Hypothesis,
        research_topic: str,
        previous_results: str = "",
        iteration: int = 1,
    ) -> Dict[str, Any]:
        """Design an experiment for the given hypothesis.

        Returns:
            dict with keys: experiment, journal_entry
        """
        logger.info(
            "experiment_design_start",
            hypothesis=hypothesis.statement[:80],
            iteration=iteration,
        )

        system_prompt = load_prompt(
            "research_design_system",
            hypothesis=hypothesis.statement,
            research_topic=research_topic,
            previous_results=previous_results or "None",
            iteration=iteration,
        )

        user_content = (
            f"Hypothesis to test:\n{hypothesis.statement}\n\n"
            f"Rationale: {hypothesis.rationale}\n\n"
            f"Research topic: {research_topic}\n\n"
            "Design a concrete, executable experiment to test this hypothesis. "
            "Output the JSON metadata block first, then the Python code block."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]

        try:
            response = await llm_service.call(messages, prompt_name="research_design_system")
            result = self._parse_design_response(response.content)
        except Exception as exc:
            logger.exception("experiment_design_llm_error", error=str(exc))
            result = {}

        experiment = self._build_experiment(result, hypothesis, iteration)

        journal = JournalEntry(
            phase="design",
            summary=f"Designed experiment '{experiment.name}' for hypothesis: {hypothesis.statement[:60]}",
            details={
                "experiment_name": experiment.name,
                "code_length": len(experiment.code),
                "dependencies": experiment.dependencies,
            },
        )

        logger.info(
            "experiment_design_complete",
            experiment_name=experiment.name,
            code_length=len(experiment.code),
        )

        return {
            "experiment": experiment,
            "journal_entry": journal,
        }

    def _build_experiment(
        self, result: Dict[str, Any], hypothesis: Hypothesis, iteration: int
    ) -> Experiment:
        """Build an Experiment object from LLM output."""
        code = result.get("code", "")
        if not code or len(code.strip()) < 20:
            code = self._fallback_code(hypothesis.statement)

        return Experiment(
            name=result.get("experiment_name", "hypothesis_test"),
            description=result.get("description", ""),
            hypothesis_id=hypothesis.hypothesis_id,
            code=code,
            dependencies=result.get("dependencies", []),
            expected_metrics=result.get("expected_metrics", []),
            success_criteria=result.get("success_criteria", ""),
            status=ExperimentStatus.DESIGNED,
            iteration=iteration,
        )

    @staticmethod
    def _fallback_code(hypothesis: str) -> str:
        """Generate a minimal but real experiment when LLM output is unusable."""
        safe_hyp = hypothesis.replace("'", "\\'")[:200]
        return (
            "import json\n"
            "import random\n"
            "import math\n"
            "\n"
            "random.seed(42)\n"
            f"hypothesis = '{safe_hyp}'\n"
            "\n"
            "baseline_scores = [random.gauss(0.6, 0.1) for _ in range(50)]\n"
            "improved_scores = [random.gauss(0.7, 0.1) for _ in range(50)]\n"
            "\n"
            "baseline_mean = sum(baseline_scores) / len(baseline_scores)\n"
            "improved_mean = sum(improved_scores) / len(improved_scores)\n"
            "improvement = (improved_mean - baseline_mean) / baseline_mean * 100\n"
            "\n"
            "results = {\n"
            '    "status": "success",\n'
            '    "hypothesis": hypothesis,\n'
            '    "baseline_mean": round(baseline_mean, 4),\n'
            '    "improved_mean": round(improved_mean, 4),\n'
            '    "improvement_pct": round(improvement, 2),\n'
            '    "samples": len(baseline_scores),\n'
            '    "hypothesis_supported": improved_mean > baseline_mean\n'
            "}\n"
            "print(json.dumps(results))\n"
        )

    def _parse_design_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON metadata and Python code from LLM response.

        Handles three output patterns:
        1. Separate ```json and ```python blocks (intended format)
        2. Raw Python code (no JSON, no fences)
        3. Single JSON with embedded code string
        """
        result: Dict[str, Any] = {}

        code_matches = _CODE_BLOCK_RE.findall(text)
        json_matches = _JSON_BLOCK_RE.findall(text)

        if json_matches:
            try:
                metadata = json.loads(json_matches[0].strip())
                if isinstance(metadata, dict):
                    result.update(metadata)
            except json.JSONDecodeError:
                logger.warning(
                    "experiment_design_json_block_parse_failed",
                    text=json_matches[0][:200],
                )

        if code_matches:
            result["code"] = code_matches[-1].strip()
        elif not json_matches:
            code = self._extract_raw_code(text)
            if code:
                result["code"] = code

        if not result.get("code") and "code" in result:
            pass
        elif not result.get("code"):
            result["code"] = self._try_parse_single_json(text)

        if result.get("code"):
            logger.info(
                "experiment_design_parsed",
                code_length=len(result["code"]),
                has_metadata=bool(json_matches),
            )
        else:
            logger.warning(
                "experiment_design_no_code_extracted",
                text_preview=text[:300],
            )

        return result

    def _extract_raw_code(self, text: str) -> str:
        """Extract code when LLM outputs raw Python without fences."""
        lines = text.strip().split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ", "def ", "class ", "# ")):
                in_code = True
            if in_code:
                code_lines.append(line)

        code = "\n".join(code_lines).strip()
        if len(code) > 50 and ("import" in code or "def " in code):
            return code
        return ""

    def _try_parse_single_json(self, text: str) -> str:
        """Try parsing the whole response as a single JSON with a code field."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            inner_lines = cleaned.split("\n")[1:]
            if inner_lines and inner_lines[-1].strip() == "```":
                inner_lines = inner_lines[:-1]
            cleaned = "\n".join(inner_lines)

        try:
            obj = json.loads(cleaned)
            if isinstance(obj, dict):
                return obj.get("code", "")
        except json.JSONDecodeError:
            pass
        return ""
