"""Hungarian Matcher — DETR-inspired bipartite matching for evaluation.

Adapts the DETR (DEtection TRansformer) Hungarian matching idea to the
text generation evaluation domain:

In DETR, predictions are matched to ground-truth objects using the
Hungarian algorithm on a cost matrix, ensuring each prediction is matched
to at most one ground-truth (and vice versa).

Here, we match:
- Generated content sections (proposals) ↔ Expected quality criteria
- Or: Generated sections ↔ Retrieved evidence (faithfulness check)

The cost matrix is computed using the redesigned scoring function
(dense + sparse with denoising).

This enables:
- Automated, assignment-based evaluation (not just aggregate scores)
- Per-section quality assessment
- Foundation for auto-prompt optimization (we know WHICH sections are weak)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.core.logging import logger
from app.agents.evaluation.scoring import ScoringFunction


class HungarianMatcher:
    """Bipartite matching between generated content and quality criteria."""

    def __init__(
        self,
        scoring_fn: Optional[ScoringFunction] = None,
    ) -> None:
        self.scoring_fn = scoring_fn or ScoringFunction(
            sparse_weight=0.3,
            denoising_strength=0.1,
        )

    def match(
        self,
        predictions: List[Dict[str, Any]],
        targets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform Hungarian matching between predictions and targets.

        Args:
            predictions: List of dicts with 'embedding', 'tokens', 'content', 'id'.
            targets: List of dicts with 'embedding', 'tokens', 'content', 'id'.

        Returns:
            Dict with:
              - matches: list of (pred_idx, target_idx, score) tuples
              - unmatched_preds: list of prediction indices with no good match
              - unmatched_targets: list of target indices with no match
              - cost_matrix: the full cost matrix
              - overall_score: weighted average match quality
        """
        n_pred = len(predictions)
        n_targ = len(targets)

        if n_pred == 0 or n_targ == 0:
            return {
                "matches": [],
                "unmatched_preds": list(range(n_pred)),
                "unmatched_targets": list(range(n_targ)),
                "cost_matrix": np.array([]),
                "overall_score": 0.0,
            }

        # Compute score matrix (similarity, not cost)
        score_matrix = self.scoring_fn.batch_score(predictions, targets)

        # Convert to cost matrix (Hungarian minimizes cost)
        cost_matrix = 1.0 - score_matrix

        # Pad if non-square (Hungarian requires square or handles rectangular)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Build match results
        matches: List[Tuple[int, int, float]] = []
        matched_preds = set()
        matched_targets = set()
        total_score = 0.0

        for row, col in zip(row_indices, col_indices):
            if row < n_pred and col < n_targ:
                match_score = float(score_matrix[row, col])
                matches.append((int(row), int(col), round(match_score, 4)))
                matched_preds.add(int(row))
                matched_targets.add(int(col))
                total_score += match_score

        unmatched_preds = [i for i in range(n_pred) if i not in matched_preds]
        unmatched_targets = [i for i in range(n_targ) if i not in matched_targets]

        overall = total_score / max(len(matches), 1)

        logger.info(
            "hungarian_matching_complete",
            predictions=n_pred,
            targets=n_targ,
            matches=len(matches),
            unmatched_preds=len(unmatched_preds),
            overall_score=round(overall, 4),
        )

        return {
            "matches": matches,
            "unmatched_preds": unmatched_preds,
            "unmatched_targets": unmatched_targets,
            "cost_matrix": cost_matrix.tolist(),
            "score_matrix": score_matrix.tolist(),
            "overall_score": round(overall, 4),
        }

    def evaluate_faithfulness(
        self,
        generated_sections: List[Dict[str, Any]],
        evidence_sources: List[Dict[str, Any]],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Evaluate whether each generated section is faithful to evidence.

        Uses Hungarian matching to align generated sections with their
        most likely evidence sources, then checks alignment quality.
        """
        result = self.match(generated_sections, evidence_sources)

        faithful_count = 0
        section_scores: List[Dict[str, Any]] = []

        for pred_idx, targ_idx, score in result["matches"]:
            is_faithful = score >= threshold
            if is_faithful:
                faithful_count += 1
            section_scores.append({
                "generated_idx": pred_idx,
                "evidence_idx": targ_idx,
                "score": score,
                "is_faithful": is_faithful,
            })

        total = len(generated_sections)
        faithfulness_rate = faithful_count / max(total, 1)

        return {
            "faithfulness_rate": round(faithfulness_rate, 4),
            "section_scores": section_scores,
            "unmatched_sections": result["unmatched_preds"],
            "overall_score": result["overall_score"],
        }
