"""Content Quality Pre-filter — unsupervised gibberish/random content detection.

Detects and scores content quality using statistical features only (no LLM
calls). Designed to catch random character sequences that can fool
embedding-based similarity due to high-dimensional "hub" effects.

Features:
1. Character-level Shannon entropy (extreme high/low = suspicious)
2. Character class consistency (CJK/Latin/symbol mixing ratio)
3. Bigram repetition rate (real text has natural bigram repetition)
4. Valid structure ratio (reasonable punctuation, sentence length)

Output: quality score in [0, 1]. Score < threshold triggers penalty
on the embedding similarity score during Hungarian matching.

Implements FR-6 from evaluation agent requirements.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import List

from app.core.config import settings
from app.core.logging import logger

# Unicode category ranges for character class detection
_CJK_RANGES = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff"
    r"\U00020000-\U0002a6df\U0002a700-\U0002ebef]"
)
_LATIN_RANGE = re.compile(r"[a-zA-Z]")
_DIGIT_RANGE = re.compile(r"[0-9]")
_PUNCT_RANGE = re.compile(
    r"[,，.。;；:：!！?？、\-—–\(\)（）\[\]【】\"\'""''…·\n\r\t ]"
)


@dataclass
class ContentQualityResult:
    """Result of content quality assessment."""

    quality_score: float
    char_entropy_score: float
    char_class_score: float
    bigram_repetition_score: float
    structure_score: float
    is_low_quality: bool
    reason: str


def assess_content_quality(
    text: str,
    quality_threshold: float | None = None,
) -> ContentQualityResult:
    """Assess content quality using unsupervised statistical features.

    Args:
        text: The text content to assess.
        quality_threshold: Threshold below which content is considered
            low quality. Uses EVALUATION_QUALITY_THRESHOLD from settings
            if not provided.

    Returns:
        ContentQualityResult with component scores and overall quality.
    """
    if quality_threshold is None:
        quality_threshold = getattr(settings, "EVALUATION_QUALITY_THRESHOLD", 0.4)

    if not text or not text.strip():
        return ContentQualityResult(
            quality_score=0.0,
            char_entropy_score=0.0,
            char_class_score=0.0,
            bigram_repetition_score=0.0,
            structure_score=0.0,
            is_low_quality=True,
            reason="empty_content",
        )

    stripped = text.strip()

    # Very short content gets a pass-through (not enough signal)
    if len(stripped) < 5:
        return ContentQualityResult(
            quality_score=0.5,
            char_entropy_score=0.5,
            char_class_score=0.5,
            bigram_repetition_score=0.5,
            structure_score=0.5,
            is_low_quality=False,
            reason="too_short_for_assessment",
        )

    # ── Component scores ─────────────────────────────────────────────
    is_cjk_dominated = _is_cjk_dominated(stripped)

    entropy_score = _character_entropy_score(stripped)
    char_class = _character_class_score(stripped)
    bigram_rep = _bigram_repetition_score(stripped, is_cjk=is_cjk_dominated)
    structure = _structure_score(stripped)

    # ── Weighted combination ─────────────────────────────────────────
    quality = (
        0.30 * entropy_score
        + 0.30 * char_class
        + 0.20 * bigram_rep
        + 0.20 * structure
    )
    quality = max(0.0, min(1.0, quality))

    # Gate 1: if any single critical signal is extremely bad (e.g.,
    # pure repetition where entropy → 0), hard-cap quality.
    min_critical = min(entropy_score, char_class)
    if min_critical < 0.10:
        quality = min(quality, 0.20 + min_critical)

    # Gate 2: if two or more signals are weak, it's very likely
    # gibberish even if the other signals are fine. This catches
    # random alphanumeric strings (valid chars but no structure)
    # and mixed gibberish (some valid chars + random symbols).
    all_scores = sorted([entropy_score, char_class, bigram_rep, structure])
    two_lowest_avg = (all_scores[0] + all_scores[1]) / 2.0
    if two_lowest_avg < 0.30:
        quality = min(quality, two_lowest_avg + 0.10)

    # Gate 3: detect degenerate phrase repetition (e.g., "知识图谱"
    # repeated 5 times). This is a different failure mode from random
    # chars but equally problematic for quality.
    phrase_penalty = _phrase_repetition_penalty(stripped)
    if phrase_penalty < 1.0:
        quality = quality * phrase_penalty

    is_low = quality < quality_threshold

    reason = "normal"
    if is_low:
        weakest = min(
            ("entropy", entropy_score),
            ("char_class", char_class),
            ("bigram", bigram_rep),
            ("structure", structure),
            key=lambda x: x[1],
        )
        reason = f"low_quality_{weakest[0]}={weakest[1]:.2f}"

    return ContentQualityResult(
        quality_score=round(quality, 4),
        char_entropy_score=round(entropy_score, 4),
        char_class_score=round(char_class, 4),
        bigram_repetition_score=round(bigram_rep, 4),
        structure_score=round(structure, 4),
        is_low_quality=is_low,
        reason=reason,
    )


def apply_quality_penalty(
    similarity_score: float,
    quality_result: ContentQualityResult,
) -> float:
    """Apply quality-based penalty to a similarity score.

    If content quality is below threshold, the similarity score is
    decayed proportionally: penalized = original * quality_score.
    """
    if quality_result.is_low_quality:
        return similarity_score * quality_result.quality_score
    return similarity_score


# ── Component scoring functions ──────────────────────────────────────────


def _character_entropy_score(text: str) -> float:
    """Score based on character-level Shannon entropy.

    Real text has moderate entropy (structured but varied).
    Random gibberish has very high entropy (near uniform distribution).
    Single-character repetition has very low entropy.

    Maps entropy to a score where moderate = high score, extremes = low.
    """
    char_counts = Counter(text)
    total = len(text)

    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    # Maximum possible entropy for this alphabet size
    unique_chars = len(char_counts)
    max_entropy = math.log2(max(unique_chars, 2))

    # Normalized entropy ratio
    if max_entropy > 0:
        normalized = entropy / max_entropy
    else:
        return 0.0

    # Expected range for real text: 0.4 - 0.85 normalized entropy
    # Random text: > 0.90 normalized entropy
    # Repetitive text: < 0.20 normalized entropy
    if 0.35 <= normalized <= 0.90:
        return 1.0
    elif normalized > 0.90:
        # High entropy penalty (increasingly suspicious)
        return max(0.0, 1.0 - (normalized - 0.90) * 5.0)
    else:
        # Low entropy penalty (repetitive)
        return max(0.0, normalized / 0.35)


def _character_class_score(text: str) -> float:
    """Score based on character class distribution consistency.

    Real text is dominated by one character class (CJK for Chinese,
    Latin for English) with reasonable punctuation. Random text often
    has erratic mixing of character classes or excessive symbols.
    """
    chars = re.sub(r"\s+", "", text)
    if not chars:
        return 0.0

    total = len(chars)
    cjk_count = len(_CJK_RANGES.findall(chars))
    latin_count = len(_LATIN_RANGE.findall(chars))
    digit_count = len(_DIGIT_RANGE.findall(chars))
    punct_count = len(_PUNCT_RANGE.findall(chars))
    other_count = total - cjk_count - latin_count - digit_count - punct_count

    meaningful = cjk_count + latin_count + digit_count + punct_count
    meaningful_ratio = meaningful / max(total, 1)
    other_ratio = other_count / max(total, 1)

    # High "other" ratio = random symbols (@#$%^&* etc.)
    if other_ratio > 0.30:
        return max(0.0, 0.3 * (1.0 - other_ratio))

    if meaningful_ratio >= 0.70 and other_ratio <= 0.15:
        return 1.0
    elif meaningful_ratio >= 0.50:
        return min(1.0, 0.4 + 0.6 * (meaningful_ratio - 0.50) / 0.20)
    else:
        return max(0.0, meaningful_ratio * 0.8)


def _is_cjk_dominated(text: str) -> bool:
    """Check if text is primarily CJK characters (>40% of non-whitespace)."""
    chars = re.sub(r"\s+", "", text)
    if not chars:
        return False
    cjk_count = len(_CJK_RANGES.findall(chars))
    return cjk_count / len(chars) > 0.40


def _bigram_repetition_score(text: str, *, is_cjk: bool = False) -> float:
    """Score based on bigram diversity.

    CJK text has naturally very high bigram diversity (each char-pair
    is nearly unique), so for CJK we only detect pathological repetition
    (very low diversity). For Latin text, both extremes are penalized.
    """
    chars = list(text.replace(" ", ""))
    if len(chars) < 4:
        return 0.5

    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    total_bigrams = len(bigrams)
    unique_bigrams = len(set(bigrams))

    if total_bigrams == 0:
        return 0.0

    diversity = unique_bigrams / total_bigrams

    if is_cjk:
        # CJK: penalize low diversity (pathological or phrase repetition).
        # Normal CJK naturally has diversity ~0.85-0.99.
        # Phrase repetition ("知识图谱知识图谱"): diversity ~0.15-0.35.
        # Single-char repetition ("啊啊啊啊"): diversity < 0.10.
        if diversity < 0.40:
            return max(0.0, diversity / 0.40)
        return 1.0
    else:
        # Latin: penalize both extremes
        if 0.15 <= diversity <= 0.90:
            return 1.0
        elif diversity < 0.15:
            return max(0.0, diversity / 0.15)
        else:
            return max(0.0, 1.0 - (diversity - 0.90) * 10.0)


def _structure_score(text: str) -> float:
    """Score based on text structural features.

    Real text has:
    - Reasonable average segment length (split by punctuation/spaces)
    - Some variation in segment lengths
    - Not all-caps or all-symbols
    """
    # Split into segments by sentence-ending punctuation
    segments = re.split(r"[.。!！?？;；\n]+", text)
    segments = [s.strip() for s in segments if s.strip()]

    if not segments:
        return 0.0

    avg_len = sum(len(s) for s in segments) / len(segments)

    # Real text segments: typically 5-200 chars
    if 3 <= avg_len <= 300:
        length_score = 1.0
    elif avg_len < 3:
        length_score = avg_len / 3.0
    else:
        length_score = max(0.0, 1.0 - (avg_len - 300) / 500)

    # Check if text has any sentence-like structure at all
    has_punctuation = bool(re.search(r"[,，.。;；:：!！?？]", text))
    punct_bonus = 1.0 if has_punctuation else 0.6

    return min(1.0, length_score * punct_bonus)


def _phrase_repetition_penalty(text: str) -> float:
    """Detect degenerate phrase repetition in text.

    Checks if a short phrase (2-10 chars) is repeated so many times
    that it dominates the text. Returns 1.0 (no penalty) for normal
    text, or a value < 1.0 for pathological repetition.
    """
    text_len = len(text)
    if text_len < 6:
        return 1.0

    max_phrase_len = min(11, text_len // 2)
    for phrase_len in range(2, max_phrase_len):
        phrase = text[:phrase_len]
        count = text.count(phrase)
        coverage = (count * phrase_len) / text_len
        if count >= 3 and coverage > 0.70:
            return 0.3
    return 1.0


def batch_assess_quality(
    texts: List[str],
    quality_threshold: float | None = None,
) -> List[ContentQualityResult]:
    """Assess quality for a batch of texts."""
    return [assess_content_quality(t, quality_threshold) for t in texts]
