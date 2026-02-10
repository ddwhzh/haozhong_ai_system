"""Unified query keyword extraction utilities.

Supports:
1. Local extraction (regex + jieba + stopwordsiso)
2. Optional LLM-assisted extraction (configurable)
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Iterable, List, Set

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.core.logging import logger
from app.infra.llm import llm_service
from app.prompts import load_prompt

try:
    import jieba
except Exception:  # pragma: no cover - fallback path if package unavailable
    jieba = None

try:
    import stopwordsiso
except Exception:  # pragma: no cover - fallback path if package unavailable
    stopwordsiso = None


_ZH_SPLIT_PATTERN = re.compile(
    r"(能不能|是否|可不可以|可否|会不会|打不打得过|打得过|打不过|打过|"
    r"谁更强|谁更厉害|强不强|厉不厉害|和|与|跟|对|比|及|还是|或者|并且|以及)"
)

_BASE_ZH_STOPWORDS = {
    "什么", "如何", "为什么", "哪些", "关系", "影响", "作用", "定义",
    "数据", "情况", "问题", "分析", "中国", "地区", "领域", "能不能",
    "是否", "打过", "打不过", "可以", "会不会",
}


def _is_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


@lru_cache(maxsize=4)
def _load_stopwords(lang: str) -> Set[str]:
    """Load stopwords from stopwordsiso with local fallback."""
    if stopwordsiso is not None:
        try:
            words = stopwordsiso.stopwords(lang)
            if words:
                return set(words) | (_BASE_ZH_STOPWORDS if lang == "zh" else set())
        except Exception as e:
            logger.warning("stopwordsiso_load_failed", lang=lang, error=str(e))
    return _BASE_ZH_STOPWORDS if lang == "zh" else set()


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class QueryKeywordExtractor:
    """Extract and match query keywords with optional LLM assistance."""

    async def extract_keywords(
        self,
        query: str,
        use_llm: bool = False,
        max_keywords: int | None = None,
    ) -> List[str]:
        """Extract keywords, optionally using LLM first."""
        limit = max_keywords or settings.KEYWORD_EXTRACTION_MAX_TERMS

        if use_llm:
            llm_keywords = await self._extract_keywords_with_llm(query=query, max_keywords=limit)
            if llm_keywords:
                return llm_keywords[:limit]

        return self.extract_keywords_sync(query=query, max_keywords=limit)

    def extract_keywords_sync(
        self,
        query: str,
        max_keywords: int | None = None,
    ) -> List[str]:
        """Local keyword extraction using stopword package and tokenizer."""
        if not query:
            return []

        limit = max_keywords or settings.KEYWORD_EXTRACTION_MAX_TERMS
        normalized = re.sub(r"[，。！？、,:;；()\[\]{}\-_/]+", " ", query)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return []

        terms: List[str] = []

        # English-ish tokens
        terms.extend(re.findall(r"[A-Za-z0-9_]{3,}", normalized))

        # Chinese chunks and split by relation/comparison connectors.
        zh_chunks = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
        for chunk in zh_chunks:
            parts = re.split(_ZH_SPLIT_PATTERN, chunk)
            for part in parts:
                if not part:
                    continue
                if re.fullmatch(_ZH_SPLIT_PATTERN, part):
                    continue
                part = part.strip()
                if len(part) >= 2:
                    terms.append(part)

        # Additional segmentation from jieba (if installed).
        if jieba is not None and _is_zh(normalized):
            try:
                for token in jieba.lcut(normalized, HMM=False):
                    token = token.strip()
                    if len(token) >= 2 and _is_zh(token):
                        terms.append(token)
            except Exception as e:
                logger.warning("jieba_cut_failed", error=str(e), query=normalized[:120])

        stopwords_zh = _load_stopwords("zh")
        keywords: List[str] = []
        for term in terms:
            t = term.strip()
            if not t:
                continue
            if _is_zh(t):
                if t in stopwords_zh:
                    continue
            keywords.append(t)

        keywords = _dedupe_keep_order(keywords)

        # Last-resort fallback: keep Chinese chunks if everything filtered out.
        if not keywords:
            keywords = _dedupe_keep_order([c for c in zh_chunks if c.strip()])

        return keywords[:limit]

    @staticmethod
    def match_keywords(keywords: List[str], text: str) -> bool:
        """Check if evidence text matches at least one keyword."""
        if not keywords:
            return True

        haystack = (text or "").lower()
        for kw in keywords:
            kw_norm = (kw or "").strip().lower()
            if not kw_norm:
                continue
            if kw_norm in haystack:
                return True

            # Chinese alias tolerance for one-char mismatch:
            # e.g. 鹿紫云 vs 鹿紫云一
            if _is_zh(kw_norm) and len(kw_norm) >= 3:
                if kw_norm[:-1] in haystack or kw_norm[1:] in haystack:
                    return True
        return False

    async def _extract_keywords_with_llm(
        self,
        query: str,
        max_keywords: int,
    ) -> List[str]:
        """LLM-assisted query keyword extraction."""
        if not query:
            return []

        try:
            messages = [
                SystemMessage(content=load_prompt(
                    "query_keyword_extract_system",
                    max_keywords=max_keywords,
                )),
                HumanMessage(content=f"Query:\n{query}"),
            ]
            response = await llm_service.call(
                messages,
                prompt_name="query_keyword_extract_system",
            )
            raw = response.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                return []

            keywords = []
            for item in parsed:
                if isinstance(item, str):
                    t = item.strip()
                    if t:
                        keywords.append(t)
            return _dedupe_keep_order(keywords)[:max_keywords]

        except Exception as e:
            logger.warning(
                "llm_keyword_extraction_failed",
                error=str(e),
                query=query[:120],
            )
            return []


query_keyword_extractor = QueryKeywordExtractor()
