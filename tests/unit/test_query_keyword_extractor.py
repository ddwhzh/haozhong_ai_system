"""Unit tests for unified query keyword extractor."""

from app.core.nlp import query_keyword_extractor


class TestQueryKeywordExtractor:
    """Verify extraction and matching behavior."""

    def test_extract_keywords_sync_for_comparison_query(self):
        query = "鹿紫云能不能打过五条悟"
        keywords = query_keyword_extractor.extract_keywords_sync(query, max_keywords=10)
        assert "鹿紫云" in keywords
        assert "五条悟" in keywords
        assert "鹿紫云能不能打过五条悟" not in keywords

    def test_extract_keywords_dedup_and_limit(self):
        query = "五条悟和五条悟谁更强"
        keywords = query_keyword_extractor.extract_keywords_sync(query, max_keywords=3)
        assert len(keywords) <= 3
        assert keywords.count("五条悟") <= 1

    def test_match_keywords_alias_tolerance(self):
        assert query_keyword_extractor.match_keywords(
            ["鹿紫云一"], "鹿紫云被称为雷神"
        )

    def test_match_keywords_false_when_unrelated(self):
        assert not query_keyword_extractor.match_keywords(
            ["咒术回战"], "机器学习与数据库"
        )
