"""matcher.py 中 enable_web_search 集成路径的单元测试.

注意: 这些测试 mock 掉 web_searcher, 不真调 Tavily.
"""
from unittest.mock import patch, MagicMock

import pytest

from agents.matcher import MatcherConfig, match_resume_to_jd
from agents.web_searcher import search_company_info
from core.schemas import (
    JDRequirement, MatchDimension, MatchResult,
    ParsedJD, ParsedResume,
)


def _make_jd():
    return ParsedJD(
        job_title="Python 后端工程师",
        company="字节跳动",
        summary="3 年 Python 经验要求, 熟悉 FastAPI, PostgreSQL, Docker.",
        requirements=[
            JDRequirement(category="required_skill", description="3 years Python", weight=9, is_must_have=True),
            JDRequirement(category="required_skill", description="FastAPI experience", weight=8, is_must_have=True),
        ],
    )


def _make_resume():
    return ParsedResume(
        candidate_name="测试候选人",
        years_of_experience=3,
        skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
    )


class TestWebSearchIntegration:
    def test_enable_web_search_passes_context(self):
        """enable_web_search=True 时, search_company_info 被调用并 context 传入 prompt."""
        # 跑通整个 match_resume_to_jd 会调 LLM, 我们只验证 search_company_info 被调用
        # 实际跑 LLM 容易失败, 改用直接 patch structured_call
        jd = _make_jd()
        resume = _make_resume()

        mock_result = MatchResult(
            overall_score=85,
            dimensions=[
                MatchDimension(requirement="x", candidate_evidence="y", score=8, reasoning="ok")
            ],
            strengths=["strong"],
            weaknesses=["weak"],
            confidence="high",
        )

        with patch("agents.web_searcher.search_company_info", return_value="字节跳动 30-50K FastAPI") as mock_search:
            with patch("agents.matcher.structured_call", return_value=mock_result) as mock_sc:
                match_resume_to_jd(jd, resume, config=MatcherConfig(enable_web_search=True, enable_reflection=False))

        # 验证 search_company_info 被调
        mock_search.assert_called_once()
        # 验证 structured_call 收到的 user prompt 包含 web context
        call_args = mock_sc.call_args
        user_prompt = call_args.kwargs["user"]
        assert "联网搜索补充信息" in user_prompt
        assert "字节跳动 30-50K FastAPI" in user_prompt

    def test_disable_web_search_no_context(self):
        """enable_web_search=False 时, search_company_info 不被调用, prompt 不含 web context."""
        jd = _make_jd()
        resume = _make_resume()
        mock_result = MatchResult(
            overall_score=85,
            dimensions=[MatchDimension(requirement="x", candidate_evidence="y", score=8, reasoning="ok")],
            strengths=[], weaknesses=[], confidence="high",
        )

        with patch("agents.web_searcher.search_company_info") as mock_search:
            with patch("agents.matcher.structured_call", return_value=mock_result) as mock_sc:
                match_resume_to_jd(jd, resume, config=MatcherConfig(enable_web_search=False, enable_reflection=False))

        mock_search.assert_not_called()
        user_prompt = mock_sc.call_args.kwargs["user"]
        assert "联网搜索" not in user_prompt

    def test_web_search_empty_result_no_context(self):
        """search_company_info 返回空字符串时, prompt 也不加 web section."""
        jd = _make_jd()
        resume = _make_resume()
        mock_result = MatchResult(
            overall_score=85,
            dimensions=[MatchDimension(requirement="x", candidate_evidence="y", score=8, reasoning="ok")],
            strengths=[], weaknesses=[], confidence="high",
        )

        with patch("agents.web_searcher.search_company_info", return_value="") as mock_search:
            with patch("agents.matcher.structured_call", return_value=mock_result) as mock_sc:
                match_resume_to_jd(jd, resume, config=MatcherConfig(enable_web_search=True, enable_reflection=False))

        user_prompt = mock_sc.call_args.kwargs["user"]
        # search 被调了 (因为 enable_web_search=True)
        mock_search.assert_called_once()
        # 但 prompt 里没有 "联网搜索补充信息" 标识 (空字符串)
        assert "联网搜索补充信息" not in user_prompt