"""web_searcher.py 的单元测试 - 用 monkeypatch mock httpx, 不真调 Tavily."""
import os
from unittest.mock import patch, MagicMock

import pytest

from agents.web_searcher import _is_available, search_company_info, web_search


class TestWebSearcher:
    def test_is_available_with_key(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        assert _is_available() is True

    def test_is_available_without_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert _is_available() is False

    def test_is_available_empty_key(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "   ")
        assert _is_available() is False

    def test_web_search_no_key(self, monkeypatch):
        """无 Key 时返回空列表, 不报错."""
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        assert web_search("test query") == []

    def test_web_search_success(self, monkeypatch):
        """正常返回 Tavily 结果."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"title": "字节跳动官网", "url": "https://bytedance.com", "content": "ByteDance is a tech company", "score": 0.9}
            ]
        }
        with patch("agents.web_searcher.httpx.post", return_value=mock_resp) as mock_post:
            results = web_search("字节跳动 Python", max_results=2)
        assert len(results) == 1
        assert results[0]["title"] == "字节跳动官网"
        # 验证请求 payload
        call_args = mock_post.call_args
        assert call_args.kwargs["json"]["query"] == "字节跳动 Python"
        assert call_args.kwargs["json"]["api_key"] == "test-key"

    def test_web_search_http_error(self, monkeypatch):
        """HTTP 非 200 时返回空列表."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "rate limit"
        with patch("agents.web_searcher.httpx.post", return_value=mock_resp):
            assert web_search("test") == []

    def test_web_search_timeout(self, monkeypatch):
        """httpx 超时时返回空列表, 不抛异常."""
        import httpx
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        with patch("agents.web_searcher.httpx.post", side_effect=httpx.TimeoutException("timeout")):
            assert web_search("test") == []

    def test_search_company_info_empty_inputs(self, monkeypatch):
        """company 或 role 为空时返回空字符串."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        assert search_company_info(None, "Python") == ""
        assert search_company_info("字节", None) == ""
        assert search_company_info("", "") == ""

    def test_search_company_info_no_results(self, monkeypatch):
        """搜索没结果时返回空字符串."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"results": []}
        with patch("agents.web_searcher.httpx.post", return_value=mock_resp):
            result = search_company_info("字节跳动", "Python 后端")
        assert result == ""

    def test_search_company_info_with_results(self, monkeypatch):
        """有结果时返回格式化文本."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [
                {"title": "字节招聘", "content": "ByteDance hires Python engineers with FastAPI experience. Salary 30-50K."}
            ]
        }
        with patch("agents.web_searcher.httpx.post", return_value=mock_resp):
            result = search_company_info("字节跳动", "Python 后端")
        assert "联网搜索结果" in result
        assert "FastAPI" in result
        assert "30-50K" in result

    def test_search_company_info_content_truncated(self, monkeypatch):
        """长 content 被截断到 200 字."""
        monkeypatch.setenv("TAVILY_API_KEY", "test-key")
        long_content = "x" * 1000
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": [{"title": "t", "content": long_content}]
        }
        with patch("agents.web_searcher.httpx.post", return_value=mock_resp):
            result = search_company_info("C", "R")
        # 检查 result 中没超过 200 字符的 content
        assert "x" * 200 in result
        assert "x" * 201 not in result