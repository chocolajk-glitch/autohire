"""联网搜索封装 (Tavily) - 给匹配 Agent 提供外部知识.

特点:
- 优雅降级: Key 缺失 / API 失败 / 超时 都不报错, 返回空
- 单一搜索接口: web_search(query, max_results=3) -> list[dict]
- 给 matcher 用的辅助函数: search_company_info(company, role) -> str (拼接好的 context 文本)
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_DEFAULT_TIMEOUT = 8.0  # 秒, 联网搜索不应阻塞主流程太久


def _is_available() -> bool:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    return bool(api_key)


def web_search(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """用 Tavily 搜一个查询, 返回结果列表 (每个是 dict).

    Returns:
        list of {"title": str, "url": str, "content": str, "score": float}
        失败/无 Key 时返回 []
    """
    if not _is_available():
        logger.debug("TAVILY_API_KEY not set, skipping web search")
        return []
    api_key = os.getenv("TAVILY_API_KEY")
    try:
        resp = httpx.post(
            _TAVILY_ENDPOINT,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",  # basic 比 advanced 快
                "include_answer": False,
            },
            timeout=_DEFAULT_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("Tavily returned %s: %s", resp.status_code, resp.text[:200])
            return []
        data = resp.json()
        return data.get("results", [])
    except (httpx.TimeoutException, httpx.RequestError) as e:
        logger.warning("Tavily request failed: %s", str(e)[:200])
        return []
    except Exception as e:  # noqa: BLE001
        logger.warning("Tavily unexpected error: %s", str(e)[:200])
        return []


def search_company_info(company: str | None, role: str | None) -> str:
    """搜索公司+岗位信息, 返回拼接好的文本 (给 LLM 用作 context).

    输入: company="字节跳动", role="Python 后端工程师"
    输出: 类似 "公司最新动态: ... 技术栈关键词: ..."
    失败时: 返回空字符串
    """
    if not company or not role:
        return ""

    # 两条搜索: 公司最新动态 + 岗位技术栈
    queries = [
        f"{company} {role} 技术栈 招聘要求",
        f"{company} 最新技术 薪资",
    ]
    snippets: list[str] = []
    for q in queries:
        results = web_search(q, max_results=2)
        for r in results:
            content = r.get("content", "").strip()
            if content:
                # 截断到 200 字避免 prompt 爆炸
                snippets.append(content[:200])
    if not snippets:
        return ""
    return "联网搜索结果:\n" + "\n---\n".join(snippets[:3])