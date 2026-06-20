"""联网搜索封装 - 优先 Tavily 远程 MCP, 失败兜底裸 httpx.

设计:
- 集成 Tavily 官方远程 MCP 服务 (https://mcp.tavily.com/mcp/)
- 走 MCP 协议 (streamable-http) 调 tavily_search tool
- 复用 core.mcp_client.MCPClientPool (N 个 worker 连接池 + 轮询)
- 失败时降级到裸 httpx 直连 api.tavily.com/search
- 没 TAVILY_API_KEY 时直接走 httpx fallback

为什么用 MCP:
- MCP 是 Anthropic 推的协议标准, "即插即用" 理念
- 官方 server 免部署, 维护成本 0
- 简历亮点: "集成 Tavily 官方 MCP 服务"

为什么有 fallback:
- MCP 协议层多了 1 层依赖 (远程 server 可用性)
- httpx 直连更稳定 (单一依赖)
- 优雅降级: MCP 挂了用户感知不到

并发能力:
- MCPClientPool 内置 N 个 _HTTPWorkerClient + round-robin
- 5 份简历并发 batch 时, pool_size=3 真并发调 Tavily
- 跟之前简历解析 MCP HTTP pool 架构统一
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"
_TAVILY_TOOL_NAME = "tavily_search"
_POOL_SIZE = 3
_DEFAULT_TIMEOUT = 8.0  # 秒


def _is_available() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def _http_search(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """裸 httpx 直连 Tavily API (MCP 失败时的 fallback)."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []
    try:
        resp = httpx.post(
            _TAVILY_ENDPOINT,
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=_DEFAULT_TIMEOUT,
        )
        if resp.status_code != 200:
            logger.warning("Tavily httpx returned %s: %s", resp.status_code, resp.text[:200])
            return []
        data = resp.json()
        return data.get("results", [])
    except (httpx.TimeoutException, httpx.RequestError) as e:
        logger.warning("Tavily httpx failed: %s", str(e)[:200])
        return []
    except Exception as e:  # noqa: BLE001
        logger.warning("Tavily httpx unexpected error: %s", str(e)[:200])
        return []


# ========== Tavily MCP 连接池 (单例) ==========
_mcp_pool = None


def _get_tavily_mcp_pool():
    """获取 Tavily MCP 连接池 (懒加载单例, 没 key 返回 None)."""
    global _mcp_pool
    if _mcp_pool is not None:
        return _mcp_pool
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None
    from core.mcp_client import MCPClientPool
    url = f"{_TAVILY_MCP_URL}?tavilyApiKey={api_key}"
    _mcp_pool = MCPClientPool(url=url, pool_size=_POOL_SIZE)
    logger.info("Tavily MCP pool created (size=%d)", _POOL_SIZE)
    return _mcp_pool


def web_search(query: str, max_results: int = 3) -> list[dict[str, Any]]:
    """优先 Tavily MCP 连接池, 失败降级到裸 httpx.

    Returns:
        list of {"title", "url", "content", "score"} dicts
        失败/无 Key 时返回 []
    """
    if not _is_available():
        return []

    # 路径 1: Tavily 远程 MCP (复用 MCPClientPool, 支持并发)
    try:
        pool = _get_tavily_mcp_pool()
        if pool is not None:
            data = pool.call_tool_sync(_TAVILY_TOOL_NAME, {
                "query": query,
                "max_results": max_results,
            })
            results = data.get("results", [])
            if results:
                logger.debug("Tavily MCP returned %d results", len(results))
                return results
    except Exception as e:
        logger.warning("Tavily MCP failed (%s), falling back to httpx", str(e)[:200])

    # 路径 2: 裸 httpx fallback
    return _http_search(query, max_results)


def search_company_info(company: str | None, role: str | None) -> str:
    """搜索公司+岗位信息, 返回拼接好的文本 (给 LLM 用作 context).

    输入: company="字节跳动", role="Python 后端工程师"
    输出: 类似 "联网搜索结果: ..."
    失败时: 返回空字符串
    """
    if not company or not role:
        return ""

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
                snippets.append(content[:200])
    if not snippets:
        return ""
    return "联网搜索结果:\n" + "\n---\n".join(snippets[:3])