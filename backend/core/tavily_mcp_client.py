"""Tavily 远程 MCP 客户端 - 集成 Tavily 官方远程 MCP 服务.

URL: https://mcp.tavily.com/mcp/?tavilyApiKey=YOUR_KEY
官方文档: https://docs.tavily.com/documentation/mcp

设计选择:
- 用远程 MCP 而非自建 server: 官方维护, 免部署, "MCP 即插即用" 理念
- 用 streamable-http transport: 远程 server 只能用 HTTP
- 单例模式: Tavily 调用 1-2s, 不需要连接池
- 失败由 web_searcher fallback 到裸 httpx

工具列表 (远程 server 暴露):
- tavily_search   综合网络搜索
- tavily_extract  URL 内容提取
- tavily_crawl    网站爬取
- tavily_map      网站结构映射
- tavily_research 深度研究
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"


def build_tavily_mcp_url(api_key: str) -> str:
    """构造 Tavily MCP URL (key 放在 query string)."""
    return f"{TAVILY_MCP_URL}?tavilyApiKey={api_key}"


class TavilyMCPClient:
    """Tavily 官方远程 MCP 客户端 (单例).

    同步接口 search() 内部用持久 event loop, 跟 stdio client 设计一致.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY", "")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY 未设置, 无法连接远程 MCP server")
        self.url = build_tavily_mcp_url(self.api_key)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._cm = None
        self._ready = False

    def _ensure_started(self) -> None:
        if self._ready:
            return
        from mcp.client.session import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        self._loop = asyncio.new_event_loop()
        self._cm = streamablehttp_client(self.url, timeout=30.0)
        read, write, _ = self._loop.run_until_complete(self._cm.__aenter__())
        self._session = ClientSession(read, write)
        self._loop.run_until_complete(self._session.__aenter__())
        self._loop.run_until_complete(self._session.initialize())
        self._ready = True
        logger.info("Tavily MCP client connected (remote)")

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """同步调用 MCP tool, 返回 dict. 失败抛异常 (由调用方 fallback)."""
        self._ensure_started()
        assert self._session is not None and self._loop is not None
        try:
            result = self._loop.run_until_complete(
                self._session.call_tool(tool_name, arguments=arguments)
            )
        except Exception as e:
            raise RuntimeError(f"Tavily MCP call {tool_name} failed: {e}") from e

        if getattr(result, "isError", False):
            err_text = result.content[0].text if result.content else "未知错误"
            raise RuntimeError(f"Tavily MCP {tool_name} error: {err_text}")

        if not result.content:
            raise RuntimeError(f"Tavily MCP {tool_name} returned empty content")

        import json
        try:
            data = json.loads(result.content[0].text)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Tavily MCP {tool_name} non-JSON: {result.content[0].text[:200]}"
            ) from e
        if not isinstance(data, dict):
            raise RuntimeError(f"Tavily MCP {tool_name} non-dict: {type(data)}")
        return data

    def close(self) -> None:
        if not self._ready:
            return
        try:
            if self._session and self._loop:
                self._loop.run_until_complete(self._session.__aexit__(None, None, None))
            if self._cm and self._loop:
                self._loop.run_until_complete(self._cm.__aexit__(None, None, None))
            if self._loop:
                self._loop.close()
        except Exception as e:
            logger.warning("Tavily MCP close error: %s", e)
        self._ready = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ========== 单例 (懒加载) ==========
_singleton: TavilyMCPClient | None = None


def get_tavily_mcp_client() -> TavilyMCPClient | None:
    """获取全局 Tavily MCP 客户端单例. 没 key 返回 None."""
    global _singleton
    if _singleton is None:
        try:
            _singleton = TavilyMCPClient()
        except ValueError as e:
            logger.debug("Tavily MCP client not initialized: %s", e)
    return _singleton