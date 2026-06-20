"""MCP HTTP 连接池 - 通过 streamable-http 协议连接 MCP server.

设计原则:
- 不再封装自己的 MCP server (resume_server.py 已删除, 用函数直接调更合适)
- 保留 HTTP pool 实现, 用于接入外部 MCP 服务 (如 Tavily 远程 MCP)
- pool 维护 N 个长连接到同一个 MCP server URL, 轮询分配请求

典型用法 (Tavily 远程 MCP):
    from core.mcp_client import MCPClientPool
    pool = MCPClientPool(
        url_template="https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}",
        pool_size=2,
        api_key="tvly-xxx",
    )
    result = pool.call_tool_sync("tavily_search", {"query": "...", "max_results": 3})
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class MCPClientError(RuntimeError):
    pass


class _HTTPWorkerClient:
    """单个 HTTP worker, 长期持有 streamable-http 连接."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._cm = None
        self._ready = False
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._lock:
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
            logger.info("MCP worker connected: %s", self.url[:80])

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """同步调用 MCP tool, 返回 dict. 异常向上抛 (由调用方决定是否 fallback)."""
        self._ensure_started()
        assert self._session is not None and self._loop is not None
        try:
            result = self._loop.run_until_complete(
                self._session.call_tool(tool_name, arguments=arguments)
            )
        except Exception as e:
            raise MCPClientError(f"MCP worker call {tool_name} failed: {e}") from e

        if getattr(result, "isError", False):
            err_text = result.content[0].text if result.content else "未知错误"
            raise MCPClientError(f"MCP tool {tool_name} error: {err_text}")

        if not result.content:
            raise MCPClientError(f"MCP tool {tool_name} returned empty content")
        try:
            data = json.loads(result.content[0].text)
        except json.JSONDecodeError as e:
            raise MCPClientError(
                f"MCP tool {tool_name} returned non-JSON: {result.content[0].text[:200]}"
            ) from e
        if not isinstance(data, dict):
            raise MCPClientError(f"MCP tool {tool_name} returned non-dict: {type(data)}")
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
            logger.warning("MCP worker close error: %s", e)
        self._ready = False

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MCPClientPool:
    """N 个 HTTP worker 的 MCP 连接池, 轮询分配请求.

    Args:
        url: 完整的 MCP server URL (含 query params, 如 API key)
        pool_size: worker 数量, 默认 2
    """

    def __init__(self, url: str, pool_size: int = 2) -> None:
        self.url = url
        self.pool_size = pool_size
        self._workers = [_HTTPWorkerClient(url) for _ in range(pool_size)]
        self._rr = 0
        self._rr_lock = threading.Lock()

    def _next_worker(self) -> _HTTPWorkerClient:
        with self._rr_lock:
            w = self._workers[self._rr % self.pool_size]
            self._rr += 1
        return w

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """轮询分配到下一个 worker. 单 worker 失败不重试 (上层 fallback 兜底)."""
        return self._next_worker().call_tool_sync(tool_name, arguments)

    def close(self) -> None:
        for w in self._workers:
            w.close()