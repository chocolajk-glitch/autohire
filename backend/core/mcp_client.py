"""MCP 客户端 wrapper - 在主 AutoHire 进程内连接 MCP 服务.

两种传输模式 (由 MCP_TRANSPORT 环境变量控制, 默认 stdio):
1. stdio (默认): 单 client 单例, 懒启动子进程, 串行调用
2. http:  N 个 HTTP client 连接池, supervisor 管理 N 个子进程, 轮询分配请求

特点:
- 懒启动: 第一次调用才连 (stdio 起子进程 / http 等 supervisor)
- 自动管理生命周期: 程序退出时关闭
- 提供 sync 接口 (在事件循环外也能用)
- 优雅降级: MCP 调用失败 → 本地直接调用 (use_mcp=False 防递归)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_SCRIPT = BACKEND_ROOT / "mcp_servers" / "resume_server.py"
PYTHON_EXE = Path(sys.executable)


class MCPClientError(RuntimeError):
    pass


# ============================================================
# Stdio 单 client (向后兼容, 默认模式)
# ============================================================
class ResumeMCPClient:
    """懒启动 + 同步/异步双接口的 stdio MCP 客户端."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._ready = False
        self._closed = False

    def _ensure_started(self) -> None:
        if self._closed:
            raise MCPClientError("MCP client has been closed; create a new one")
        if self._ready:
            return
        if not MCP_SERVER_SCRIPT.exists():
            raise MCPClientError(f"MCP server script not found: {MCP_SERVER_SCRIPT}")

        from mcp.client.stdio import stdio_client, StdioServerParameters
        from mcp.client.session import ClientSession

        params = StdioServerParameters(
            command=str(PYTHON_EXE),
            args=["-m", "mcp_servers.resume_server", "--transport", "stdio"],
            cwd=str(BACKEND_ROOT),
        )
        self._loop = asyncio.new_event_loop()
        cm = stdio_client(params)
        read, write = self._loop.run_until_complete(cm.__aenter__())
        self._stdio_cm = cm
        self._read = read
        self._write = write
        self._session = ClientSession(read, write)
        self._loop.run_until_complete(self._session.__aenter__())
        self._loop.run_until_complete(self._session.initialize())
        self._ready = True
        logger.info("MCP stdio client connected to resume server")

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_started()
        assert self._session is not None
        assert self._loop is not None
        try:
            result = self._loop.run_until_complete(
                self._session.call_tool(tool_name, arguments=arguments)
            )
        except Exception as e:
            raise MCPClientError(f"MCP 工具 {tool_name} 执行失败: {e}") from e

        if getattr(result, "isError", False):
            err_text = result.content[0].text if result.content else "未知错误"
            raise MCPClientError(f"MCP 工具 {tool_name} 报错: {err_text}")

        if not result.content:
            raise MCPClientError(f"MCP 工具 {tool_name} 返回空内容")
        try:
            data = json.loads(result.content[0].text)
        except json.JSONDecodeError as e:
            raise MCPClientError(
                f"MCP tool {tool_name} returned non-JSON: {result.content[0].text[:200]}"
            ) from e
        if not isinstance(data, dict):
            raise MCPClientError(
                f"MCP tool {tool_name} returned non-dict JSON: {type(data)}"
            )
        return data

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._ready:
            self._ensure_started()
        assert self._session is not None
        return await self._session.call_tool(tool_name, arguments=arguments)

    def close(self) -> None:
        if not self._ready and not self._closed:
            return
        try:
            if self._session and self._loop:
                self._loop.run_until_complete(self._session.__aexit__(None, None, None))
            if self._stdio_cm and self._loop:
                self._loop.run_until_complete(self._stdio_cm.__aexit__(None, None, None))
            if self._loop:
                self._loop.close()
        except Exception as e:
            logger.warning("MCP client close error: %s", e)
        self._ready = False
        self._closed = True
        self._loop = None
        self._session = None
        self._stdio_cm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# ============================================================
# HTTP pool client (新模式, 支持并发)
# ============================================================
class _HTTPWorkerClient:
    """单个 HTTP worker 的 MCP client (长期持有 streamable-http 连接)."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/mcp"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._cm = None  # streamablehttp_client context manager
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
            logger.info("HTTP worker connected: %s", self.url)

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_started()
        assert self._session is not None and self._loop is not None
        try:
            result = self._loop.run_until_complete(
                self._session.call_tool(tool_name, arguments=arguments)
            )
        except Exception as e:
            raise MCPClientError(f"HTTP worker {self.port} call {tool_name} 失败: {e}") from e

        if getattr(result, "isError", False):
            err_text = result.content[0].text if result.content else "未知错误"
            raise MCPClientError(f"HTTP worker {self.port} 工具 {tool_name} 报错: {err_text}")

        if not result.content:
            raise MCPClientError(f"HTTP worker {self.port} 工具 {tool_name} 返回空内容")
        try:
            data = json.loads(result.content[0].text)
        except json.JSONDecodeError as e:
            raise MCPClientError(
                f"HTTP worker {self.port} tool {tool_name} non-JSON: {result.content[0].text[:200]}"
            ) from e
        if not isinstance(data, dict):
            raise MCPClientError(
                f"HTTP worker {self.port} tool {tool_name} non-dict: {type(data)}"
            )
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
            logger.warning("HTTP worker %d close error: %s", self.port, e)
        self._ready = False
        self._loop = None
        self._session = None
        self._cm = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class HTTPMCPClientPool:
    """N 个 HTTP worker 的连接池 + 轮询分配."""

    def __init__(self, pool_size: int, host: str, base_port: int) -> None:
        self.pool_size = pool_size
        self.host = host
        self.base_port = base_port
        self._workers: list[_HTTPWorkerClient] = [
            _HTTPWorkerClient(host, base_port + i) for i in range(pool_size)
        ]
        self._rr = 0  # round-robin index
        self._rr_lock = threading.Lock()

    def _next_worker(self) -> _HTTPWorkerClient:
        with self._rr_lock:
            w = self._workers[self._rr % self.pool_size]
            self._rr += 1
        return w

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """轮询分配到下一个 worker. 单个 worker 失败不重试 (上层 fallback 兜底)."""
        w = self._next_worker()
        return w.call_tool_sync(tool_name, arguments)

    def close(self) -> None:
        for w in self._workers:
            w.close()


# ============================================================
# 单例工厂 (根据 MCP_TRANSPORT 决定)
# ============================================================
_stdio_singleton: ResumeMCPClient | None = None
_http_pool: HTTPMCPClientPool | None = None
_factory_lock = threading.Lock()


def get_mcp_client() -> Any:
    """根据 MCP_TRANSPORT 环境变量返回单例 client (stdio 或 http pool).

    MCP_TRANSPORT=stdio (默认): 返回 ResumeMCPClient 单例
    MCP_TRANSPORT=http:        返回 HTTPMCPClientPool (由 supervisor 管理子进程)
    """
    global _stdio_singleton, _http_pool
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    with _factory_lock:
        if transport == "http":
            if _http_pool is None:
                from mcp_servers.supervisor import get_supervisor
                sup = get_supervisor()  # 懒启动 supervisor
                _http_pool = HTTPMCPClientPool(
                    pool_size=sup.pool_size,
                    host=sup.host,
                    base_port=sup.base_port,
                )
                logger.info(
                    "MCP HTTP pool created: %d workers on %s:%d-%d",
                    sup.pool_size, sup.host, sup.base_port, sup.base_port + sup.pool_size - 1,
                )
            return _http_pool
        # default: stdio
        if _stdio_singleton is None:
            _stdio_singleton = ResumeMCPClient()
        return _stdio_singleton


# 向后兼容: 旧的 get_resume_mcp_client() 仍可用, 等价于 stdio 模式
def get_resume_mcp_client() -> ResumeMCPClient:
    """获取 stdio MCP 客户端单例 (向后兼容)."""
    client = get_mcp_client()
    if isinstance(client, ResumeMCPClient):
        return client
    raise MCPClientError(
        f"当前 MCP_TRANSPORT={os.getenv('MCP_TRANSPORT', 'stdio')} 不是 stdio, "
        "请用 get_mcp_client() 拿连接池"
    )


# ============================================================
# 便捷函数 (MCP 调用降级到本地直接调用)
# ============================================================
def _parse_tool_result(content: list) -> dict[str, Any]:
    """从 MCP CallToolResult.content 解析出 dict (复用逻辑)."""
    if not content:
        raise MCPClientError("MCP 返回空内容")
    try:
        data = json.loads(content[0].text)
    except json.JSONDecodeError as e:
        raise MCPClientError(f"MCP returned non-JSON: {content[0].text[:200]}") from e
    if not isinstance(data, dict):
        raise MCPClientError(f"MCP returned non-dict: {type(data)}")
    return data


def parse_resume_via_mcp_or_local(
    file_path: str, llm_provider: str = "qwen"
) -> dict[str, Any]:
    """优先通过 MCP 调 parse_resume, 失败时降级到本地直接调用."""
    try:
        client = get_mcp_client()
        return client.call_tool_sync("parse_resume", {
            "file_path": file_path,
            "llm_provider": llm_provider,
        })
    except Exception as e:
        logger.warning("MCP parse_resume failed (%s), falling back to local", e)
        from agents.resume_parser import parse_resume_file
        return parse_resume_file(file_path, provider=llm_provider, use_mcp=False).model_dump(exclude_none=True)


def parse_jd_via_mcp_or_local(
    text: str, llm_provider: str = "qwen"
) -> dict[str, Any]:
    """优先通过 MCP 调 parse_jd, 失败时降级到本地."""
    try:
        client = get_mcp_client()
        return client.call_tool_sync("parse_jd", {
            "text": text,
            "llm_provider": llm_provider,
        })
    except Exception as e:
        logger.warning("MCP parse_jd failed (%s), falling back to local", e)
        from agents.jd_parser import parse_jd_text
        return parse_jd_text(text, provider=llm_provider, use_mcp=False).model_dump(exclude_none=True)