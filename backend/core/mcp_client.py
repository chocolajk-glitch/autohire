"""MCP 客户端 wrapper - 在主 AutoHire 进程内启动并连接 MCP 服务.

用法:
    from core.mcp_client import get_resume_mcp_client

    client = get_resume_mcp_client()
    result = await client.call_tool("parse_jd", {"text": "..."})
    # result 是 dict (MCP 工具返回的 JSON)

特点:
- 懒启动: 第一次调用才启动 MCP 子进程
- 自动管理生命周期: 程序退出时关闭
- 提供 sync 接口 (在事件循环外也能用)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
MCP_SERVER_SCRIPT = BACKEND_ROOT / "mcp_servers" / "resume_server.py"
PYTHON_EXE = Path(sys.executable)  # 用当前 venv 的 python


class MCPClientError(RuntimeError):
    pass


class ResumeMCPClient:
    """懒启动 + 同步/异步双接口的 MCP 客户端."""

    def __init__(self) -> None:
        self._process: subprocess.Popen | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._ready = False
        self._closed = False  # 显式调用 close() 后置 True, 防止复用

    def _ensure_started(self) -> None:
        if self._closed:
            raise MCPClientError("MCP client has been closed; create a new one")
        if self._ready:
            return
        if not MCP_SERVER_SCRIPT.exists():
            raise MCPClientError(f"MCP server script not found: {MCP_SERVER_SCRIPT}")

        # 用 stdio_client 启动子进程 + 通信
        from mcp.client.stdio import stdio_client, StdioServerParameters
        from mcp.client.session import ClientSession

        params = StdioServerParameters(
            command=str(PYTHON_EXE),
            args=["-m", "mcp_servers.resume_server"],
            cwd=str(BACKEND_ROOT),
        )
        # 进入一个持久的事件循环
        self._loop = asyncio.new_event_loop()
        # 在新 loop 里启动 stdio_client
        cm = stdio_client(params)
        read, write = self._loop.run_until_complete(cm.__aenter__())
        self._stdio_cm = cm
        self._read = read
        self._write = write
        # 启动 ClientSession
        self._session = ClientSession(read, write)
        self._loop.run_until_complete(self._session.__aenter__())
        self._loop.run_until_complete(self._session.initialize())
        self._ready = True
        logger.info("MCP client connected to resume server (pid=%s)", self._process.pid if self._process else "n/a")

    def call_tool_sync(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """同步调用 MCP 工具, 返回 dict.

        MCP 服务端返回的 CallToolResult 有两个字段:
        - structuredContent: 启用了 output_schema 时有值, 当前我们没启用, 永远 None
        - content[0].text: 服务端返回的 JSON 字符串 (我们这个服务的格式)

        所以我们解析 content[0].text.
        """
        self._ensure_started()
        assert self._session is not None
        assert self._loop is not None
        try:
            result = self._loop.run_until_complete(
                self._session.call_tool(tool_name, arguments=arguments)
            )
        except Exception as e:
            raise MCPClientError(f"MCP tool {tool_name} failed: {e}") from e

        if getattr(result, "isError", False):
            # 服务端显式报错 (比如 raise ValueError)
            err_text = result.content[0].text if result.content else "unknown error"
            raise MCPClientError(f"MCP tool {tool_name} error: {err_text}")

        # 解析 content[0].text 为 JSON
        if not result.content:
            raise MCPClientError(f"MCP tool {tool_name} returned empty content")
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
        """异步调用 MCP 工具 (优先, 在已有事件循环时用)."""
        if not self._ready:
            # 懒启动, 但 startup 内部用了同步 loop, 这里跑就行
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


# ========== 单例 (懒加载) ==========
_singleton: ResumeMCPClient | None = None


def get_resume_mcp_client() -> ResumeMCPClient:
    """获取全局 MCP 客户端单例. 第一次调用启动 MCP 子进程."""
    global _singleton
    if _singleton is None:
        _singleton = ResumeMCPClient()
    return _singleton


# ========== 便捷函数 (MCP 调用降级到本地直接调用) ==========
def parse_resume_via_mcp_or_local(
    file_path: str, llm_provider: str = "qwen"
) -> dict[str, Any]:
    """优先通过 MCP 调 parse_resume, 失败时降级到本地直接调用.

    降级时强制 use_mcp=False 防止递归.
    """
    try:
        client = get_resume_mcp_client()
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
    """优先通过 MCP 调 parse_jd, 失败时降级到本地.

    降级时强制 use_mcp=False 防止递归 (parse_jd_text 默认 use_mcp=True).
    """
    try:
        client = get_resume_mcp_client()
        return client.call_tool_sync("parse_jd", {
            "text": text,
            "llm_provider": llm_provider,
        })
    except Exception as e:
        logger.warning("MCP parse_jd failed (%s), falling back to local", e)
        from agents.jd_parser import parse_jd_text
        # 关键: use_mcp=False 防止无限递归
        return parse_jd_text(text, provider=llm_provider, use_mcp=False).model_dump(exclude_none=True)
