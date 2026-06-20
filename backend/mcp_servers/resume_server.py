"""AutoHire MCP Server - 简历解析独立服务

作为独立进程运行, 通过 stdio / streamable-http 与主 AutoHire 后端通信.
暴露两个工具:
- parse_resume: 简历文件路径 -> ParsedResume JSON
- parse_jd:     JD 文本 -> ParsedJD JSON

启动方式:
  # stdio 模式 (单 client, 适合开发/单进程)
  python -m mcp_servers.resume_server --transport stdio

  # HTTP 模式 (支持连接池并发, 适合生产)
  python -m mcp_servers.resume_server --transport http --port 9001
被主进程启动: 详见 core/mcp_client.py 和 mcp_config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# 让 MCP 服务能 import backend 的 core 模块
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from agents._prompts import JD_SYSTEM_PROMPT, RESUME_SYSTEM_PROMPT  # noqa: E402
from core.llm_factory import get_llm  # noqa: E402
from core.schemas import ParsedJD, ParsedResume  # noqa: E402
from core.structured_output import structured_call  # noqa: E402
from core.tools.document_parser import parse_any  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [mcp-resume] %(message)s")
logger = logging.getLogger(__name__)

mcp = FastMCP("autohire-resume-parser")


@mcp.tool()
def parse_resume(file_path: str, llm_provider: str = "qwen") -> dict[str, Any]:
    """解析一份简历 (PDF/DOCX/TXT/MD) -> ParsedResume.

    Args:
        file_path: 简历文件绝对路径
        llm_provider: LLM provider (qwen / deepseek / minimax)

    Returns:
        ParsedResume 的 JSON 字典 (含 candidate_name, skills, projects 等)
    """
    logger.info("parse_resume called: %s (provider=%s)", file_path, llm_provider)
    text = parse_any(file_path)
    if not text or len(text.strip()) < 50:
        raise ValueError(f"简历文本太短或为空: {file_path}")
    client = get_llm(llm_provider)
    result: ParsedResume = structured_call(
        client, system=RESUME_SYSTEM_PROMPT,
        user=f"请解析以下简历:\n\n{text}", output_model=ParsedResume,
    )
    return result.model_dump(exclude_none=True)


@mcp.tool()
def parse_jd(text: str, llm_provider: str = "qwen") -> dict[str, Any]:
    """解析一段 JD 文本 -> ParsedJD.

    Args:
        text: JD 文本
        llm_provider: LLM provider

    Returns:
        ParsedJD 的 JSON 字典
    """
    logger.info("parse_jd called (provider=%s, len=%d)", llm_provider, len(text))
    if not text or len(text.strip()) < 20:
        raise ValueError("JD 文本太短")
    client = get_llm(llm_provider)
    result: ParsedJD = structured_call(
        client, system=JD_SYSTEM_PROMPT,
        user=f"请解析以下 JD:\n\n{text}", output_model=ParsedJD,
    )
    return result.model_dump(exclude_none=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoHire 简历/JD 解析 MCP 服务")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="通信方式: stdio (默认, 单 client) / http (支持连接池并发)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 模式监听地址")
    parser.add_argument("--port", type=int, default=9001, help="HTTP 模式监听端口")
    args = parser.parse_args()

    if args.transport == "stdio":
        logger.info("starting autohire-resume-parser MCP server (stdio)")
        mcp.run(transport="stdio")
        return

    # HTTP 模式: 改 settings 的 host/port 后启动
    logger.info("starting autohire-resume-parser MCP server (http %s:%d)", args.host, args.port)
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
