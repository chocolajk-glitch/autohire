"""AutoHire MCP Server - 简历解析独立服务

作为独立进程运行, 通过 stdio 与主 AutoHire 后端通信.
暴露两个工具:
- parse_resume: 简历文件路径 -> ParsedResume JSON
- parse_jd:     JD 文本 -> ParsedJD JSON

启动方式: python -m mcp_servers.resume_server
被主进程启动: 详见 core/mcp_client.py 和 mcp_config.json
"""
from __future__ import annotations

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
        raise ValueError(f"resume text too short or empty: {file_path}")
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
        raise ValueError("JD text too short")
    client = get_llm(llm_provider)
    result: ParsedJD = structured_call(
        client, system=JD_SYSTEM_PROMPT,
        user=f"请解析以下 JD:\n\n{text}", output_model=ParsedJD,
    )
    return result.model_dump(exclude_none=True)


if __name__ == "__main__":
    logger.info("starting autohire-resume-parser MCP server (stdio)")
    mcp.run(transport="stdio")
