"""JD (Job Description) 解析 Agent.

输入: JD 纯文本
输出: ParsedJD 结构化数据

SYSTEM_PROMPT 在 agents/_prompts.py 统一管理 (MCP 和 fallback 路径共用同一份).
"""
from __future__ import annotations

import logging
from pathlib import Path

from agents._prompts import JD_SYSTEM_PROMPT
from core.llm_factory import get_llm
from core.schemas import ParsedJD
from core.structured_output import structured_call

logger = logging.getLogger(__name__)


def parse_jd_text(text: str, provider: str = "deepseek", *, use_mcp: bool = True) -> ParsedJD:
    """把 JD 纯文本解析为 ParsedJD.

    Args:
        text: JD 文本
        provider: LLM provider (qwen / deepseek / minimax)
        use_mcp: True 优先走 MCP 简历解析服务 (独立进程), 失败回退本地调用
    """
    if not text or len(text.strip()) < 20:
        raise ValueError("JD text too short, need at least 20 chars")

    # 优先通过 MCP 调用 (跨进程, 解耦, 面试亮点)
    if use_mcp:
        try:
            from core.mcp_client import parse_jd_via_mcp_or_local
            data = parse_jd_via_mcp_or_local(text, llm_provider=provider)
            return ParsedJD.model_validate(data)
        except Exception as e:
            logger.warning("MCP parse_jd failed (%s), falling back to local", e)

    # 本地直接调用 (fallback)
    client = get_llm(provider)
    return structured_call(
        client,
        system=JD_SYSTEM_PROMPT,
        user=f"请解析以下 JD:\n\n{text}",
        output_model=ParsedJD,
    )


def parse_jd_file(path: str | Path, provider: str = "deepseek", *, use_mcp: bool = True) -> ParsedJD:
    """从文件 (PDF/DOCX/TXT) 读 JD 文本再解析."""
    from core.tools.document_parser import parse_any

    text = parse_any(path)
    return parse_jd_text(text, provider=provider, use_mcp=use_mcp)
