"""简历解析 Agent.

输入: 简历 PDF / DOCX / TXT 文本
输出: ParsedResume 结构化数据

SYSTEM_PROMPT 在 agents/_prompts.py 统一管理 (MCP 和 fallback 路径共用同一份).
"""
from __future__ import annotations

import logging
from pathlib import Path

from agents._prompts import RESUME_SYSTEM_PROMPT
from core.llm_factory import get_llm
from core.schemas import ParsedResume
from core.structured_output import structured_call

logger = logging.getLogger(__name__)


def parse_resume_text(text: str, provider: str = "deepseek", *, use_mcp: bool = True) -> ParsedResume:
    """把简历纯文本解析为 ParsedResume.

    Args:
        text: 简历文本
        provider: LLM provider
        use_mcp: True 优先走 MCP, 失败回退本地
    """
    if not text or len(text.strip()) < 50:
        raise ValueError("resume text too short, need at least 50 chars")

    # 简历纯文本没有文件路径, MCP 服务的 parse_resume 工具需要 file_path
    # 所以这里只走本地; 文件入口 parse_resume_file 会用 MCP
    client = get_llm(provider)
    return structured_call(
        client,
        system=RESUME_SYSTEM_PROMPT,
        user=f"请解析以下简历:\n\n{text}",
        output_model=ParsedResume,
    )


def parse_resume_file(path: str | Path, provider: str = "deepseek", *, use_mcp: bool = True) -> ParsedResume:
    """从文件 (PDF/DOCX/TXT) 读简历文本再解析.

    优先通过 MCP 服务解析 (跨进程, 解耦), 失败回退本地.
    """
    if use_mcp:
        try:
            from core.mcp_client import parse_resume_via_mcp_or_local
            data = parse_resume_via_mcp_or_local(str(path), llm_provider=provider)
            return ParsedResume.model_validate(data)
        except Exception as e:
            logger.warning("MCP parse_resume failed (%s), falling back to local", e)

    # 回退到本地: 读文件 + 本地解析
    from core.tools.document_parser import parse_any
    text = parse_any(path)
    return parse_resume_text(text, provider=provider, use_mcp=False)
