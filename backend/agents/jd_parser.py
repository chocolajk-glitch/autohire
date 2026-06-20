"""JD (Job Description) 解析 Agent.

输入: JD 纯文本
输出: ParsedJD 结构化数据

注意: 早期版本通过 MCP 子进程调用, 现已重构为直接函数调用.
MCP 在这个场景下属于过度设计 (单进程内调用, 无故障隔离价值).
简历解析同理 (agents/resume_parser.py).
"""
from __future__ import annotations

import logging
from pathlib import Path

from agents._prompts import JD_SYSTEM_PROMPT
from core.llm_factory import get_llm
from core.schemas import ParsedJD
from core.structured_output import structured_call

logger = logging.getLogger(__name__)


def parse_jd_text(text: str, provider: str = "deepseek") -> ParsedJD:
    """把 JD 纯文本解析为 ParsedJD.

    Args:
        text: JD 文本
        provider: LLM provider (qwen / deepseek / minimax)
    """
    if not text or len(text.strip()) < 20:
        raise ValueError("JD 文本太短, 至少需要 20 个字符")

    client = get_llm(provider)
    return structured_call(
        client,
        system=JD_SYSTEM_PROMPT,
        user=f"请解析以下 JD:\n\n{text}",
        output_model=ParsedJD,
    )


def parse_jd_file(path: str | Path, provider: str = "deepseek") -> ParsedJD:
    """从文件 (PDF/DOCX/TXT) 读 JD 文本再解析."""
    from core.tools.document_parser import parse_any

    text = parse_any(path)
    return parse_jd_text(text, provider=provider)