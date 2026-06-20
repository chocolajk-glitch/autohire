"""简历解析 Agent.

输入: 简历 PDF / DOCX / TXT 文本
输出: ParsedResume 结构化数据

注意: 早期版本通过 MCP 子进程调用, 现已重构为直接函数调用.
MCP 在这个场景下属于过度设计 (PyMuPDF 故障隔离价值小, 单进程调用无并发收益).
"""
from __future__ import annotations

import logging
from pathlib import Path

from agents._prompts import RESUME_SYSTEM_PROMPT
from core.llm_factory import get_llm
from core.schemas import ParsedResume
from core.structured_output import structured_call

logger = logging.getLogger(__name__)


def parse_resume_text(text: str, provider: str = "deepseek") -> ParsedResume:
    """把简历纯文本解析为 ParsedResume."""
    if not text or len(text.strip()) < 50:
        raise ValueError("简历文本太短, 至少需要 50 个字符")

    client = get_llm(provider)
    return structured_call(
        client,
        system=RESUME_SYSTEM_PROMPT,
        user=f"请解析以下简历:\n\n{text}",
        output_model=ParsedResume,
    )


def parse_resume_file(path: str | Path, provider: str = "deepseek") -> ParsedResume:
    """从文件 (PDF/DOCX/TXT) 读简历文本再解析."""
    from core.tools.document_parser import parse_any

    text = parse_any(path)
    return parse_resume_text(text, provider=provider)