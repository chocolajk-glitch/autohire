"""JD (Job Description) 解析 Agent.

输入: JD 纯文本
输出: ParsedJD 结构化数据
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from core.llm_factory import get_llm
from core.schemas import ParsedJD
from core.structured_output import structured_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个资深的招聘需求分析专家.
你的任务是把一段招聘 JD (Job Description) 文本解析成结构化字段.

【输出格式 - 必须严格遵守】
- 输出一个 JSON 对象, 顶层字段 (按顺序):
  - job_title: 字符串, 岗位名
  - company: 字符串或 null
  - salary_range: 字符串或 null (如 "30K-50K")
  - location: 字符串或 null
  - experience_years_min: 整数或 null (最低年限)
  - experience_years_max: 整数或 null (最高年限, null 表示"以上")
  - requirements: 数组, 每条是一个对象 {category, description, weight, is_must_have}
  - summary: 50-200 字的总结

【requirements 数组 - 扁平结构, 不要嵌套】
每条 requirement 对象的字段:
  - category: 必须是以下之一: "required_skill" / "nice_to_have" / "experience" / "education" / "responsibility" / "other"
  - description: 字符串, 描述该要求
  - weight: 整数 1-10, 越重要越高
  - is_must_have: 布尔, 含"必须""必备""至少"等关键词则 true

【常见错误避免】
- 不要把 requirements 按 category 拆成多个数组 (如 {required_skill: [...], nice_to_have: [...]})
  这是错的! 必须是扁平数组, 每条带 category 字段.
- 不要在顶层省略 job_title / requirements / summary
- 不要使用 "content" 字段, 用 "description"
"""


def parse_jd_text(text: str, provider: str = "deepseek") -> ParsedJD:
    """把 JD 纯文本解析为 ParsedJD.

    Args:
        text: JD 文本
        provider: LLM provider (qwen / minimax / deepseek)
    """
    if not text or len(text.strip()) < 20:
        raise ValueError("JD text too short, need at least 20 chars")
    client = get_llm(provider)
    return structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=f"请解析以下 JD:\n\n{text}",
        output_model=ParsedJD,
    )


def parse_jd_file(path: str | Path, provider: str = "deepseek") -> ParsedJD:
    """从文件 (PDF/DOCX/TXT) 读 JD 文本再解析."""
    from core.tools.document_parser import parse_any

    text = parse_any(path)
    return parse_jd_text(text, provider=provider)
