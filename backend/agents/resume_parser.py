"""简历解析 Agent.

输入: 简历 PDF / DOCX / TXT 文本
输出: ParsedResume 结构化数据
"""
from __future__ import annotations

import logging
from pathlib import Path

from core.llm_factory import get_llm
from core.schemas import ParsedResume
from core.structured_output import structured_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个资深的简历解析专家.
你的任务是把一份简历 (通常是中英文混合的纯文本) 解析成结构化字段.

【输出格式 - 必须严格遵守】
- 输出一个 JSON 对象, 顶层字段 (按顺序):
  - candidate_name: 字符串, 候选人姓名
  - email: 字符串或 null
  - phone: 字符串或 null
  - years_of_experience: 整数, 总工作年限 (按整数四舍五入)
  - educations: 数组, 元素为 {school, degree, major, start_year, end_year}
  - work_experiences: 数组, 元素为 {company, title, start_date, end_date, description}
  - projects: 数组, 元素为 {name, role, description, tech_stack, duration_months}
  - skills: 字符串数组
  - self_summary: 字符串或 null (自我评价)

【字段细节】
- degree 必须是以下之一: "high_school" / "associate" / "bachelor" / "master" / "phd" / "other"
- start_date / end_date 格式 "YYYY-MM" 或 "YYYY-MM-DD", end_date 缺失或为 "至今" 则填 null
- tech_stack: 写从中识别出的技术名词 (如 ["Python", "FastAPI", "LangGraph"])
- skills: 个人技能标签数组, 不要重复 tech_stack 里的项
- duration_months: 项目持续月数, 不知道则 null

【常见错误避免】
- 不要把 educations / work_experiences / projects 写成字符串
- 不要遗漏 candidate_name 或 years_of_experience
- 如果简历里有数据缺失, 用 null 或空数组, 不要瞎编
"""


def parse_resume_text(text: str, provider: str = "deepseek") -> ParsedResume:
    """把简历纯文本解析为 ParsedResume.

    Args:
        text: 简历文本
        provider: LLM provider
    """
    if not text or len(text.strip()) < 50:
        raise ValueError("resume text too short, need at least 50 chars")
    client = get_llm(provider)
    return structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=f"请解析以下简历:\n\n{text}",
        output_model=ParsedResume,
    )


def parse_resume_file(path: str | Path, provider: str = "deepseek") -> ParsedResume:
    """从文件 (PDF/DOCX/TXT) 读简历文本再解析."""
    from core.tools.document_parser import parse_any

    text = parse_any(path)
    return parse_resume_text(text, provider=provider)
