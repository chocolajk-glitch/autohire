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

【输出格式 - 必须严格遵守 JSON】
- 只输出一个 JSON 对象, 不要任何其他文字.

【完整 JSON 示例 (格式必须照此)】
```json
{
  "candidate_name": "张三",
  "email": "zhangsan@example.com",
  "phone": "138-0000-0000",
  "years_of_experience": 3,
  "educations": [
    {"school": "清华大学", "degree": "bachelor", "major": "计算机科学", "start_year": 2019, "end_year": 2023}
  ],
  "work_experiences": [
    {"company": "字节跳动", "title": "后端工程师", "start_date": "2023-07", "end_date": null, "description": "..."}
  ],
  "projects": [
    {"name": "AutoHire", "role": "主程", "description": "...", "tech_stack": ["Python", "FastAPI"], "duration_months": 6}
  ],
  "skills": ["Python", "LangGraph", "PostgreSQL"],
  "self_summary": "3 年后端经验, 专注 AI 基础设施."
}
```

【字段类型严格规定】
- candidate_name / email / phone / school / company / title / start_date / end_date / description / role / major / self_summary: **字符串**
- years_of_experience / start_year / end_year / duration_months: **整数** (number, 不是字符串)
- degree: 必须是以下字符串之一: "high_school" / "associate" / "bachelor" / "master" / "phd" / "other"
- educations / work_experiences / projects / skills / tech_stack: **数组**

【常见错误 (会导致重试)】
- educations/work_experiences/projects 写成字符串 (错! 必须数组)
- degree 写成 "本科" (错! 用 "bachelor")
- years_of_experience 写成 "3" 字符串 (错! 整数 3)
- 数据缺失用 null 或空数组, 不要瞎编
"""


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
        system=SYSTEM_PROMPT,
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
            import logging
            logging.getLogger(__name__).warning(
                "MCP parse_resume failed (%s), falling back to local", e
            )

    # 回退到本地: 读文件 + 本地解析
    from core.tools.document_parser import parse_any
    text = parse_any(path)
    return parse_resume_text(text, provider=provider, use_mcp=False)
