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

【输出格式 - 必须严格遵守 JSON】
- 只输出一个 JSON 对象, 不要任何其他文字.

【完整 JSON 示例 (格式必须照此)】
```json
{
  "job_title": "Python 后端工程师",
  "company": "字节跳动",
  "salary_range": "30K-50K",
  "location": "上海",
  "experience_years_min": 3,
  "experience_years_max": null,
  "requirements": [
    {"category": "required_skill", "description": "3 年 Python 后端经验", "weight": 9, "is_must_have": true},
    {"category": "required_skill", "description": "熟悉 FastAPI 或 Django", "weight": 8, "is_must_have": true},
    {"category": "nice_to_have", "description": "有 LangGraph 经验", "weight": 4, "is_must_have": false}
  ],
  "summary": "我们正在招聘一位 Python 后端工程师, 要求 3 年以上经验, 熟悉 FastAPI 和 PostgreSQL."
}
```

【字段类型严格规定】
- job_title / company / salary_range / location / summary: **字符串** (string)
- experience_years_min / experience_years_max: **整数** (number, 不是字符串)
- requirements: **数组**, 每个元素必须包含这 4 个字段:
  - category: 必须是以下字符串之一 (严格匹配, 不要造新词): "required_skill" / "nice_to_have" / "experience" / "education" / "responsibility" / "other"
  - description: **字符串**
  - weight: **整数 1-10**, 必备 8-10, 加分 3-5
  - is_must_have: **布尔 true 或 false**, 不要写 "true" 字符串

【常见错误 (会导致重试)】
- 把 requirements 按 category 分成多个数组 (错!)
- category 写成 "technical_skills" / "programming" 等不在白名单的值 (错! 用 "other")
- weight 写成 "required" 字符串或 0.8 浮点 (错! 整数 1-10)
- is_must_have 写成 "true" 字符串 (错! 布尔)
"""


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
            # MCP 调用失败, 优雅降级到本地
            import logging
            logging.getLogger(__name__).warning(
                "MCP parse_jd failed (%s), falling back to local", e
            )

    # 本地直接调用
    client = get_llm(provider)
    return structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=f"请解析以下 JD:\n\n{text}",
        output_model=ParsedJD,
    )


def parse_jd_file(path: str | Path, provider: str = "deepseek", *, use_mcp: bool = True) -> ParsedJD:
    """从文件 (PDF/DOCX/TXT) 读 JD 文本再解析."""
    from core.tools.document_parser import parse_any

    text = parse_any(path)
    return parse_jd_text(text, provider=provider, use_mcp=use_mcp)
