"""匹配度评估 Agent.

核心能力:
1. 逐维评分 - 把 JD 每条要求和简历做证据对比
2. 加权汇总 - 产出 0-100 总分

设计:
- 默认实现: LLM 单次评估 (无反思)
- AutoGen 模式: 用 SelectorGroupChat 让 Assessor + Refiner 双 Agent 协作
  (在 auto_gen_orchestrator._run_matcher_team 里, 不在本文件)
- 联网搜索增强: 可选, 通过 Tavily 查公司背景
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from core.llm_factory import ChatClient, get_llm
from core.schemas import MatchResult, ParsedJD, ParsedResume
from core.structured_output import structured_call

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """你是一个严格的招聘匹配度评估专家.

你的任务是基于一份"已解析的 JD"和一份"已解析的简历", 评估候选人匹配度.

【工作步骤】
1. 对 JD 的每一条 requirement, 在简历中找证据
2. 给出该维度的 score (0-10):
   - 0-2: 完全不满足或简历中找不到相关证据
   - 3-5: 部分满足, 有相关经验但深度/广度不足
   - 6-8: 较好满足, 有明确证据
   - 9-10: 完全满足且超出预期
3. candidate_evidence 必须从简历中引用具体信息 (项目/工作经历/技能), 找不到则写"无"
4. 给出 reasoning (一句话解释为何这个分)
5. 汇总 strengths (3-5 条, 候选人最强的地方) 和 weaknesses (2-3 条, 不足的地方)
6. overall_score 0-100 = 用 requirement 的 weight 加权平均各维度, 再 ×10
7. confidence: 你对评分的把握程度 (low/medium/high)

【输出格式严格遵守】
- 顶层: overall_score, dimensions (数组), strengths, weaknesses, confidence
- dimensions 元素字段: requirement (string), candidate_evidence, score, reasoning
"""


def _build_user_prompt(
    jd: ParsedJD,
    resume: ParsedResume,
    web_context: str = "",
) -> str:
    jd_dict = jd.model_dump(exclude_none=True)
    resume_dict = resume.model_dump(exclude_none=True)
    web_section = f"\n\n【联网搜索补充信息】\n{web_context}" if web_context else ""
    return (
        "请评估以下候选人 vs 职位的匹配度.\n\n"
        f"【JD】\n{json.dumps(jd_dict, ensure_ascii=False, indent=2)}\n\n"
        f"【简历】\n{json.dumps(resume_dict, ensure_ascii=False, indent=2)}"
        f"{web_section}"
    )


@dataclass
class MatcherConfig:
    provider: str = "deepseek"
    enable_web_search: bool = True  # 联网搜索增强


def match_resume_to_jd(
    jd: ParsedJD,
    resume: ParsedResume,
    config: MatcherConfig | None = None,
) -> MatchResult:
    """评估一份简历对一份 JD 的匹配度.

    默认实现: LLM 单次评估 (无反思).
    如果要"反思"功能, 用 use_autogen=True 走 AutoGen SelectorGroupChat.

    流程:
    1. (可选) 联网搜索补充公司+岗位信息
    2. LLM 单次评估
    """
    from agents.web_searcher import search_company_info

    cfg = config or MatcherConfig()
    web_context = ""
    if cfg.enable_web_search:
        web_context = search_company_info(jd.company, jd.job_title)
        if web_context:
            logger.info("web search context added (%d chars)", len(web_context))

    client = get_llm(cfg.provider)
    result = structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=_build_user_prompt(jd, resume, web_context=web_context),
        output_model=MatchResult,
    )
    logger.info(
        "match: candidate=%s overall=%d confidence=%s",
        resume.candidate_name, result.overall_score, result.confidence,
    )
    return result
