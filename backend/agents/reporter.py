"""报告生成 Agent.

把 MatchResult + InterviewQuestionSet 合并成 CandidateReport,
并基于规则判断是否需要 HR 人工复核 (HITL).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from core.llm_factory import get_llm
from core.schemas import (
    CandidateReport,
    InterviewQuestionSet,
    MatchResult,
    ParsedJD,
    ParsedResume,
)
from core.structured_output import structured_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个资深的招聘报告撰写专家.

你的任务是基于"匹配度评估结果"和"面试题"生成一份候选人的最终评估报告.

【输出 JSON 顶层字段】
- candidate_name: 字符串
- job_title: 字符串 (从 JD 拿)
- match: 对象, 完整复制入参的 MatchResult
- interview_questions: 对象或 null, 完整复制入参的 InterviewQuestionSet
- recommendation: 必须是以下之一:
    "strong_recommend" / "recommend" / "neutral" / "not_recommend"
- recommendation_reason: 一段话 (10-200 字), 解释为什么给这个推荐
- needs_human_review: 布尔
- human_review_reason: 字符串或 null (如果 needs_human_review=true, 必须填)

【recommendation 判断标准】
- strong_recommend: overall_score >= 85, must_have 全满足
- recommend: overall_score >= 70, must_have 满足率 >= 80%
- neutral: overall_score >= 50, 但有 1-2 个 must_have 弱
- not_recommend: overall_score < 50, 或多个 must_have 不满足

【needs_human_review 触发条件 (任一即 true)】
- overall_score 在 60-75 之间 (边界分数, 难以自动判断)
- confidence 为 "low"
- must_have 中有 1-2 个 evidence 写 "无" 或很短
- recommendation 为 "neutral"
"""


@dataclass
class HITLConfig:
    boundary_low: int = 60
    boundary_high: int = 75


def _build_prompt(
    jd: ParsedJD,
    resume: ParsedResume,
    match: MatchResult,
    questions: InterviewQuestionSet | None,
) -> str:
    parts = [
        f"【JD 岗位】{jd.job_title}",
        f"【候选人】{resume.candidate_name}, {resume.years_of_experience} 年经验",
        f"【匹配度结果】\n{json.dumps(match.model_dump(exclude_none=True), ensure_ascii=False, indent=2)}",
    ]
    if questions is not None:
        parts.append(f"【面试题】\n{json.dumps(questions.model_dump(exclude_none=True), ensure_ascii=False, indent=2)}")
    else:
        parts.append("【面试题】无")
    parts.append("请基于以上信息生成最终报告.")
    return "\n\n".join(parts)


def _rule_based_hitl_check(
    match: MatchResult, jd: ParsedJD, hitl_cfg: HITLConfig
) -> tuple[bool, str | None]:
    """基于规则的 HITL 触发检查 (跟 LLM 并行, 任一触发即 true)."""
    reasons: list[str] = []
    s = match.overall_score
    if hitl_cfg.boundary_low <= s <= hitl_cfg.boundary_high:
        reasons.append(f"分数 {s} 处于边界区间 [{hitl_cfg.boundary_low}, {hitl_cfg.boundary_high}]")
    if match.confidence == "low":
        reasons.append("LLM 评估置信度为 low")
    must_have_reqs = [r for r in jd.requirements if r.is_must_have]
    if must_have_reqs:
        missing_evidence = 0
        for req in must_have_reqs:
            for dim in match.dimensions:
                if dim.requirement and (req.description in dim.requirement or dim.requirement in req.description):
                    if not dim.candidate_evidence or dim.candidate_evidence.strip() in ("无", "none", "N/A"):
                        missing_evidence += 1
        if missing_evidence > 0:
            reasons.append(f"{missing_evidence} 条 must_have 要求证据不足")
    if reasons:
        return True, "; ".join(reasons)
    return False, None


def generate_candidate_report(
    jd: ParsedJD,
    resume: ParsedResume,
    match: MatchResult,
    questions: InterviewQuestionSet | None = None,
    *,
    provider: str = "deepseek",
    hitl_cfg: HITLConfig | None = None,
) -> CandidateReport:
    """生成最终报告.

    流程:
    1. 调 LLM 生成 CandidateReport
    2. 用规则补充/覆盖 needs_human_review (避免 LLM 漏判)
    """
    hitl_cfg = hitl_cfg or HITLConfig()
    client = get_llm(provider)
    report = structured_call(
        client,
        system=SYSTEM_PROMPT,
        user=_build_prompt(jd, resume, match, questions),
        output_model=CandidateReport,
    )
    # 规则补充: 如果 LLM 没说需要 HITL, 但规则认为需要, 强制开启
    rule_needs, rule_reason = _rule_based_hitl_check(match, jd, hitl_cfg)
    if rule_needs and not report.needs_human_review:
        logger.info("rule-based HITL override for %s: %s", resume.candidate_name, rule_reason)
        report = report.model_copy(update={
            "needs_human_review": True,
            "human_review_reason": rule_reason,
        })
    elif rule_needs and report.human_review_reason:
        # 合并理由
        report = report.model_copy(update={
            "human_review_reason": f"{report.human_review_reason} | 规则补充: {rule_reason}",
        })
    return report
