"""匹配度评估 Agent.

核心能力:
1. 逐维评分 - 把 JD 每条要求和简历做证据对比
2. Self-Reflection 反思 - 初评后让 LLM 自己审视是否漏判/误判
3. 加权汇总 - 产出 0-100 总分

设计:
- 初评用 DeepSeek (中文长上下文强)
- 反思用 Qwen (便宜, 中文反思同样强)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from core.llm_factory import ChatClient, get_llm
from core.schemas import JDRequirement, MatchDimension, MatchResult, ParsedJD, ParsedResume
from core.structured_output import structured_call

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_INITIAL = """你是一个严格的招聘匹配度评估专家.

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

【输出 JSON 顶层字段】
job_title (string), requirements (array of {category, description, weight, is_must_have})
candidate_name, candidate_evidence, score (0-10), reasoning

【输出格式严格遵守】
- 顶层: overall_score, dimensions (数组), strengths, weaknesses, confidence
- dimensions 元素字段: requirement (string), candidate_evidence, score, reasoning
- 不要再包含 job_title / requirements 字段 (那属于 JD, 不属于本任务)
"""


SYSTEM_PROMPT_REFLECT = """你是一个严格的质量审查员, 负责检查刚才的简历匹配度评估.

你的任务是审视下面这份"初评结果", 找出可能的:
- 漏判: 简历里有明确证据但被打了低分 (如 "FastAPI" 在项目里出现过但被忽略)
- 误判: 把等价技能判成了"无" (如 简历写 "Flask", JD 写 "Python Web", 不算完全无)
- 加权偏差: must_have 维度应该重点关注, 但分给低了
- 漏掉的项目经验: 简历 projects 里有相关技术, 没被引用到 evidence

【严格输出】只输出一个 JSON 对象:
{
  "issues": ["issue 1", "issue 2", ...],   // 列出所有发现的问题, 没发现则空数组
  "overall_score_adjust": <integer, -10 到 +10>,  // 建议总分调整幅度
  "reflection_note": "<string, 一句话总结反思过程>"
}
不要任何其他文字, 不要 markdown, 只要 JSON.
"""


def _build_initial_user_prompt(
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


def _build_reflection_user_prompt(
    jd: ParsedJD,
    resume: ParsedResume,
    initial: MatchResult,
) -> str:
    jd_text = "\n".join(f"- [{r.category}] {r.description} (weight={r.weight}, must={r.is_must_have})" for r in jd.requirements)
    resume_skills = ", ".join(resume.skills) if resume.skills else "(无)"
    resume_projects = "\n".join(f"  * {p.name}: {', '.join(p.tech_stack or [])}" for p in resume.projects) or "(无)"
    dims_text = "\n".join(
        f"  - req: {d.requirement} | score: {d.score} | evidence: {d.candidate_evidence}"
        for d in initial.dimensions
    )
    return (
        "请审查下面的初评结果, 找出漏判/误判.\n\n"
        f"【JD 要求】\n{jd_text}\n\n"
        f"【简历技能】\n{resume_skills}\n\n"
        f"【简历项目】\n{resume_projects}\n\n"
        f"【初评结果】\noverall_score: {initial.overall_score}\n"
        f"dimensions:\n{dims_text}\n"
        f"strengths: {initial.strengths}\n"
        f"weaknesses: {initial.weaknesses}\n"
    )


def _adjust_with_reflection(initial: MatchResult, reflection: dict) -> MatchResult:
    """把反思结果合并进 final MatchResult."""
    adjust = int(reflection.get("overall_score_adjust", 0))
    new_overall = max(0, min(100, initial.overall_score + adjust))
    note = reflection.get("reflection_note", "")
    if reflection.get("issues"):
        note = (note + " | 发现: " + "; ".join(reflection["issues"])).strip(" |")

    # 如果反思给到了 issues 但分数没调整, 至少把 confidence 降一档 (表示评估有争议)
    confidence = initial.confidence
    if reflection.get("issues") and adjust == 0:
        if confidence == "high":
            confidence = "medium"
        elif confidence == "medium":
            confidence = "low"

    return MatchResult(
        overall_score=new_overall,
        dimensions=initial.dimensions,
        strengths=initial.strengths,
        weaknesses=initial.weaknesses,
        reflection_note=note or None,
        confidence=confidence,
    )


@dataclass
class MatcherConfig:
    initial_provider: str = "deepseek"
    reflect_provider: str = "qwen"
    enable_reflection: bool = True
    enable_web_search: bool = True  # 联网搜索增强


def match_resume_to_jd(
    jd: ParsedJD,
    resume: ParsedResume,
    config: MatcherConfig | None = None,
) -> MatchResult:
    """评估一份简历对一份 JD 的匹配度.

    流程:
    1. (可选) 联网搜索补充公司+岗位信息
    2. 初评 (initial_provider)
    3. 反思 (reflect_provider) - 可关闭
    4. 合并输出
    """
    from agents.web_searcher import search_company_info

    cfg = config or MatcherConfig()
    web_context = ""
    if cfg.enable_web_search:
        web_context = search_company_info(jd.company, jd.job_title)
        if web_context:
            logger.info("web search context added (%d chars)", len(web_context))

    initial_client = get_llm(cfg.initial_provider)
    initial = structured_call(
        initial_client,
        system=SYSTEM_PROMPT_INITIAL,
        user=_build_initial_user_prompt(jd, resume, web_context=web_context),
        output_model=MatchResult,
    )
    logger.info(
        "initial match: candidate=%s overall=%d confidence=%s",
        resume.candidate_name, initial.overall_score, initial.confidence,
    )

    if not cfg.enable_reflection:
        return initial

    try:
        reflect_client = get_llm(cfg.reflect_provider)
        reflect_user = _build_reflection_user_prompt(jd, resume, initial)
        reflect_raw = reflect_client.chat(SYSTEM_PROMPT_REFLECT, reflect_user)
        # 反思结果用宽松解析 (不一定要严格 schema, 只要 issues/overall_score_adjust 几个字段)
        from core.structured_output import _extract_json
        reflection = _extract_json(reflect_raw)
        # 容错: 缺字段则补默认
        if not isinstance(reflection, dict):
            raise ValueError("reflection not a dict")
        reflection.setdefault("issues", [])
        reflection.setdefault("overall_score_adjust", 0)
        reflection.setdefault("reflection_note", "")
        final = _adjust_with_reflection(initial, reflection)
        logger.info(
            "after reflection: adjust=%+d final=%d note=%s",
            reflection["overall_score_adjust"], final.overall_score, final.reflection_note,
        )
        return final
    except Exception as e:
        logger.warning("reflection failed, returning initial: %s", str(e)[:200])
        # 反思失败不影响主流程, 返回 initial
        return initial
