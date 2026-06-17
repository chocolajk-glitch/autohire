"""AutoGen 主从多智能体编排器.

架构:
- Supervisor Agent (AssistantAgent): 有 4 个 tool, 决定调用顺序
- JD Parser Tool: 调 LLM 解析 JD → ParsedJD
- Resume Parser Tool: 调 LLM 解析简历 → ParsedResume
- Matcher Team (SelectorGroupChat): Assessor + Refiner 协作评估 → MatchResult
- Reporter Tool: 调 LLM 生成报告 → CandidateReport

Matcher 是唯一的 Agent 协作环节 (SelectorGroupChat):
  Assessor 初评 → Refiner 审查修正 → Assessor 回应 → 输出最终结果
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================
# AutoGen Model Client (MiniMax, 无 thinking)
# ============================================================

def _make_model_client():
    """创建 AutoGen 兼容的 model client (MiniMax)."""
    import os
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model="MiniMax-M2.7",
        api_key=os.getenv("MiniMax_API_KEY"),
        base_url=os.getenv("MiniMax_BASE_URL", "https://api.minimaxi.com/v1"),
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
    )


# ============================================================
# JSON 提取工具
# ============================================================

import re

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_BARE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json(text: str) -> dict | list:
    """从 LLM 文本中提取 JSON."""
    # 去掉 <think>...</think> 块 (MiniMax thinking)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = _JSON_FENCE.search(text)
    if m:
        return json.loads(m.group(1))
    m = _JSON_BARE.search(text)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"no JSON found in: {text[:200]}...")


# ============================================================
# Tool 1: JD Parser
# ============================================================

_JD_SYSTEM = """你是一个资深的招聘需求分析专家.
把 JD 文本解析成结构化 JSON.

【输出格式】只输出一个 JSON 对象, 不要任何其他文字.
{
  "job_title": "字符串",
  "company": "字符串或null",
  "salary_range": "字符串或null",
  "location": "字符串或null",
  "experience_years_min": 整数或null,
  "experience_years_max": 整数或null,
  "requirements": [
    {"category": "required_skill|nice_to_have|experience|education|responsibility|other",
     "description": "字符串", "weight": 1-10整数, "is_must_have": true/false}
  ],
  "summary": "10-1000字摘要"
}"""


def _parse_jd_tool(text: str) -> dict:
    """Tool: 解析 JD 文本 → ParsedJD dict."""
    from core.llm_factory import get_llm
    from core.schemas import ParsedJD
    from core.structured_output import structured_call

    client = get_llm("minimax")
    result = structured_call(client, system=_JD_SYSTEM, user=f"请解析以下 JD:\n\n{text}", output_model=ParsedJD)
    return result.model_dump(exclude_none=True)


# ============================================================
# Tool 2: Resume Parser
# ============================================================

_RESUME_SYSTEM = """你是一个资深的简历解析专家.
把简历文本解析成结构化 JSON.

【输出格式】只输出一个 JSON 对象.
{
  "candidate_name": "字符串",
  "email": "字符串或null",
  "phone": "字符串或null",
  "years_of_experience": 整数,
  "educations": [{"school":"字符串","degree":"high_school|associate|bachelor|master|phd|other","major":"字符串或null","start_year":整数,"end_year":整数}],
  "work_experiences": [{"company":"字符串","title":"字符串","start_date":"YYYY-MM","end_date":"YYYY-MM或null","description":"字符串或null"}],
  "projects": [{"name":"字符串","role":"字符串或null","description":"字符串","tech_stack":["字符串"],"duration_months":整数或null}],
  "skills": ["字符串"],
  "self_summary": "字符串或null"
}"""


def _parse_resume_tool(text: str) -> dict:
    """Tool: 解析简历文本 → ParsedResume dict."""
    from core.llm_factory import get_llm
    from core.schemas import ParsedResume
    from core.structured_output import structured_call

    client = get_llm("minimax")
    result = structured_call(client, system=_RESUME_SYSTEM, user=f"请解析以下简历:\n\n{text}", output_model=ParsedResume)
    return result.model_dump(exclude_none=True)


# ============================================================
# Tool 3: Matcher Team (SelectorGroupChat - Agent 协作)
# ============================================================

_ASSESSOR_SYSTEM = """你是一个严格的招聘匹配度评估专家.

你的任务: 基于 JD 和简历, 评估候选人匹配度.

【工作步骤】
1. 对 JD 的每条 requirement, 在简历中找证据
2. 给出该维度的 score (0-10):
   - 0-2: 完全不满足
   - 3-5: 部分满足
   - 6-8: 较好满足
   - 9-10: 完全满足且超出预期
3. candidate_evidence 必须从简历中引用具体信息
4. 给出 overall_score (0-100, 用 weight 加权平均各维度再 ×10)
5. 列出 strengths (3-5条) 和 weaknesses (2-3条)

【输出格式】严格输出以下 JSON, 不要任何其他文字:
```json
{
  "overall_score": 75,
  "dimensions": [
    {"requirement": "要求描述", "candidate_evidence": "从简历找到的证据", "score": 7, "reasoning": "一句话解释"}
  ],
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["不足1", "不足2"],
  "confidence": "low|medium|high"
}
```"""

_REFINER_SYSTEM = """你是一个严格的质量审查员, 负责检查简历匹配度评估.

你会收到评估专家的初评结果. 你的任务:
- 漏判: 简历里有明确证据但被打低分
- 误判: 把等价技能判成"无" (如简历写 Flask, JD 写 Web 框架, 不算完全无)
- 加权偏差: must_have 维度重点不够
- 漏掉的项目经验

【输出格式】严格输出以下 JSON:
```json
{
  "approved": true或false,
  "adjusted_score": 修正后的总分 (0-100, 不改则与原分相同),
  "issues": ["问题1", "问题2"],
  "adjustments": [{"requirement": "某要求", "old_score": 旧分, "new_score": 新分, "reason": "修正原因"}],
  "reflection_note": "一句话总结"
}
```"""


async def _run_matcher_team(jd_dict: dict, resume_dict: dict) -> dict:
    """Tool: SelectorGroupChat — Assessor + Refiner 协作评估.

    流程:
    1. Assessor 初评 → 输出 JSON
    2. Refiner 审查 → 输出 JSON
    3. Assessor 回应 (如有修正) → 输出最终 JSON
    4. 提取最终结果返回
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import SelectorGroupChat

    client = _make_model_client()

    jd_text = json.dumps(jd_dict, ensure_ascii=False, indent=2)
    resume_text = json.dumps(resume_dict, ensure_ascii=False, indent=2)
    task_msg = (
        f"请评估以下候选人 vs 职位的匹配度.\n\n"
        f"【JD】\n{jd_text}\n\n"
        f"【简历】\n{resume_text}"
    )

    assessor = AssistantAgent(
        name="Assessor",
        model_client=client,
        system_message=_ASSESSOR_SYSTEM,
    )

    refiner = AssistantAgent(
        name="Refiner",
        model_client=client,
        system_message=_REFINER_SYSTEM,
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)

    team = SelectorGroupChat(
        [assessor, refiner],
        model_client=client,
        termination_condition=termination,
    )

    logger.info("matcher team: starting SelectorGroupChat")
    t0 = time.time()
    result = await team.run(task=task_msg)
    elapsed = time.time() - t0
    logger.info("matcher team: done in %.1fs, %d messages", elapsed, len(result.messages))

    # 从消息历史中提取最终评估结果
    return _extract_match_from_messages(result.messages)


def _extract_match_from_messages(messages: list) -> dict:
    """从 Agent 消息历史中提取最终的 MatchResult dict.

    策略: 优先取最后一个能解析出 JSON 且含 overall_score 的消息.
    """
    from core.schemas import MatchResult

    last_valid = None
    for msg in reversed(messages):
        source = getattr(msg, "source", "")
        content = str(getattr(msg, "content", ""))
        if not content or source == "user":
            continue
        try:
            data = _extract_json(content)
            if isinstance(data, dict) and "overall_score" in data:
                # 校验是否符合 MatchResult schema
                last_valid = MatchResult.model_validate(data)
                break
        except Exception:
            continue

    if last_valid is None:
        # 兜底: 返回空结果
        logger.warning("matcher team: no valid JSON found in messages, using fallback")
        return MatchResult(
            overall_score=0,
            dimensions=[],
            strengths=[],
            weaknesses=[],
            reflection_note="matcher team failed to produce valid output",
            confidence="low",
        ).model_dump(exclude_none=True)

    return last_valid.model_dump(exclude_none=True)


# ============================================================
# Tool 4: Reporter
# ============================================================

_REPORT_SYSTEM = """你是一个资深的招聘报告撰写专家.
基于匹配度评估结果和面试题, 生成候选人最终评估报告.

【输出 JSON】
{
  "candidate_name": "字符串",
  "job_title": "字符串",
  "recommendation": "strong_recommend|recommend|neutral|not_recommend",
  "recommendation_reason": "10-200字解释",
  "needs_human_review": true/false,
  "human_review_reason": "字符串或null"
}

【recommendation 标准】
- strong_recommend: overall_score >= 85, must_have 全满足
- recommend: overall_score >= 70, must_have 满足率 >= 80%
- neutral: overall_score >= 50, 有 1-2 个 must_have 弱
- not_recommend: overall_score < 50

【needs_human_review 触发】
- overall_score 60-75 之间
- confidence 为 low
- must_have 有 evidence 为"无"
- recommendation 为 neutral"""


def _generate_report_tool(
    jd_dict: dict, resume_dict: dict, match_dict: dict, questions_dict: dict | None = None
) -> dict:
    """Tool: 生成最终报告 → CandidateReport dict."""
    from core.llm_factory import get_llm
    from core.schemas import CandidateReport
    from core.structured_output import structured_call

    client = get_llm("minimax")
    prompt = (
        f"【JD 岗位】{jd_dict.get('job_title', '')}\n"
        f"【候选人】{resume_dict.get('candidate_name', '')}, "
        f"{resume_dict.get('years_of_experience', 0)} 年经验\n"
        f"【匹配度结果】\n{json.dumps(match_dict, ensure_ascii=False, indent=2)}\n"
    )
    if questions_dict:
        prompt += f"【面试题】\n{json.dumps(questions_dict, ensure_ascii=False, indent=2)}\n"
    prompt += "请生成最终报告."

    result = structured_call(client, system=_REPORT_SYSTEM, user=prompt, output_model=CandidateReport)
    return result.model_dump(exclude_none=True)


# ============================================================
# Supervisor Agent (AutoGen AssistantAgent with tools)
# ============================================================

_SUPERVISOR_SYSTEM = """你是 AutoHire 招聘评估系统的主管 Agent.

你有 4 个工具可用:
1. parse_jd: 解析 JD 文本
2. parse_resume: 解析简历文本
3. run_match_assessment: 评估匹配度 (Agent 协作)
4. generate_report: 生成最终报告

请按顺序执行:
1. 先解析 JD
2. 再解析简历
3. 然后评估匹配度
4. 最后生成报告

每一步完成后, 将结果传递给下一步. 所有任务完成后说 TERMINATE."""


# ============================================================
# Pipeline 入口
# ============================================================

@dataclass
class AutoGenPipelineResult:
    """AutoGen pipeline 的执行结果."""
    jd: dict
    resume: dict
    match: dict
    report: dict
    matcher_messages: list[dict]  # Assessor + Refiner 的对话历史
    duration_ms: int = 0


async def run_auto_gen_pipeline(
    jd_text: str,
    resume_text: str,
    *,
    step_callback=None,
) -> AutoGenPipelineResult:
    """执行完整的 AutoGen 主从多智能体 Pipeline.

    流程:
    1. JD Parser Tool (独立 LLM 调用)
    2. Resume Parser Tool (独立 LLM 调用)
    3. Matcher Team (SelectorGroupChat: Assessor + Refiner 协作)
    4. Reporter Tool (独立 LLM 调用)

    Returns:
        AutoGenPipelineResult 含所有中间结果
    """

    def _cb(name: str, status: str, dur: int = 0, **kw):
        if step_callback:
            step_callback(name, status, duration_ms=dur, **kw)

    t0 = time.time()

    # Step 1: Parse JD
    _cb("parse_jd", "running")
    t1 = time.time()
    jd_dict = _parse_jd_tool(jd_text)
    dur1 = int((time.time() - t1) * 1000)
    _cb("parse_jd", "success", dur1)
    logger.info("auto_gen: JD parsed in %dms, title=%s", dur1, jd_dict.get("job_title"))

    # Step 2: Parse Resume
    _cb("parse_resume", "running")
    t2 = time.time()
    resume_dict = _parse_resume_tool(resume_text)
    dur2 = int((time.time() - t2) * 1000)
    _cb("parse_resume", "success", dur2)
    logger.info("auto_gen: Resume parsed in %dms, candidate=%s", dur2, resume_dict.get("candidate_name"))

    # Step 3: Matcher Team (Agent 协作 - SelectorGroupChat)
    _cb("match_with_reflection", "running")
    t3 = time.time()
    match_dict = await _run_matcher_team(jd_dict, resume_dict)
    dur3 = int((time.time() - t3) * 1000)
    _cb("match_with_reflection", "success", dur3)
    logger.info(
        "auto_gen: Match done in %dms, score=%d",
        dur3, match_dict.get("overall_score", 0),
    )

    # Step 4: Generate Report
    _cb("generate_report", "running")
    t4 = time.time()
    report_dict = _generate_report_tool(jd_dict, resume_dict, match_dict)
    dur4 = int((time.time() - t4) * 1000)
    _cb("generate_report", "success", dur4)
    logger.info("auto_gen: Report done in %dms", dur4)

    total = int((time.time() - t0) * 1000)
    logger.info(
        "auto_gen pipeline done in %.1fs, candidate=%s score=%d",
        total / 1000, resume_dict.get("candidate_name"), match_dict.get("overall_score", 0),
    )

    return AutoGenPipelineResult(
        jd=jd_dict,
        resume=resume_dict,
        match=match_dict,
        report=report_dict,
        matcher_messages=[],  # TODO: 从 SelectorGroupChat 提取
        duration_ms=total,
    )


# ============================================================
# 同步包装 (给 planner.py 调用)
# ============================================================

def run_auto_gen_pipeline_sync(
    jd_text_or_path: str | Path,
    resume_text_or_path: str | Path,
    *,
    jd_is_file: bool = False,
    resume_is_file: bool = False,
    run_interview_questions: bool = True,
    enable_reflection: bool = True,
    llm_provider: str = "minimax",
    step_callback=None,
) -> Any:
    """同步版本, 供 planner.py 调用.

    返回 PipelineContext (与 planner.py 兼容).
    """
    from agents.planner import PipelineContext

    # 读取文本
    if jd_is_file:
        from core.tools.document_parser import parse_any
        jd_text = parse_any(jd_text_or_path)
    else:
        jd_text = str(jd_text_or_path)

    if resume_is_file:
        from core.tools.document_parser import parse_any
        resume_text = parse_any(resume_text_or_path)
    else:
        resume_text = str(resume_text_or_path)

    # 路由检测
    from agents.router import detect_route
    route_decision = detect_route(None, resume_path=resume_text_or_path if resume_is_file else None)

    # 跑 async pipeline
    result = asyncio.run(run_auto_gen_pipeline(jd_text, resume_text, step_callback=step_callback))

    # 转换为 PipelineContext
    from core.schemas import (
        CandidateReport,
        MatchResult,
        ParsedJD,
        ParsedResume,
    )

    ctx = PipelineContext()
    ctx.jd = ParsedJD.model_validate(result.jd)
    ctx.resume = ParsedResume.model_validate(result.resume)
    ctx.match = MatchResult.model_validate(result.match)
    ctx.report = CandidateReport.model_validate(result.report)
    ctx.needs_human_review = ctx.report.needs_human_review
    ctx.human_review_reason = ctx.report.human_review_reason
    ctx.metadata["route"] = route_decision.route
    ctx.metadata["route_reason"] = route_decision.reason
    ctx.metadata["pipeline"] = "autogen"
    ctx.metadata["duration_ms"] = result.duration_ms

    return ctx
