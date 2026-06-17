"""Planner - 端到端 Pipeline 协调器.

两种模式:
1. 默认模式: 确定性线性调度 (按步骤顺序跑各个 Agent)
2. AutoGen 模式 (use_autogen=True): Matcher 环节用 SelectorGroupChat (Agent 协作)

职责:
1. 进度追踪
2. 错误处理 (某个 Agent 失败时记录但不中断)
3. 上下文传递 (JD/简历/匹配结果在 Agent 间流转)
4. 关键节点触发 HITL
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.schemas import (
    CandidateReport,
    InterviewQuestionSet,
    MatchResult,
    ParsedJD,
    ParsedResume,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    name: str
    status: str = "pending"  # pending / running / success / failed / skipped
    duration_ms: int = 0
    error: str | None = None


@dataclass
class PipelineContext:
    """端到端 Pipeline 的上下文容器."""
    jd: ParsedJD | None = None
    resume: ParsedResume | None = None
    match: MatchResult | None = None
    questions: InterviewQuestionSet | None = None
    report: CandidateReport | None = None
    steps: list[PipelineStep] = field(default_factory=list)
    needs_human_review: bool = False
    human_review_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, name: str) -> PipelineStep:
        s = PipelineStep(name=name)
        self.steps.append(s)
        return s

    def to_dict(self) -> dict:
        return {
            "candidate": self.resume.candidate_name if self.resume else None,
            "job_title": self.jd.job_title if self.jd else None,
            "match_score": self.match.overall_score if self.match else None,
            "needs_human_review": self.needs_human_review,
            "human_review_reason": self.human_review_reason,
            "steps": [
                {"name": s.name, "status": s.status, "duration_ms": s.duration_ms, "error": s.error}
                for s in self.steps
            ],
            "report": self.report.model_dump(exclude_none=True) if self.report else None,
        }


def run_pipeline(
    jd_text_or_path: str | Path,
    resume_text_or_path: str | Path,
    *,
    jd_is_file: bool = False,
    resume_is_file: bool = False,
    run_interview_questions: bool = True,
    enable_reflection: bool = True,
    llm_provider: str = "deepseek",
    step_callback=None,
    use_autogen: bool = False,
) -> PipelineContext:
    """跑一份"JD + 简历"的完整评估 Pipeline.

    Args:
        jd_text_or_path: JD 文本或文件路径
        resume_text_or_path: 简历文本或文件路径
        jd_is_file: True 表示 jd_text_or_path 是文件路径
        resume_is_file: True 表示 resume_text_or_path 是文件路径
        run_interview_questions: 是否跑 CrewAI 出面试题 (默认 True)
        enable_reflection: 匹配度是否跑反思 (默认 True)
        llm_provider: 默认 LLM provider
        step_callback: 可选 fn(step_name, status, duration_ms, error), 每步开始/完成时调用
        use_autogen: True 则走 AutoGen 主从多智能体 pipeline (Matcher 用 SelectorGroupChat)
    """
    # AutoGen pipeline: Matcher 环节用 SelectorGroupChat (Agent 协作)
    # 其他环节 (JD/Resume 解析 + 出题 + 报告) 仍用现有逻辑
    if use_autogen:
        return _run_pipeline_with_autogen_matcher(
            jd_text_or_path, resume_text_or_path,
            jd_is_file=jd_is_file, resume_is_file=resume_is_file,
            run_interview_questions=run_interview_questions,
            enable_reflection=enable_reflection,
            llm_provider=llm_provider,
            step_callback=step_callback,
        )
    from agents.interview_crew import generate_interview_questions
    from agents.jd_parser import parse_jd_file, parse_jd_text
    from agents.matcher import MatcherConfig, match_resume_to_jd
    from agents.reporter import HITLConfig, generate_candidate_report
    from agents.resume_parser import parse_resume_file, parse_resume_text

    def _cb(name: str, status: str, duration_ms: int = 0, error: str | None = None, **extra: Any) -> None:
        if step_callback:
            logger.info("pipeline step_callback: %s -> %s (dur=%dms, extra=%s)", name, status, duration_ms, list(extra.keys()))
            step_callback(name, status, duration_ms=duration_ms, error=error, **extra)

    ctx = PipelineContext()
    t0 = time.time()

    # Step 1: 解析 JD
    s = ctx.add_step("parse_jd")
    s.status = "running"
    _cb(s.name, "running")
    try:
        if jd_is_file:
            ctx.jd = parse_jd_file(jd_text_or_path, provider=llm_provider)
        else:
            ctx.jd = parse_jd_text(jd_text_or_path, provider=llm_provider)
        s.status = "success"
    except Exception as e:
        s.status = "failed"
        s.error = str(e)[:300]
        s.duration_ms = int((time.time() - t0) * 1000)
        _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
        logger.exception("parse_jd failed")
        return ctx
    s.duration_ms = int((time.time() - t0) * 1000)
    _cb(s.name, "success", duration_ms=s.duration_ms)

    # Step 1.5: 动态路由决策 - 分析 JD + 简历特征, 决定走哪条路径
    # 只在简历是文件时做 OCR 检测; 否则只根据 JD 关键词分类
    from agents.router import detect_route
    # 先读一下简历文本 (为了 OCR 检测); 不依赖 LLM
    resume_text_for_route = None
    if resume_is_file:
        try:
            from core.tools.document_parser import parse_any
            resume_text_for_route = parse_any(resume_text_or_path)
        except Exception:
            pass

    route_decision = detect_route(
        ctx.jd,
        resume_path=resume_text_or_path if resume_is_file else None,
        resume_text=resume_text_for_route,
    )
    ctx.metadata["route"] = route_decision.route
    ctx.metadata["route_reason"] = route_decision.reason
    ctx.metadata["route_keywords"] = route_decision.matched_keywords
    logger.info(
        "route decision: %s (reason: %s)",
        route_decision.route, route_decision.reason
    )
    # 推 stage 事件 (前端作为 route_detected 特殊 key 处理, 不进 5 阶段进度)
    _cb("route_detected", "success", duration_ms=0,
        route=route_decision.route, reason=route_decision.reason)

    # Step 2: 解析简历
    t1 = time.time()
    s = ctx.add_step("parse_resume")
    s.status = "running"
    _cb(s.name, "running")
    try:
        if resume_is_file:
            ctx.resume = parse_resume_file(resume_text_or_path, provider=llm_provider)
        else:
            ctx.resume = parse_resume_text(resume_text_or_path, provider=llm_provider)
        s.status = "success"
    except Exception as e:
        s.status = "failed"
        s.error = str(e)[:300]
        s.duration_ms = int((time.time() - t1) * 1000)
        _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
        logger.exception("parse_resume failed")
        return ctx
    s.duration_ms = int((time.time() - t1) * 1000)
    _cb(s.name, "success", duration_ms=s.duration_ms)

    # Step 3: 匹配 + 反思
    t2 = time.time()
    s = ctx.add_step("match_with_reflection" if enable_reflection else "match")
    s.status = "running"
    _cb(s.name, "running")
    try:
        ctx.match = match_resume_to_jd(
            ctx.jd,
            ctx.resume,
            config=MatcherConfig(enable_reflection=enable_reflection, initial_provider=llm_provider),
        )
        s.status = "success"
    except Exception as e:
        s.status = "failed"
        s.error = str(e)[:300]
        s.duration_ms = int((time.time() - t2) * 1000)
        _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
        logger.exception("match failed")
        return ctx
    s.duration_ms = int((time.time() - t2) * 1000)
    _cb(s.name, "success", duration_ms=s.duration_ms)

    # Step 4: 面试出题 (CrewAI)
    if run_interview_questions:
        t3 = time.time()
        s = ctx.add_step("interview_questions_crew")
        s.status = "running"
        _cb(s.name, "running")
        try:
            ctx.questions = generate_interview_questions(
                ctx.jd, ctx.resume, ctx.match, provider=llm_provider
            )
            s.status = "success"
        except Exception as e:
            s.status = "failed"
            s.error = str(e)[:300]
            s.duration_ms = int((time.time() - t3) * 1000)
            _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
            logger.warning("interview crew failed, continuing without questions")
        else:
            s.duration_ms = int((time.time() - t3) * 1000)
            _cb(s.name, "success", duration_ms=s.duration_ms)

    # Step 5: 生成报告 + HITL 判定
    t4 = time.time()
    s = ctx.add_step("generate_report")
    s.status = "running"
    _cb(s.name, "running")
    try:
        ctx.report = generate_candidate_report(
            ctx.jd,
            ctx.resume,
            ctx.match,
            questions=ctx.questions,
            provider=llm_provider,
            hitl_cfg=HITLConfig(),
        )
        ctx.needs_human_review = ctx.report.needs_human_review
        ctx.human_review_reason = ctx.report.human_review_reason
        s.status = "success"
    except Exception as e:
        s.status = "failed"
        s.error = str(e)[:300]
        s.duration_ms = int((time.time() - t4) * 1000)
        _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
        logger.exception("report failed")
        return ctx
    s.duration_ms = int((time.time() - t4) * 1000)
    _cb(s.name, "success", duration_ms=s.duration_ms)

    logger.info(
        "pipeline done in %.1fs, candidate=%s score=%d hitl=%s",
        time.time() - t0,
        ctx.resume.candidate_name,
        ctx.match.overall_score,
        ctx.needs_human_review,
    )
    return ctx


def _run_pipeline_with_autogen_matcher(
    jd_text_or_path: str | Path,
    resume_text_or_path: str | Path,
    *,
    jd_is_file: bool = False,
    resume_is_file: bool = False,
    run_interview_questions: bool = True,
    enable_reflection: bool = True,
    llm_provider: str = "minimax",
    step_callback=None,
) -> PipelineContext:
    """AutoGen 版 Pipeline: Matcher 环节用 SelectorGroupChat (Agent 协作).

    其他环节 (JD/Resume 解析 + 出题 + 报告) 仍用现有函数.
    只有 Matcher 的 Assessor + Refiner 走 AutoGen SelectorGroupChat.
    """
    from agents.auto_gen_orchestrator import _run_matcher_team
    from agents.jd_parser import parse_jd_file, parse_jd_text
    from agents.reporter import HITLConfig, generate_candidate_report
    from agents.resume_parser import parse_resume_file, parse_resume_text
    from core.schemas import MatchResult

    def _cb(name: str, status: str, duration_ms: int = 0, error: str | None = None, **extra: Any) -> None:
        if step_callback:
            step_callback(name, status, duration_ms=duration_ms, error=error, **extra)

    ctx = PipelineContext()
    t0 = time.time()

    # Step 1: 解析 JD
    s = ctx.add_step("parse_jd")
    s.status = "running"
    _cb(s.name, "running")
    try:
        if jd_is_file:
            ctx.jd = parse_jd_file(jd_text_or_path, provider=llm_provider)
        else:
            ctx.jd = parse_jd_text(jd_text_or_path, provider=llm_provider)
        s.status = "success"
    except Exception as e:
        s.status = "failed"
        s.error = str(e)[:300]
        s.duration_ms = int((time.time() - t0) * 1000)
        _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
        logger.exception("parse_jd failed")
        return ctx
    s.duration_ms = int((time.time() - t0) * 1000)
    _cb(s.name, "success", duration_ms=s.duration_ms)

    # Step 1.5: 动态路由
    from agents.router import detect_route
    resume_text_for_route = None
    if resume_is_file:
        try:
            from core.tools.document_parser import parse_any
            resume_text_for_route = parse_any(resume_text_or_path)
        except Exception:
            pass
    route_decision = detect_route(
        ctx.jd,
        resume_path=resume_text_or_path if resume_is_file else None,
        resume_text=resume_text_for_route,
    )
    ctx.metadata["route"] = route_decision.route
    ctx.metadata["route_reason"] = route_decision.reason
    ctx.metadata["route_keywords"] = route_decision.matched_keywords
    ctx.metadata["pipeline"] = "autogen_matcher"
    _cb("route_detected", "success", duration_ms=0,
        route=route_decision.route, reason=route_decision.reason)

    # Step 2: 解析简历
    t1 = time.time()
    s = ctx.add_step("parse_resume")
    s.status = "running"
    _cb(s.name, "running")
    try:
        if resume_is_file:
            ctx.resume = parse_resume_file(resume_text_or_path, provider=llm_provider)
        else:
            ctx.resume = parse_resume_text(resume_text_or_path, provider=llm_provider)
        s.status = "success"
    except Exception as e:
        s.status = "failed"
        s.error = str(e)[:300]
        s.duration_ms = int((time.time() - t1) * 1000)
        _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
        logger.exception("parse_resume failed")
        return ctx
    s.duration_ms = int((time.time() - t1) * 1000)
    _cb(s.name, "success", duration_ms=s.duration_ms)

    # Step 3: 匹配 — AutoGen SelectorGroupChat (Agent 协作)
    t2 = time.time()
    s = ctx.add_step("autogen_matcher_team")
    s.status = "running"
    _cb(s.name, "running")
    try:
        # 把 ParsedJD/ParsedResume 转为 dict 传给 SelectorGroupChat
        jd_dict = ctx.jd.model_dump(exclude_none=True)
        resume_dict = ctx.resume.model_dump(exclude_none=True)
        # 跑异步 SelectorGroupChat
        match_dict = asyncio.run(_run_matcher_team(jd_dict, resume_dict))
        ctx.match = MatchResult.model_validate(match_dict)
        s.status = "success"
    except Exception as e:
        s.status = "failed"
        s.error = str(e)[:300]
        s.duration_ms = int((time.time() - t2) * 1000)
        _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
        logger.exception("autogen matcher team failed, falling back to direct call")
        # 兜底: 降级到直接调用
        try:
            from agents.matcher import MatcherConfig, match_resume_to_jd
            ctx.match = match_resume_to_jd(
                ctx.jd, ctx.resume,
                config=MatcherConfig(enable_reflection=enable_reflection, initial_provider=llm_provider),
            )
            s.status = "success"
            s.error = f"fallback to direct call: {str(e)[:100]}"
        except Exception as e2:
            logger.exception("fallback matcher also failed")
            return ctx
    s.duration_ms = int((time.time() - t2) * 1000)
    _cb(s.name, "success", duration_ms=s.duration_ms)

    # Step 4: 面试出题 (CrewAI)
    if run_interview_questions:
        t3 = time.time()
        s = ctx.add_step("interview_questions_crew")
        s.status = "running"
        _cb(s.name, "running")
        try:
            from agents.interview_crew import generate_interview_questions
            ctx.questions = generate_interview_questions(
                ctx.jd, ctx.resume, ctx.match, provider=llm_provider
            )
            s.status = "success"
        except Exception as e:
            s.status = "failed"
            s.error = str(e)[:300]
            s.duration_ms = int((time.time() - t3) * 1000)
            _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
            logger.warning("interview crew failed, continuing without questions")
        else:
            s.duration_ms = int((time.time() - t3) * 1000)
            _cb(s.name, "success", duration_ms=s.duration_ms)

    # Step 5: 生成报告 + HITL
    t4 = time.time()
    s = ctx.add_step("generate_report")
    s.status = "running"
    _cb(s.name, "running")
    try:
        ctx.report = generate_candidate_report(
            ctx.jd, ctx.resume, ctx.match,
            questions=ctx.questions,
            provider=llm_provider,
            hitl_cfg=HITLConfig(),
        )
        ctx.needs_human_review = ctx.report.needs_human_review
        ctx.human_review_reason = ctx.report.human_review_reason
        s.status = "success"
    except Exception as e:
        s.status = "failed"
        s.error = str(e)[:300]
        s.duration_ms = int((time.time() - t4) * 1000)
        _cb(s.name, "failed", duration_ms=s.duration_ms, error=s.error)
        logger.exception("report failed")
        return ctx
    s.duration_ms = int((time.time() - t4) * 1000)
    _cb(s.name, "success", duration_ms=s.duration_ms)

    logger.info(
        "autogen pipeline done in %.1fs, candidate=%s score=%d hitl=%s",
        time.time() - t0,
        ctx.resume.candidate_name,
        ctx.match.overall_score,
        ctx.needs_human_review,
    )
    return ctx
