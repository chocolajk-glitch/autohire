"""Planner - 端到端 Pipeline 协调器.

当前阶段: 确定性线性调度 (按步骤顺序跑各个 Agent)
后续可升级为 AutoGen GroupChat, 让 LLM 决策下一步该跑谁

职责:
1. 进度追踪
2. 错误处理 (某个 Agent 失败时记录但不中断)
3. 上下文传递 (JD/简历/匹配结果在 Agent 间流转)
4. 关键节点触发 HITL
"""
from __future__ import annotations

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
    """
    from agents.interview_crew import generate_interview_questions
    from agents.jd_parser import parse_jd_file, parse_jd_text
    from agents.matcher import MatcherConfig, match_resume_to_jd
    from agents.reporter import HITLConfig, generate_candidate_report
    from agents.resume_parser import parse_resume_file, parse_resume_text

    def _cb(name: str, status: str, duration_ms: int = 0, error: str | None = None) -> None:
        if step_callback:
            logger.info("pipeline step_callback: %s -> %s (dur=%dms)", name, status, duration_ms)
            step_callback(name, status, duration_ms=duration_ms, error=error)

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
