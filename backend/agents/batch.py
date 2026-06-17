"""批量评估 + 排行榜.

针对 1 份 JD, 评估 N 份简历, 产出:
- BatchReport (含每份的 CandidateReport)
- 排行榜 (按 overall_score 降序)
- 统计: 平均分 / 中位数 / 触发 HITL 的份数 / 各 recommendation 分布
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from agents.planner import PipelineContext, run_pipeline
from core.schemas import BatchReport, CandidateReport, ParsedJD

logger = logging.getLogger(__name__)


@dataclass
class BatchSummary:
    job_title: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    hitl_count: int = 0
    score_distribution: dict[str, int] = field(default_factory=dict)
    avg_score: float = 0.0
    median_score: float = 0.0
    recommendations: dict[str, int] = field(default_factory=dict)
    top_n: list[dict] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "job_title": self.job_title,
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "hitl_count": self.hitl_count,
            "score_distribution": self.score_distribution,
            "avg_score": round(self.avg_score, 1),
            "median_score": round(self.median_score, 1),
            "recommendations": self.recommendations,
            "top_n": self.top_n,
            "duration_seconds": round(self.duration_seconds, 1),
        }


def _score_bucket(s: int) -> str:
    if s >= 85:
        return "85-100"
    if s >= 70:
        return "70-84"
    if s >= 50:
        return "50-69"
    if s >= 30:
        return "30-49"
    return "0-29"


def _median(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def run_batch(
    jd_path: str | Path,
    resume_paths: list[str | Path],
    *,
    jd_text: str | None = None,
    enable_reflection: bool = False,
    run_interview_questions: bool = False,
    llm_provider: str = "deepseek",
    use_autogen: bool = False,
    step_callback=None,
) -> tuple[BatchReport, BatchSummary]:
    """跑批量评估.

    Args:
        jd_path: JD 文件路径 (txt/pdf/docx)
        resume_paths: 简历文件路径列表
        jd_text: 如果不想传文件, 可以直接传 JD 文本 (与 jd_path 二选一)
        enable_reflection: 是否跑匹配度反思
        run_interview_questions: 是否跑面试出题
        llm_provider: LLM provider
        use_autogen: 是否走 AutoGen 主从多智能体 pipeline
        step_callback: 可选, fn(step_name, status, **extra), 每步状态变化时调用

    Returns:
        (BatchReport, BatchSummary)
    """
    from agents.jd_parser import parse_jd_file, parse_jd_text

    def _cb(step: str, status: str, **extra) -> None:
        if step_callback:
            try:
                step_callback(step, status, **extra)
            except Exception:
                logger.warning("step_callback raised, ignored", exc_info=True)

    # 1. 解析 JD (只解析一次)
    _cb("parse_jd", "running")
    if jd_text is not None:
        jd: ParsedJD = parse_jd_text(jd_text, provider=llm_provider)
    else:
        jd = parse_jd_file(jd_path, provider=llm_provider)
    _cb("parse_jd", "success")

    t0 = time.time()
    candidate_reports: list[CandidateReport] = []
    succeeded = 0
    failed = 0
    hitl_count = 0
    scores: list[int] = []
    score_buckets: dict[str, int] = {}
    rec_buckets: dict[str, int] = {}
    top_records: list[dict] = []

    for i, rp in enumerate(resume_paths):
        rp = str(rp)
        cand_stem = Path(rp).stem
        logger.info("[batch] processing %s", rp)

        ctx: PipelineContext = run_pipeline(
            jd_text_or_path=jd_text if jd_text else jd_path,
            resume_text_or_path=rp,
            jd_is_file=(jd_text is None),
            resume_is_file=True,
            run_interview_questions=run_interview_questions,
            enable_reflection=enable_reflection,
            llm_provider=llm_provider,
            use_autogen=use_autogen,
            step_callback=_cb,
        )

        if ctx.report is not None and ctx.resume is not None:
            candidate_reports.append(ctx.report)
            succeeded += 1
            s = ctx.report.match.overall_score
            scores.append(s)
            score_buckets[_score_bucket(s)] = score_buckets.get(_score_bucket(s), 0) + 1
            rec_buckets[ctx.report.recommendation] = rec_buckets.get(ctx.report.recommendation, 0) + 1
            if ctx.report.needs_human_review:
                hitl_count += 1
            top_records.append({
                "candidate": ctx.resume.candidate_name,
                "score": s,
                "recommendation": ctx.report.recommendation,
                "needs_human_review": ctx.report.needs_human_review,
            })
        else:
            failed += 1
            logger.warning("[batch] failed for %s", rp)

    # 排序 + 取 top
    top_records.sort(key=lambda r: r["score"], reverse=True)
    top_n = top_records[:5]

    # 排名
    ranking = [r["candidate"] for r in top_records]

    batch_report = BatchReport(
        job_title=jd.job_title,
        total_candidates=len(resume_paths),
        candidates=candidate_reports,
        ranking=ranking,
    )

    summary = BatchSummary(
        job_title=jd.job_title,
        total=len(resume_paths),
        succeeded=succeeded,
        failed=failed,
        hitl_count=hitl_count,
        score_distribution=score_buckets,
        avg_score=sum(scores) / len(scores) if scores else 0.0,
        median_score=_median(scores),
        recommendations=rec_buckets,
        top_n=top_n,
        duration_seconds=time.time() - t0,
    )
    return batch_report, summary


def submit_batch_to_hitl(batch_report: BatchReport, db_path: str = "data/autohire.db") -> int:
    """把批量结果里所有 needs_human_review 的报告写到 HR 待审队列."""
    from agents.hr_hitl import submit_for_review

    n = 0
    for r in batch_report.candidates:
        if r.needs_human_review:
            submit_for_review(r, job_title=batch_report.job_title, db_path=db_path)
            n += 1
    return n
