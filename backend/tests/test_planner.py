"""Planner 端到端 Pipeline 集成测试.

注意: 这次测试会调 4-5 次真实 LLM, 耗时 3-5 分钟.
不要频繁跑, 仅用于里程碑验证.
"""
import json
from pathlib import Path

import fitz
import pytest

from agents.planner import run_pipeline
from tests.test_jd_parser import SAMPLE_JD
from tests.test_resume_parser import SAMPLE_RESUME_TEXT


def _make_resume_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    text = SAMPLE_RESUME_TEXT
    y = 72
    for line in text.splitlines():
        if line.strip():
            page.insert_text((50, y), line)
        y += 14
    doc.save(path)
    doc.close()


@pytest.fixture(scope="module")
def resume_pdf(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("data") / "zhang_san.pdf"
    _make_resume_pdf(p)
    return p


class TestEndToEnd:
    def test_text_inputs(self):
        """文本输入的端到端: JD 文本 + 简历文本."""
        ctx = run_pipeline(
            jd_text_or_path=SAMPLE_JD,
            resume_text_or_path=SAMPLE_RESUME_TEXT,
            jd_is_file=False,
            resume_is_file=False,
            run_interview_questions=True,
            enable_reflection=True,
            llm_provider="deepseek",
        )

        # 所有步骤都应该成功
        for step in ctx.steps:
            assert step.status == "success", (
                f"step {step.name} failed: {step.error}"
            )

        # 关键字段都填了
        assert ctx.jd is not None
        assert ctx.jd.job_title
        assert ctx.resume is not None
        assert ctx.resume.candidate_name
        assert ctx.match is not None
        assert 0 <= ctx.match.overall_score <= 100
        assert ctx.questions is not None
        assert len(ctx.questions.questions) >= 3
        assert ctx.report is not None
        assert ctx.report.recommendation in (
            "strong_recommend", "recommend", "neutral", "not_recommend"
        )
        assert ctx.report.recommendation_reason

        print(f"\n========== END-TO-END RESULT ==========")
        print(f"Candidate: {ctx.resume.candidate_name} ({ctx.resume.years_of_experience} yrs)")
        print(f"Job: {ctx.jd.job_title}")
        print(f"Score: {ctx.match.overall_score}/100 (confidence: {ctx.match.confidence})")
        print(f"Recommendation: {ctx.report.recommendation}")
        print(f"Reason: {ctx.report.recommendation_reason[:150]}")
        print(f"Reflection: {ctx.match.reflection_note[:150] if ctx.match.reflection_note else '(none)'}")
        print(f"Interview Q count: {len(ctx.questions.questions)}")
        for i, q in enumerate(ctx.questions.questions[:3], 1):
            print(f"  Q{i} [{q.difficulty}/{q.category}]: {q.question[:100]}")
        print(f"Needs HITL: {ctx.needs_human_review}")
        if ctx.needs_human_review:
            print(f"  Reason: {ctx.human_review_reason}")
        print(f"Steps:")
        for s in ctx.steps:
            print(f"  - {s.name}: {s.status} ({s.duration_ms}ms)")

    def test_file_inputs(self, resume_pdf: Path):
        """文件输入的端到端: 用真 PDF 简历 (PyMuPDF 生成)."""
        ctx = run_pipeline(
            jd_text_or_path=SAMPLE_JD,
            resume_text_or_path=str(resume_pdf),
            jd_is_file=False,
            resume_is_file=True,
            run_interview_questions=True,
            enable_reflection=True,
            llm_provider="deepseek",
        )
        # 文件解析至少要成功
        assert ctx.steps[0].status == "success", f"JD parse failed: {ctx.steps[0].error}"
        assert ctx.steps[1].status == "success", f"Resume parse failed: {ctx.steps[1].error}"
        assert ctx.resume is not None
        print(f"\n[file input] candidate={ctx.resume.candidate_name} score={ctx.match.overall_score if ctx.match else 'N/A'}")
