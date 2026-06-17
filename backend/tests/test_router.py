"""router.py 的单元测试 - 纯逻辑, 不调 LLM."""
import pytest

from agents.router import RoutingDecision, detect_route
from core.schemas import JDRequirement, ParsedJD


def _make_jd(title: str, summary: str = "默认测试用 JD summary 摘要内容填充", requirements: list[str] | None = None) -> ParsedJD:
    """构造一个简化的 ParsedJD 用于测试."""
    reqs = requirements if requirements else ["3 years experience"]
    return ParsedJD(
        job_title=title,
        requirements=[
            JDRequirement(category="required_skill", description=r, weight=5, is_must_have=False)
            for r in reqs
        ],
        summary=summary,
    )


class TestDetectRoute:
    def test_algorithm_route(self):
        jd = _make_jd("算法推荐工程师", summary="推荐系统 + DNN")
        decision = detect_route(jd, resume_path="data/resumes/01.pdf")
        assert decision.route == "algorithm_specialist"
        assert "算法" in decision.matched_keywords or "DNN" in decision.matched_keywords

    def test_algorithm_route_english_keyword(self):
        jd = _make_jd("ML Engineer", requirements=["experience with PyTorch"])
        decision = detect_route(jd, resume_path="data/resumes/01.pdf")
        assert decision.route == "algorithm_specialist"
        assert "PyTorch" in decision.matched_keywords

    def test_frontend_route(self):
        jd = _make_jd("Vue 前端工程师", requirements=["熟悉 Vue 3 + TypeScript"])
        decision = detect_route(jd, resume_path="data/resumes/01.pdf")
        assert decision.route == "frontend_specialist"
        assert any("Vue" in kw or "前端" in kw for kw in decision.matched_keywords)

    def test_backend_route_standard(self):
        """Python 后端 JD 没有算法/前端关键词, 走 standard."""
        jd = _make_jd("Python 后端工程师", summary="FastAPI + PostgreSQL", requirements=["3 years Python"])
        decision = detect_route(jd, resume_path="data/resumes/01.pdf")
        assert decision.route == "standard"

    def test_image_resume_ocr_route(self):
        """图片简历走 OCR 路径."""
        jd = _make_jd("Python 后端工程师")
        decision = detect_route(jd, resume_path="data/resumes/scan.jpg")
        assert decision.route == "ocr_fallback"

    def test_pdf_with_no_text_ocr_route(self):
        """PDF 但文本极少 (扫描件) 走 OCR 路径."""
        jd = _make_jd("Python 后端工程师")
        decision = detect_route(jd, resume_path="data/resumes/scan.pdf", resume_text="")
        assert decision.route == "ocr_fallback"

    def test_pdf_with_text_standard_route(self):
        """正常 PDF 简历, 走 standard."""
        jd = _make_jd("Python 后端工程师")
        decision = detect_route(
            jd, resume_path="data/resumes/01.pdf",
            resume_text="张三 Python 后端工程师 5 年经验..." * 10,
        )
        assert decision.route == "standard"

    def test_decision_has_reason(self):
        """每个 RoutingDecision 都应该有 reason (调试用)."""
        jd = _make_jd("算法推荐")
        decision = detect_route(jd)
        assert decision.reason
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0