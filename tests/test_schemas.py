"""schemas.py 的单元测试 - 验证字段约束、边界条件、嵌套结构。"""
import pytest
from pydantic import ValidationError

from core.schemas import (
    BatchReport,
    CandidateReport,
    Education,
    InterviewQuestion,
    InterviewQuestionSet,
    JDRequirement,
    MatchDimension,
    MatchResult,
    ParsedJD,
    ParsedResume,
    ProjectExperience,
    WorkExperience,
)


# ===== ParsedJD =====
class TestParsedJD:
    def test_valid_minimal(self):
        jd = ParsedJD(
            job_title="后端工程师",
            requirements=[JDRequirement(category="required_skill", content="Python", weight=8, is_must_have=True)],
            summary="我们正在招一位后端工程师, 负责服务端开发。",
        )
        assert jd.job_title == "后端工程师"
        assert len(jd.requirements) == 1

    def test_invalid_weight(self):
        with pytest.raises(ValidationError):
            JDRequirement(category="required_skill", content="Python", weight=11)

    def test_invalid_experience_range(self):
        with pytest.raises(ValidationError):
            ParsedJD(
                job_title="X",
                requirements=[JDRequirement(category="required_skill", content="x", weight=1)],
                summary="xxxxxxxxxx",
                experience_years=(5, 2),  # min > max
            )

    def test_empty_requirements_rejected(self):
        with pytest.raises(ValidationError):
            ParsedJD(job_title="X", requirements=[], summary="summary text here")


# ===== ParsedResume =====
class TestParsedResume:
    def test_valid_resume(self):
        r = ParsedResume(
            candidate_name="张三",
            years_of_experience=3,
            educations=[Education(school="清华大学", degree="bachelor", start_year=2018, end_year=2022)],
            work_experiences=[WorkExperience(company="字节", title="后端", start_date="2022-07")],
            projects=[ProjectExperience(name="AutoHire", description="多 Agent 招聘系统")],
            skills=["Python", "LangGraph"],
        )
        assert r.candidate_name == "张三"
        assert r.skills == ["Python", "LangGraph"]

    def test_invalid_education_years(self):
        with pytest.raises(ValidationError):
            Education(school="X", degree="bachelor", start_year=2022, end_year=2018)

    def test_too_many_skills(self):
        with pytest.raises(ValidationError):
            ParsedResume(
                candidate_name="x",
                years_of_experience=1,
                skills=[f"skill_{i}" for i in range(51)],  # max=50
            )


# ===== MatchResult =====
class TestMatchResult:
    def test_valid_match(self):
        m = MatchResult(
            overall_score=82,
            dimensions=[
                MatchDimension(
                    requirement="Python 3 年+",
                    candidate_evidence="3 年 Python 后端经验 (字节, 2022-2025)",
                    score=9,
                    reasoning="完全满足",
                )
            ],
            strengths=["Python 扎实", "有大厂经历"],
            weaknesses=["Go 经验少"],
            confidence="high",
        )
        assert m.overall_score == 82
        assert m.confidence == "high"

    def test_score_out_of_range(self):
        with pytest.raises(ValidationError):
            MatchDimension(requirement="x", candidate_evidence="x", score=11, reasoning="x")

    def test_overall_score_bounds(self):
        with pytest.raises(ValidationError):
            MatchResult(
                overall_score=101,
                dimensions=[MatchDimension(requirement="x", candidate_evidence="x", score=5, reasoning="xxxxx")],
            )


# ===== InterviewQuestion =====
class TestInterviewQuestionSet:
    def test_min_3_questions(self):
        with pytest.raises(ValidationError):
            InterviewQuestionSet(
                questions=[
                    InterviewQuestion(question="q1?", category="technical", difficulty="easy", target_skill="Python")
                ],
                rationale="rationale here",
            )

    def test_valid_set(self):
        qs = [
            InterviewQuestion(
                question=f"please describe question number {i} in detail?",
                category="technical",
                difficulty="medium",
                target_skill="Python",
            )
            for i in range(3)
        ]
        s = InterviewQuestionSet(questions=qs, rationale="基于候选人项目经验设计")
        assert len(s.questions) == 3


# ===== CandidateReport =====
class TestCandidateReport:
    def test_valid_report(self):
        m = MatchResult(
            overall_score=75,
            dimensions=[MatchDimension(requirement="x", candidate_evidence="x", score=7, reasoning="matches")],
        )
        r = CandidateReport(
            candidate_name="张三",
            job_title="后端工程师",
            match=m,
            recommendation="recommend",
            recommendation_reason="整体匹配度良好, 建议进入下一轮",
        )
        assert r.recommendation == "recommend"
        assert r.needs_human_review is False

    def test_human_review_flag(self):
        m = MatchResult(
            overall_score=60,
            dimensions=[MatchDimension(requirement="x", candidate_evidence="x", score=6, reasoning="matches")],
        )
        r = CandidateReport(
            candidate_name="x",
            job_title="x",
            match=m,
            recommendation="neutral",
            recommendation_reason="边界分数, 需 HR 复核",
            needs_human_review=True,
            human_review_reason="分数恰好在及格线, 且关键证据模糊",
        )
        assert r.needs_human_review is True


# ===== BatchReport =====
class TestBatchReport:
    def test_ranking_must_match_candidates(self):
        m = MatchResult(
            overall_score=80,
            dimensions=[MatchDimension(requirement="x", candidate_evidence="x", score=8, reasoning="ok")],
        )
        c1 = CandidateReport(
            candidate_name="A",
            job_title="X",
            match=m,
            recommendation="recommend",
            recommendation_reason="candidate A is a good fit for the role overall",
        )
        c2 = CandidateReport(
            candidate_name="B",
            job_title="X",
            match=m,
            recommendation="recommend",
            recommendation_reason="candidate B is a good fit for the role overall",
        )

        with pytest.raises(ValidationError):
            # ranking 包含了不存在的名字 C
            BatchReport(job_title="X", total_candidates=2, candidates=[c1, c2], ranking=["A", "B", "C"])
