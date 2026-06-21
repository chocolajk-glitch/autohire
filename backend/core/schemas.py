"""结构化数据模型 - 整个系统的"数据契约"。

所有 Agent 之间传递的数据都用这些 Pydantic 模型, 保证类型安全 + 易校验.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ============================================================
# JD (Job Description) 解析结果
# ============================================================
class JDRequirement(BaseModel):
    """JD 的一条具体要求 (必备技能 / 加分项 / 经验等)."""
    category: Literal["required_skill", "nice_to_have", "experience", "education", "responsibility", "other"]
    description: str = Field(min_length=1, max_length=500)
    weight: int = Field(ge=1, le=10, description="权重 1-10, 越大越重要")
    is_must_have: bool = False


class ParsedJD(BaseModel):
    """JD 解析 Agent 的输出."""
    job_title: str = Field(min_length=1, max_length=100)
    company: str | None = None
    salary_range: str | None = None
    location: str | None = None
    experience_years_min: int | None = Field(default=None, ge=0, le=60, description="最低经验年限")
    experience_years_max: int | None = Field(default=None, ge=0, le=60, description="最高经验年限")
    requirements: list[JDRequirement] = Field(min_length=1, max_length=20)
    summary: str = Field(min_length=10, max_length=1000)

    @field_validator("experience_years_max")
    @classmethod
    def _check_exp_range(cls, v: int | None, info) -> int | None:
        if v is None:
            return v
        lo = info.data.get("experience_years_min")
        if lo is not None and v < lo:
            raise ValueError(f"experience_years_max {v} 小于 min {lo}")
        return v


# ============================================================
# 简历 (Resume) 解析结果
# ============================================================
class Education(BaseModel):
    school: str
    degree: Literal["high_school", "associate", "bachelor", "master", "phd", "other"]
    major: str | None = None
    start_year: int
    end_year: int

    @field_validator("end_year")
    @classmethod
    def _check_years(cls, v: int, info) -> int:
        start = info.data.get("start_year")
        if start is not None and v < start:
            raise ValueError(f"end_year {v} 小于 start_year {start}")
        return v


class ProjectExperience(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    role: str | None = None
    description: str = Field(min_length=1, max_length=2000)
    tech_stack: list[str] = Field(default_factory=list, max_length=30)
    duration_months: int | None = Field(default=None, ge=1, le=600)


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str = Field(description="YYYY-MM or YYYY-MM-DD")
    end_date: str | None = Field(default=None, description="null 表示至今")
    description: str | None = None


class ParsedResume(BaseModel):
    """简历解析 Agent 的输出."""
    candidate_name: str
    email: str | None = None
    phone: str | None = None
    years_of_experience: int = Field(ge=0, le=60)
    educations: list[Education] = Field(default_factory=list, max_length=10)
    work_experiences: list[WorkExperience] = Field(default_factory=list, max_length=20)
    projects: list[ProjectExperience] = Field(default_factory=list, max_length=20)
    skills: list[str] = Field(default_factory=list, max_length=50)
    self_summary: str | None = Field(default=None, max_length=2000)


# ============================================================
# 匹配度 (Match) 评估结果
# ============================================================
class MatchDimension(BaseModel):
    """JD 某条要求 vs 简历的匹配维度评分."""
    requirement: str
    candidate_evidence: str = Field(description="从简历中找到的证据; 无证据则写 '无'")
    score: int = Field(ge=0, le=10, description="0-10 分")
    reasoning: str = Field(min_length=1, max_length=500)


class MatchResult(BaseModel):
    """匹配度 Agent 的输出, 含反思重判痕迹."""
    overall_score: int = Field(ge=0, le=100, description="加权后的总评分")
    dimensions: list[MatchDimension] = Field(min_length=1, max_length=20)
    strengths: list[str] = Field(default_factory=list, max_length=10)
    weaknesses: list[str] = Field(default_factory=list, max_length=10)
    reflection_note: str | None = Field(
        default=None,
        description="反思记录: 第一次打分时漏掉/误判了什么, 修正后结果如何",
    )
    confidence: Literal["low", "medium", "high"] = "medium"


# ============================================================
# 面试题
# ============================================================
class InterviewQuestion(BaseModel):
    question: str = Field(min_length=10, max_length=500)
    category: Literal["technical", "project", "behavioral", "system_design", "coding", "other"]
    difficulty: Literal["easy", "medium", "hard"]
    target_skill: str = Field(description="这道题主要考察候选人哪方面能力")
    expected_answer_outline: str | list[str] | None = None


class InterviewQuestionSet(BaseModel):
    """面试出题 Crew 的输出."""
    questions: list[InterviewQuestion] = Field(min_length=3, max_length=10)
    rationale: str = Field(min_length=10, max_length=1000, description="为什么出这几道题")


# ============================================================
# 最终报告
# ============================================================
class CandidateReport(BaseModel):
    """一份简历的完整评估报告 (报告 Agent 输出)."""
    candidate_name: str
    job_title: str
    match: MatchResult
    interview_questions: InterviewQuestionSet | None = None
    recommendation: Literal["strong_recommend", "recommend", "neutral", "not_recommend"]
    recommendation_reason: str = Field(min_length=10, max_length=500)
    needs_human_review: bool = False
    human_review_reason: str | None = None
    # AutoGen SelectorGroupChat 完整对话历史 (含 Assessor/Refiner 发言 + Selector 选人决策)
    # 仅 use_autogen=True 时填充, 降级到单评或未启用反思时为 None
    reflection_messages: list[dict] | None = None


class BatchReport(BaseModel):
    """批量评估的汇总报告."""
    job_title: str
    total_candidates: int = Field(ge=0)
    candidates: list[CandidateReport]
    ranking: list[str] = Field(description="按 overall_score 降序的候选人姓名列表")

    @field_validator("ranking")
    @classmethod
    def _check_ranking(cls, v: list[str], info) -> list[str]:
        names = {c.candidate_name for c in info.data.get("candidates", [])}
        missing = [n for n in v if n not in names]
        if missing:
            raise ValueError(f"ranking 包含未知姓名: {missing}")
        return v
