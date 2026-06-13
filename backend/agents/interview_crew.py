"""面试出题 Crew (CrewAI 实现).

三角色协作:
- Researcher: 根据候选人简历 + JD 匹配结果, 列出 3-5 个值得考察的技术点
- QuestionDesigner: 基于技术点设计 3-5 道面试题 (含目标能力/难度/期望答案)
- Reviewer: 审视题目质量, 如有问题打回; 否则输出 PASS

入参: 已解析的 JD + 简历 + 匹配度结果 (都是结构化对象)
出参: InterviewQuestionSet
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai import LLM

from core.schemas import InterviewQuestionSet, MatchResult, ParsedJD, ParsedResume

logger = logging.getLogger(__name__)


def _build_llm(provider: str = "deepseek") -> LLM:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return LLM(
        model="openai/deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key=api_key,
        temperature=0.4,
    )


def _build_inputs(jd: ParsedJD, resume: ParsedResume, match: MatchResult) -> dict[str, Any]:
    """把入参序列化成 CrewAI 的 inputs dict, 占位符引用."""
    jd_text = "\n".join(
        f"- [{r.category} | weight={r.weight} | must={r.is_must_have}] {r.description}"
        for r in jd.requirements
    )
    resume_text = (
        f"Candidate: {resume.candidate_name}\n"
        f"Years of experience: {resume.years_of_experience}\n"
        f"Skills: {', '.join(resume.skills) if resume.skills else '(none)'}\n"
        f"Projects:\n" + "\n".join(
            f"  * {p.name} ({', '.join(p.tech_stack or [])}): {p.description[:200]}"
            for p in resume.projects
        )
    )
    match_text = (
        f"Overall score: {match.overall_score}/100\n"
        f"Strengths: {match.strengths}\n"
        f"Weaknesses: {match.weaknesses}\n"
        f"Per-dimension evidence:\n" + "\n".join(
            f"  - {d.requirement} (score {d.score}/10): {d.candidate_evidence}"
            for d in match.dimensions
        )
    )
    return {
        "jd_requirements": jd_text,
        "resume_summary": resume_text,
        "match_analysis": match_text,
    }


def _create_crew(llm: LLM) -> Crew:
    researcher = Agent(
        role="Senior Technical Researcher",
        goal="Identify 3-5 most testable technical points based on the candidate's actual project experience and the JD's must-have requirements",
        backstory=(
            "You are a senior technical interviewer. You focus on the candidate's project "
            "details and the JD's must-have skills, identifying specific, answerable "
            "technical points (not vague topics)."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    designer = Agent(
        role="Interview Question Designer",
        goal="Design 3-5 high-quality interview questions, each tied to a specific technical point and with a clear expected answer outline",
        backstory=(
            "You craft interview questions that are answerable, have a clear target skill, "
            "an appropriate difficulty level, and an expected answer outline."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    reviewer = Agent(
        role="Quality Reviewer",
        goal="Verify the questions are fair, well-targeted, and not biased",
        backstory=(
            "You are a strict but fair reviewer. If a question is unanswerable from the "
            "candidate's background, too vague, or duplicates another, you reject it."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Given the following context, identify 3-5 most testable technical points.\n\n"
            "JD requirements:\n{jd_requirements}\n\n"
            "Resume summary:\n{resume_summary}\n\n"
            "Match analysis:\n{match_analysis}\n\n"
            "Output: a short bulleted list of 3-5 specific technical points. "
            "Each point should be something the candidate could actually be asked about "
            "based on their listed project experience."
        ),
        expected_output="A short bulleted list of 3-5 testable technical points.",
        agent=researcher,
    )

    design_task = Task(
        description=(
            "Based on the researcher's findings, design 3-5 interview questions.\n\n"
            "Context:\n"
            "JD: {jd_requirements}\n"
            "Resume: {resume_summary}\n"
            "Match: {match_analysis}\n"
            "Researcher's findings are in the previous task output.\n\n"
            "Each question MUST include: question, category, difficulty, target_skill, "
            "expected_answer_outline.\n\n"
            "CRITICAL - YOUR OUTPUT IS THE FINAL DELIVERABLE:\n"
            "After this, a Reviewer agent will give feedback, but the FINAL output of "
            "this crew is YOUR output here. Output ONLY a single JSON object in a "
            "```json``` code block. The JSON must have:\n"
            "- 'questions': an array of 3-5 question objects\n"
            "- 'rationale': a short string explaining the design choices\n"
            "Do not include any other text outside the JSON code block."
        ),
        expected_output=(
            "A single JSON object in a code block, with 'questions' array and 'rationale' string."
        ),
        agent=designer,
    )

    review_task = Task(
        description=(
            "Review the 3-5 questions designed by the Designer. For each question, check:\n"
            "1. Is it answerable from the candidate's actual project experience?\n"
            "2. Does it have a clear target_skill?\n"
            "3. Is the difficulty appropriate (easy/medium/hard)?\n"
            "4. Is there an expected_answer_outline?\n"
            "5. Are there duplicates?\n\n"
            "If all 3-5 questions pass these checks, reply exactly 'PASS'.\n"
            "Otherwise, list the issues concisely and ask for a fix.\n\n"
            "Context:\n"
            "Resume: {resume_summary}\n"
            "JD: {jd_requirements}"
        ),
        expected_output="Either 'PASS' or a concise list of issues to fix.",
        agent=reviewer,
    )

    return Crew(
        agents=[researcher, designer, reviewer],
        tasks=[research_task, design_task, review_task],
        process=Process.sequential,
        verbose=False,
    )


_CATEGORY_ALIASES = {
    "api design": "system_design",
    "database": "technical",
    "db": "technical",
    "sql": "technical",
    "algorithm": "coding",
    "data structure": "coding",
    "data structures": "coding",
    "architecture": "system_design",
    "system": "system_design",
    "project experience": "project",
    "experience": "project",
    "soft skill": "behavioral",
    "communication": "behavioral",
    "teamwork": "behavioral",
    "leadership": "behavioral",
}


def _normalize_category(cat: str) -> str:
    """把 LLM 自由发挥的 category 归一化到 schema 允许的值."""
    if not cat:
        return "other"
    cat_lower = str(cat).strip().lower()
    allowed = {"technical", "project", "behavioral", "system_design", "coding", "other"}
    if cat_lower in allowed:
        return cat_lower
    return _CATEGORY_ALIASES.get(cat_lower, "other")


def _normalize_difficulty(d: str) -> str:
    """大小写归一化 + 常见同义词."""
    if not d:
        return "medium"
    d_lower = str(d).strip().lower()
    allowed = {"easy", "medium", "hard"}
    if d_lower in allowed:
        return d_lower
    return {
        "junior": "easy",
        "beginner": "easy",
        "intermediate": "medium",
        "mid": "medium",
        "senior": "hard",
        "advanced": "hard",
        "expert": "hard",
    }.get(d_lower, "medium")


def _normalize_questions_dict(data: dict) -> dict:
    """在 validate 之前对 questions 列表做 category + difficulty 归一化."""
    if "questions" in data and isinstance(data["questions"], list):
        for q in data["questions"]:
            if isinstance(q, dict):
                if "category" in q:
                    q["category"] = _normalize_category(q["category"])
                if "difficulty" in q:
                    q["difficulty"] = _normalize_difficulty(q["difficulty"])
    return data


def _extract_questions_from_output(raw: str) -> InterviewQuestionSet:
    """从 CrewAI 最后的 raw 输出里抠出 InterviewQuestionSet.

    CrewAI 的 final output 通常是 design_task 的结果 (一个 list of questions).
    我们需要把它映射成 Pydantic 结构.
    """
    from core.structured_output import _extract_json

    data = _extract_json(raw)
    # 兼容三种格式:
    # A. {"questions": [...], "rationale": "..."}   (理想)
    # B. 直接是 [...]                              (list)
    # C. {"<其他>": ..., "questions": [...]}        (嵌在其他字段)
    if isinstance(data, dict):
        if "questions" in data:
            return InterviewQuestionSet.model_validate(_normalize_questions_dict(data))
        # 兜底: 找任何 list 字段
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "question" in v[0]:
                return InterviewQuestionSet.model_validate(
                    _normalize_questions_dict({"questions": v, "rationale": "extracted from crew output"})
                )
    if isinstance(data, list) and data and isinstance(data[0], dict) and "question" in data[0]:
        return InterviewQuestionSet.model_validate(
            _normalize_questions_dict({"questions": data, "rationale": "extracted from crew output"})
        )
    raise ValueError(f"could not extract question list from crew output: {raw[:300]}")


def generate_interview_questions(
    jd: ParsedJD,
    resume: ParsedResume,
    match: MatchResult,
    *,
    provider: str = "deepseek",
) -> InterviewQuestionSet:
    """跑 CrewAI 出面试题.

    Returns:
        InterviewQuestionSet: 至少 3 道题, 最多 10 道
    """
    llm = _build_llm(provider)
    crew = _create_crew(llm)
    inputs = _build_inputs(jd, resume, match)
    result = crew.kickoff(inputs=inputs)
    # 取 design_task 的输出 (顺序: research -> design -> review, 所以是 -2)
    if hasattr(result, "tasks_output") and result.tasks_output:
        # design_task 是倒数第二个 (review 是最后一个)
        design_output = result.tasks_output[-2] if len(result.tasks_output) >= 2 else result.tasks_output[-1]
        raw = (
            design_output.raw
            if hasattr(design_output, "raw")
            else str(design_output)
        )
    else:
        raw = result.raw if hasattr(result, "raw") else str(result)
    return _extract_questions_from_output(raw)
