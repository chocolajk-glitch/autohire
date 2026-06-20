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
    """根据 provider 构建 CrewAI 的 LLM (走 litellm 直连)."""
    if provider == "minimax":
        api_key = os.getenv("MiniMax_API_KEY")
        if not api_key:
            raise RuntimeError("MiniMax_API_KEY 未设置")
        base_url = os.getenv("MiniMax_BASE_URL", "https://api.minimaxi.com/v1")
        return LLM(
            model="MiniMax-M2.7",
            base_url=base_url,
            api_key=api_key,
            temperature=0.4,
        )
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置")
        return LLM(
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=0.4,
        )
    # qwen
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY 未设置")
    return LLM(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
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
        role="高级技术研究员 (Senior Technical Researcher)",
        goal="根据候选人实际项目经验 + JD 的 must_have 要求, 找出 3-5 个最值得考察的技术点",
        backstory=(
            "你是一名资深的技术面试官. 你专注于候选人项目细节和 JD 的 must_have 技能, "
            "找出**具体的、可回答的**技术点 (而不是宽泛的话题)."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    designer = Agent(
        role="面试题设计师 (Interview Question Designer)",
        goal="设计 3-5 道高质量面试题, 每题关联具体技术点 + 有清晰的期望答案大纲",
        backstory=(
            "你设计的题目: 可回答、有明确的目标技能、难度合适、附期望答案大纲."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )
    reviewer = Agent(
        role="质量审核员 (Quality Reviewer)",
        goal="验证题目是否公平、目标明确、没有偏见",
        backstory=(
            "你是严格但公平的审核员. 如果题目: "
            "1) 候选人无法根据简历回答, 2) 过于模糊, 3) 与其他题重复 —— 你就打回."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "基于以下上下文, 找出 3-5 个最值得考察的技术点.\n\n"
            "JD 要求:\n{jd_requirements}\n\n"
            "简历摘要:\n{resume_summary}\n\n"
            "匹配度分析:\n{match_analysis}\n\n"
            "输出: 3-5 个具体技术点的简短列表. 每个点都应该是候选人可以根据其项目经验实际回答的."
        ),
        expected_output="3-5 个具体技术点的简短列表",
        agent=researcher,
    )

    design_task = Task(
        description=(
            "基于 Research 找出的技术点, 设计 3-5 道面试题.\n\n"
            "上下文:\n"
            "JD: {jd_requirements}\n"
            "简历: {resume_summary}\n"
            "匹配度: {match_analysis}\n"
            "Research 找出的考点在上一任务输出里.\n\n"
            "{review_feedback}\n\n"
            "每道题必须包含: question (题目), category (类别), difficulty (难度), "
            "target_skill (目标技能), expected_answer_outline (期望答案大纲).\n\n"
            "**关键 - 你的输出是最终交付物**:\n"
            "后面会有 Reviewer 给出反馈, 但本 Crew 的最终输出就是你这里输出. "
            "**只输出一个 JSON 对象, 用 ```json``` 代码块包裹**. JSON 必须包含:\n"
            "- 'questions': 3-5 道题的数组\n"
            "- 'rationale': 解释设计思路的简短字符串\n"
            "代码块外不要有任何其他文字."
        ),
        expected_output="单个 JSON 对象 (代码块), 含 'questions' 数组和 'rationale' 字符串",
        agent=designer,
    )

    review_task = Task(
        description=(
            "审核 Designer 设计的 3-5 道题. 每道题检查:\n"
            "1. 候选人能根据其项目经验实际回答?\n"
            "2. 有明确的目标技能 (target_skill)?\n"
            "3. 难度合适 (easy/medium/hard)?\n"
            "4. 有期望答案大纲?\n"
            "5. 题目之间有重复?\n\n"
            "如果所有 3-5 道题都通过, **只回复 'PASS'**.\n"
            "否则, 简洁列出问题并要求修复.\n\n"
            "上下文:\n"
            "简历: {resume_summary}\n"
            "JD: {jd_requirements}"
        ),
        expected_output="要么 'PASS', 要么简洁的问题清单",
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
    raise ValueError(f"无法从 Crew 输出中提取面试题列表: {raw[:300]}")


def _get_design_raw(result) -> str:
    """从 CrewAI 结果中提取 design_task 的原始输出."""
    if hasattr(result, "tasks_output") and result.tasks_output:
        design_output = result.tasks_output[-2] if len(result.tasks_output) >= 2 else result.tasks_output[-1]
        return design_output.raw if hasattr(design_output, "raw") else str(design_output)
    return result.raw if hasattr(result, "raw") else str(result)


def _get_review_raw(result) -> str:
    """从 CrewAI 结果中提取 review_task 的原始输出."""
    if hasattr(result, "tasks_output") and result.tasks_output:
        review_output = result.tasks_output[-1]
        return review_output.raw if hasattr(review_output, "raw") else str(review_output)
    return ""


def generate_interview_questions(
    jd: ParsedJD,
    resume: ParsedResume,
    match: MatchResult,
    *,
    provider: str = "deepseek",
) -> InterviewQuestionSet:
    """跑 CrewAI 出面试题.

    如果 Reviewer 未通过, 最多重试一次 (将反馈注入 design_task).

    Returns:
        InterviewQuestionSet: 至少 3 道题, 最多 10 道
    """
    llm = _build_llm(provider)
    inputs = _build_inputs(jd, resume, match)
    inputs["review_feedback"] = ""  # 首次无反馈

    max_attempts = 2
    for attempt in range(max_attempts):
        crew = _create_crew(llm)
        result = crew.kickoff(inputs=inputs)
        review_raw = _get_review_raw(result)

        if "PASS" in review_raw.upper() or attempt == max_attempts - 1:
            # Reviewer 通过, 或已到最大重试次数
            raw = _get_design_raw(result)
            return _extract_questions_from_output(raw)

        # Reviewer 未通过, 注入反馈重试
        logger.info("Reviewer 未通过 (第 %d 次), 反馈: %s", attempt + 1, review_raw[:200])
        inputs["review_feedback"] = f"[审核反馈 - 请根据以下问题修改题目]\n{review_raw}"

    # 不会走到这里, 但保险起见
    raw = _get_design_raw(result)
    return _extract_questions_from_output(raw)
