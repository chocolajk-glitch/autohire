"""CrewAI 最小三角色 Crew Demo.

演示目的:
- 验证 CrewAI 1.14.7 在我们环境里能跑
- 验证 CrewAI 能用自定义 base_url (Qwen / DeepSeek / MiniMax)
- 验证 Agent -> Task -> Crew 流程

三个 Agent 角色 (对应真实项目里"面试出题"场景):
- researcher: 根据候选人简历/项目, 列出可考察的技术点
- question_designer: 基于 researcher 的技术点, 出 3 道面试题
- reviewer: 检查题目质量, 给改进建议

不是生产代码, 仅用于阶段验证.
"""
import os

from crewai import Agent, Crew, Process, Task
from crewai import LLM


def _build_llm() -> LLM:
    """用 DeepSeek (便宜), 演示 CrewAI 接 OpenAI 兼容 API."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    # CrewAI 的 LLM 包装, 通过 provider:model 字符串支持自定义 base_url
    return LLM(
        model="openai/deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key=api_key,
        temperature=0.3,
        litellm_params={"extra_body": {"thinking": {"type": "enabled"}}},
    )


def build_demo_crew() -> Crew:
    llm = _build_llm()

    researcher = Agent(
        role="Technical Researcher",
        goal="Identify the most testable technical points from a candidate's project experience",
        backstory=(
            "You are a senior technical interviewer. You excel at reading a candidate's "
            "project descriptions and pinpointing 3-5 specific technical points worth probing."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    question_designer = Agent(
        role="Interview Question Designer",
        goal="Design 3 high-quality interview questions based on the researcher's findings",
        backstory=(
            "You are an expert at crafting interview questions. Each question should target "
            "a specific technical point and have a clear expected answer outline."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Quality Reviewer",
        goal="Critique the questions for clarity, depth, and fairness",
        backstory=(
            "You are a fair but strict reviewer. You check whether each question is "
            "answerable, has appropriate difficulty, and is not biased."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "Given the following candidate project: "
            "'Built a multi-agent recruitment system using LangGraph + AutoGen, "
            "with FastAPI backend, Vue frontend, and ChromaDB for memory.' "
            "List 3-4 most testable technical points. Output as a short bulleted list."
        ),
        expected_output="A short bulleted list of 3-4 testable technical points.",
        agent=researcher,
    )

    design_task = Task(
        description=(
            "Based on the researcher's findings, design 3 interview questions. "
            "Each question should clearly state what skill it tests."
        ),
        expected_output="3 interview questions with their target skills.",
        agent=question_designer,
    )

    review_task = Task(
        description=(
            "Review the 3 questions. If they are good enough, reply exactly 'PASS'. "
            "Otherwise list the issues in one sentence."
        ),
        expected_output="Either 'PASS' or a short list of issues.",
        agent=reviewer,
    )

    return Crew(
        agents=[researcher, question_designer, reviewer],
        tasks=[research_task, design_task, review_task],
        process=Process.sequential,
        verbose=False,
    )


if __name__ == "__main__":
    crew = build_demo_crew()
    result = crew.kickoff()
    print("\n========== FINAL OUTPUT ==========")
    print(result.raw if hasattr(result, "raw") else str(result))
