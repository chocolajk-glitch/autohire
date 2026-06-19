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
        role="技术研究员 (Technical Researcher)",
        goal="从候选人项目经验中找出最值得考察的技术点",
        backstory=(
            "你是一名资深技术面试官. 你擅长阅读候选人的项目描述, "
            "找出 3-5 个**具体可考察**的技术点."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    question_designer = Agent(
        role="面试题设计师 (Interview Question Designer)",
        goal="基于研究员的发现, 设计 3 道高质量面试题",
        backstory=(
            "你擅长设计面试题. 每道题针对具体技术点, 附清晰的期望答案大纲."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="质量审核员 (Quality Reviewer)",
        goal="审核题目是否清晰、有深度、公平",
        backstory=(
            "你是公平但严格的审核员. 检查每道题是否可答、难度合适、无偏见."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            "给定以下候选人项目: "
            "'用 LangGraph + AutoGen 搭建多 Agent 招聘系统, FastAPI 后端, Vue 前端, ChromaDB 记忆存储.' "
            "列出 3-4 个最值得考察的技术点. 输出简短的项目符号列表."
        ),
        expected_output="3-4 个具体技术点的简短列表",
        agent=researcher,
    )

    design_task = Task(
        description=(
            "基于研究员的发现, 设计 3 道面试题. "
            "每道题要清晰说明考察什么技能."
        ),
        expected_output="3 道面试题 + 对应的目标技能",
        agent=question_designer,
    )

    review_task = Task(
        description=(
            "审核这 3 道题. 如果质量过关, **只回复 'PASS'**. "
            "否则用一句话列出问题."
        ),
        expected_output="要么 'PASS', 要么简短的问题清单",
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
