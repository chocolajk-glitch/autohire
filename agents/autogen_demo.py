"""AutoGen 0.4+ 最小双 Agent Demo.

演示目的:
- 验证 AutoGen 0.4+ 在我们环境里能跑
- 验证 OpenAIChatCompletionClient 能用自定义 base_url (即 Qwen / DeepSeek)
- 验证异步 RoundRobinGroupChat 流程

两个 Agent:
- primary_researcher: 拿到任务后输出结论
- reviewer: 复核 primary 的结论, 指出问题

不是生产代码, 仅用于阶段验证.
"""
import asyncio
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


def _build_model_client() -> OpenAIChatCompletionClient:
    """从环境变量读取 DeepSeek 配置; 没配则报错."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    return OpenAIChatCompletionClient(
        model="deepseek-v4-pro",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        # 同步调用, 不走流式; 模型信息补全
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "deepseek",
            "structured_output": False,
        },
    )


async def run_two_agent_demo() -> None:
    client = _build_model_client()

    primary = AssistantAgent(
        name="primary_researcher",
        model_client=client,
        system_message=(
            "You are a senior HR analyst. Given a job description, "
            "produce 3 key hiring criteria in one short paragraph. "
            "Reply concisely in Chinese."
        ),
    )
    reviewer = AssistantAgent(
        name="reviewer",
        model_client=client,
        system_message=(
            "You are a strict reviewer. Read the primary_researcher's output. "
            "If it is acceptable, reply exactly 'TERMINATE'. "
            "Otherwise, point out the problems in one short sentence and ask to fix."
        ),
    )

    team = RoundRobinGroupChat(
        [primary, reviewer],
        termination_condition=TextMentionTermination("TERMINATE"),
        max_turns=6,
    )

    task = (
        "我们要招聘一位 Python 后端工程师, 要求 3 年以上经验, 熟悉 FastAPI 与 PostgreSQL. "
        "请 primary_researcher 给出 3 条关键招聘标准, 然后 reviewer 复核."
    )
    result = await team.run(task=task)
    print("\n========== RESULT ==========")
    for msg in result.messages:
        print(f"[{msg.source}] {msg.content[:200] if isinstance(msg.content, str) else msg.content}")


if __name__ == "__main__":
    asyncio.run(run_two_agent_demo())
