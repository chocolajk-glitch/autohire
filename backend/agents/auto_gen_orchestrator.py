"""AutoGen 匹配反思编排器.

核心功能: SelectorGroupChat 双 Agent 协作评估
- Assessor: 初评匹配度, 输出 MatchResult JSON
- Refiner: 审查初评, 找出漏判/误判/加权偏差
- 通过 Selector 动态选人 + 终止条件实现反思收敛

调用方: planner.py (_run_pipeline_with_autogen_matcher)
输入: 已解析的 jd_dict + resume_dict (由 planner 提前解析)
输出: MatchResult dict
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ============================================================
# AutoGen Model Client (MiniMax, 无 thinking)
# ============================================================

def _make_model_client():
    """创建 AutoGen 兼容的 model client (MiniMax)."""
    import os
    from autogen_core.models import ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model="MiniMax-M2.7",
        api_key=os.getenv("MiniMax_API_KEY"),
        base_url=os.getenv("MiniMax_BASE_URL", "https://api.minimaxi.com/v1"),
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
    )


# ============================================================
# JSON 提取工具
# ============================================================

import re

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_BARE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json(text: str) -> dict | list:
    """从 LLM 文本中提取 JSON."""
    # 去掉 <think>...</think> 块 (MiniMax thinking)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = _JSON_FENCE.search(text)
    if m:
        return json.loads(m.group(1))
    m = _JSON_BARE.search(text)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"文本中未找到 JSON: {text[:200]}...")


# ============================================================
# Matcher Team (SelectorGroupChat - Assessor + Refiner 协作)
# ============================================================

_ASSESSOR_SYSTEM = """你是一个严格的招聘匹配度评估专家.

你的任务: 基于 JD 和简历, 评估候选人匹配度.

【工作步骤】
1. 对 JD 的每条 requirement, 在简历中找证据
2. 给出该维度的 score (0-10):
   - 0-2: 完全不满足
   - 3-5: 部分满足
   - 6-8: 较好满足
   - 9-10: 完全满足且超出预期
3. candidate_evidence 必须从简历中引用具体信息
4. 给出 overall_score (0-100, 用 weight 加权平均各维度再 ×10)
5. 列出 strengths (3-5条) 和 weaknesses (2-3条)

【输出格式】严格输出以下 JSON, 不要任何其他文字:
```json
{
  "overall_score": 75,
  "dimensions": [
    {"requirement": "要求描述", "candidate_evidence": "从简历找到的证据", "score": 7, "reasoning": "一句话解释"}
  ],
  "strengths": ["优势1", "优势2"],
  "weaknesses": ["不足1", "不足2"],
  "confidence": "low|medium|high"
}
```"""

_REFINER_SYSTEM = """你是一个严格的质量审查员, 负责检查简历匹配度评估.

你会收到评估专家的初评结果. 你的任务:
- 漏判: 简历里有明确证据但被打低分
- 误判: 把等价技能判成"无" (如简历写 Flask, JD 写 Web 框架, 不算完全无)
- 加权偏差: must_have 维度重点不够
- 漏掉的项目经验

【输出格式】严格输出以下 JSON:
```json
{
  "approved": true或false,
  "adjusted_score": 修正后的总分 (0-100, 不改则与原分相同),
  "issues": ["问题1", "问题2"],
  "adjustments": [{"requirement": "某要求", "old_score": 旧分, "new_score": 新分, "reason": "修正原因"}],
  "reflection_note": "一句话总结"
}
```

【终止协议 - 重要】
- 如果你的审查认为评估**准确无误** (approved=true 且无 adjustments), 在 JSON 之后单独一行输出 "TERMINATE" 标志, 提前结束对话节省成本.
- 如果你发现需要修正的问题 (approved=false 或有 adjustments), 不要输出 TERMINATE, 让 Assessor 重新评分.
- TERMINATE 标志必须是独立的一行, 便于系统解析."""


async def _run_matcher_team(jd_dict: dict, resume_dict: dict, route: str | None = None) -> dict:
    """Tool: SelectorGroupChat — Assessor + Refiner 协作评估.

    流程:
    1. Assessor 初评 → 输出 JSON
    2. Refiner 审查 → 输出 JSON
    3. Assessor 回应 (如有修正) → 输出最终结果
    4. 提取最终结果返回

    Args:
        jd_dict: ParsedJD 字典
        resume_dict: ParsedResume 字典
        route: 路由决策 (algorithm_specialist / frontend_specialist / ocr_fallback / standard),
               会注入到 Assessor 的 system prompt 里
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import SelectorGroupChat

    client = _make_model_client()

    jd_text = json.dumps(jd_dict, ensure_ascii=False, indent=2)
    resume_text = json.dumps(resume_dict, ensure_ascii=False, indent=2)
    task_msg = (
        f"请评估以下候选人 vs 职位的匹配度.\n\n"
        f"【JD】\n{jd_text}\n\n"
        f"【简历】\n{resume_text}"
    )

    # 根据路由决策拼接 Assessor 的 system prompt (复用 matcher.py 的注入逻辑)
    from agents.matcher import _build_system_prompt
    assessor_system = _build_system_prompt(route)
    logger.info("autogen matcher: route=%s, assessor system_prompt=%d chars",
                route, len(assessor_system))

    assessor = AssistantAgent(
        name="Assessor",
        model_client=client,
        system_message=assessor_system,
    )

    refiner = AssistantAgent(
        name="Refiner",
        model_client=client,
        system_message=_REFINER_SYSTEM,
    )

    termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(6)

    team = SelectorGroupChat(
        [assessor, refiner],
        model_client=client,
        termination_condition=termination,
    )

    logger.info("matcher team: starting SelectorGroupChat")
    t0 = time.time()
    result = await team.run(task=task_msg)
    elapsed = time.time() - t0
    logger.info("matcher team: done in %.1fs, %d messages", elapsed, len(result.messages))

    # 从消息历史中提取最终评估结果 + 序列化对话历史给前端展示
    match_dict = _extract_match_from_messages(result.messages)
    serialized = _serialize_messages(result.messages)
    return match_dict, serialized


def _serialize_messages(messages: list) -> list[dict]:
    """把 AutoGen 消息历史序列化成 dict 列表, 给前端展示反思对话.

    保留: TextMessage (Agent 发言), SelectSpeakerEvent/SelectorEvent (Selector 选人决策)
    跳过: StopMessage (终止信号), ToolCallSummaryMessage (本项目没用 tool)
    字段: source / type / content / round / id / created_at
    round 编号: 按 Agent 发言顺序递增 (Assessor=1, Refiner=1, Assessor=2, ...)
    """
    serialized = []
    round_counter: dict[str, int] = {}
    for m in messages:
        msg_type = getattr(m, "type", type(m).__name__)
        # 跳过终止信号 (不影响业务, 也不该展示给 HR)
        if msg_type == "StopMessage":
            continue
        source = getattr(m, "source", "") or "unknown"
        content = str(getattr(m, "content", ""))
        # 剥离 MiniMax think 块 (前端不应该看到思考过程)
        content = re.sub(r"思考过程.*?结论", "", content, flags=re.DOTALL)
        content = re.sub(r"思考.*?\n\n", "", content, flags=re.DOTALL)
        round_counter[source] = round_counter.get(source, 0) + 1
        serialized.append({
            "source": source,
            "type": msg_type,
            "content": content,
            "round": round_counter[source],
            "id": getattr(m, "id", None),
            "created_at": getattr(m, "created_at", None).isoformat() if getattr(m, "created_at", None) else None,
        })
    logger.info("matcher team: serialized %d messages (skipped StopMessage)", len(serialized))
    return serialized


def _extract_match_from_messages(messages: list) -> dict:
    """从 Agent 消息历史中提取最终的 MatchResult dict.

    策略: 优先取最后一个能解析出 JSON 且含 overall_score 的消息.
    """
    from core.schemas import MatchResult

    last_valid = None
    for msg in reversed(messages):
        source = getattr(msg, "source", "")
        content = str(getattr(msg, "content", ""))
        if not content or source == "user":
            continue
        try:
            data = _extract_json(content)
        except Exception as e:
            # 这条消息不是 JSON (常见于 Refiner 的审查结论是 markdown 表格), 跳过
            logger.debug("matcher team: msg from %s not JSON, skip (%s)", source, str(e)[:80])
            continue
        if not isinstance(data, dict) or "overall_score" not in data:
            logger.debug("matcher team: msg from %s has no overall_score, skip", source)
            continue
        try:
            last_valid = MatchResult.model_validate(data)
            break
        except Exception as e:
            # 提取到了含 overall_score 的 JSON, 但不符合 MatchResult schema
            # (常见: LLM 偷懒只给总分, dimensions 为空或字段缺失)
            logger.warning(
                "matcher team: msg from %s has overall_score but failed schema: %s | raw keys=%s",
                source, str(e)[:150], list(data.keys()),
            )
            continue

    if last_valid is None:
        # 兜底: 必须符合 MatchResult 契约 (dimensions min_length=1), 否则自己就会抛 ValidationError
        # 塞一条占位 dimension, overall_score=0 + confidence=low 标记这是降级结果
        logger.warning("matcher team: no valid MatchResult in messages, using placeholder fallback")
        return MatchResult(
            overall_score=0,
            dimensions=[{
                "requirement": "解析失败占位项",
                "candidate_evidence": "无",
                "score": 0,
                "reasoning": "SelectorGroupChat 未能产出合法 MatchResult, 此为兜底占位",
            }],
            strengths=[],
            weaknesses=["匹配度评估失败, 建议人工复核"],
            reflection_note="matcher team failed to produce valid output, placeholder used",
            confidence="low",
        ).model_dump(exclude_none=True)

    return last_valid.model_dump(exclude_none=True)

