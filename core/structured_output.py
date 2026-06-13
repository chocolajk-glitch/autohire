"""结构化输出工具: 把 LLM 输出安全地解析为 Pydantic 模型.

LLM 不一定一次就给出合规 JSON, 所以这里:
1. 提示 LLM 用 fenced ```json``` 块输出
2. 解析失败时重试, 把错误信息喂回给 LLM 让它修
3. 最终用 Pydantic 校验
"""
from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from core.llm_factory import ChatClient

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
_JSON_BARE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json(text: str) -> dict | list:
    """从 LLM 文本里抠出 JSON (优先 fence 块, 否则第一对花括号)."""
    m = _JSON_FENCE.search(text)
    if m:
        return json.loads(m.group(1))
    m = _JSON_BARE.search(text)
    if m:
        return json.loads(m.group(1))
    raise ValueError(f"no JSON object found in LLM output: {text[:200]}...")


def _format_validation_error(err: ValidationError) -> str:
    """把 Pydantic 报错压成可读的简短字符串, 给 LLM 看."""
    lines = []
    for e in err.errors()[:5]:  # 最多 5 条, 避免 prompt 爆炸
        loc = ".".join(str(x) for x in e["loc"])
        lines.append(f"- {loc}: {e['msg']} (got: {e.get('input')!r})")
    return "\n".join(lines)


def structured_call(
    client: ChatClient,
    system: str,
    user: str,
    output_model: type[T],
    *,
    max_retries: int = 2,
) -> T:
    """让 LLM 输出 JSON, 解析+校验, 失败时最多重试 max_retries 次.

    Args:
        client: LLM 客户端 (来自 core.llm_factory)
        system: system prompt
        user: user prompt (含要解析的文本)
        output_model: 期望的 Pydantic 模型类
        max_retries: 校验失败后的重试次数

    Returns:
        校验通过的 output_model 实例
    """
    full_system = (
        system
        + "\n\nIMPORTANT: Respond ONLY with a single JSON object inside a ```json``` code block. "
        "Do not include any other text, explanation, or markdown formatting outside the JSON block."
    )

    last_err: Exception | None = None
    current_user = user
    for attempt in range(max_retries + 1):
        raw = client.chat(full_system, current_user)
        try:
            data = _extract_json(raw)
            instance = output_model.model_validate(data)
            if attempt > 0:
                logger.info("structured_call succeeded on retry %d", attempt)
            return instance
        except ValidationError as e:
            last_err = e
            logger.warning("attempt %d: schema validation failed: %s", attempt, _format_validation_error(e))
            # 把错误信息 + raw 输出喂回去让 LLM 修正
            err_msg = _format_validation_error(e)
            current_user = (
                user
                + f"\n\nYour previous response failed schema validation.\n"
                f"Validation errors:\n{err_msg}\n\n"
                f"Required top-level fields: {list(output_model.model_fields.keys())}\n"
                "Please output a corrected JSON object with ALL required top-level fields."
            )
        except (ValueError, json.JSONDecodeError) as e:
            last_err = e
            logger.warning("attempt %d: JSON parse failed: %s", attempt, str(e)[:200])
            current_user = (
                user
                + "\n\nYour previous response did not contain valid JSON. "
                "Please output ONLY a single ```json``` code block with the correct JSON object."
            )

    raise RuntimeError(
        f"structured_call failed after {max_retries + 1} attempts. Last error: {last_err}"
    )
