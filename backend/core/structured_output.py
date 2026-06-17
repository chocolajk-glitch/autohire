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
            # 启发式修复: weight/int 字段常见错误 (LLM 给字符串/浮点)
            data = _heuristic_fix(data, output_model)
            instance = output_model.model_validate(data)
            if attempt > 0:
                logger.info("structured_call succeeded on retry %d", attempt)
            return instance
        except ValidationError as e:
            last_err = e
            logger.warning("attempt %d: schema validation failed: %s", attempt, _format_validation_error(e))
            # 把错误信息喂回去让 LLM 修正, 但只给前 3 个错误 (避免修一个坏一个)
            err_msg = _format_validation_error(e)
            current_user = (
                user
                + f"\n\nPrevious output had validation errors:\n{err_msg}\n\n"
                "Fix ONLY the listed errors and output the complete corrected JSON. "
                "Keep all other fields EXACTLY as they were."
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


def _heuristic_fix(data: dict, model: type[BaseModel]) -> dict:
    """在 validate 前尝试修复 LLM 常见错误.

    修复规则 (递归到嵌套模型):
    - weight/int 字段: 字符串 '10' -> int 10; 'required'/'preferred' -> 10/5; 浮点 0-1 -> int(0-10)
    - bool 字段: 'true'/'false' 字符串 -> bool; 1/0 -> bool
    - Literal 字段 (category 等): 未知值映射到 'other'
    """
    if not isinstance(data, dict):
        return data

    def fix_one(val, field):
        """修复单个字段值."""
        if val is None:
            return val
        annotation = field.annotation
        if annotation is None:
            return val
        # 提取 Optional/Union 内的真实类型
        real_type = annotation
        try:
            from typing import get_args, get_origin
            args = get_args(annotation)
            if args:
                # Optional[X] / X | None -> 取第一个非 None
                non_none = [a for a in args if a is not type(None)]
                if non_none:
                    real_type = non_none[0]
        except Exception:
            pass

        # bool 字段
        if real_type is bool and not isinstance(val, bool):
            if isinstance(val, str):
                lv = val.strip().lower()
                if lv in ("true", "yes", "1", "是"):
                    return True
                if lv in ("false", "no", "0", "否"):
                    return False
            if isinstance(val, (int, float)):
                return bool(val)
            return val

        # int 字段
        if real_type is int and not isinstance(val, bool):
            if isinstance(val, str):
                lv = val.strip().lower()
                if lv in ("required", "必备", "必须", "critical"):
                    return 10
                if lv in ("preferred", "加分", "nice", "optional"):
                    return 5
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    return 1
            if isinstance(val, float):
                # 浮点 (如 0.8) -> 推断为 0-10 缩放 (LLM 把 8/10 输出成 0.8)
                if 0 < val <= 1:
                    return max(1, min(10, round(val * 10)))
                return int(val)
            return val

        # Literal 字段 - 不修复, 让 Pydantic validate 报错, LLM 重试时自己改

        # list[InnerModel] - 不在这里处理, 由 fix_data 递归处理
        return val

    def fix_data(d, m):
        """递归修复 dict 对应一个 model."""
        if not isinstance(d, dict):
            return d
        result = {}
        for name, field in m.model_fields.items():
            if name not in d:
                result[name] = d.get(name)
                continue
            val = d[name]
            # 如果是嵌套 model 的字段, 递归
            annotation = field.annotation
            if hasattr(annotation, "model_fields"):  # 嵌套 BaseModel
                result[name] = fix_data(val, annotation)
            elif hasattr(annotation, "__args__") and len(annotation.__args__) == 1:
                inner_type = annotation.__args__[0]
                if isinstance(inner_type, type) and issubclass(inner_type, BaseModel) and isinstance(val, list):
                    result[name] = [fix_data(item, inner_type) for item in val if isinstance(item, dict)]
                else:
                    result[name] = fix_one(val, field)
            else:
                result[name] = fix_one(val, field)
        return result

    return fix_data(data, model)
