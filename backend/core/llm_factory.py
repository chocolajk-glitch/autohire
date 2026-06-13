"""LLM 工厂 - 统一封装 Qwen / MiniMax / DeepSeek 三家 LLM。

设计原则:
- 用 OpenAI 兼容 SDK (三家都支持)
- 工厂函数返回统一的 ChatClient 对象, 屏蔽 provider 差异
- 支持运行时切换模型, 不重写业务代码
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class LLMConfig:
    """单个 provider 的配置。"""
    provider: str
    api_key: str
    base_url: str
    model: str
    extra_body: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.3


class ChatClient:
    """统一封装的 LLM 客户端, 业务侧只调 chat() / chat_stream()。"""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        """非流式调用, 返回文本。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
        }
        if self.config.extra_body:
            params["extra_body"] = self.config.extra_body
        params.update(kwargs)

        resp = self._client.chat.completions.create(**params)
        return (resp.choices[0].message.content or "").strip()

    def chat_stream(self, system: str, user: str, **kwargs: Any):
        """流式调用, yield 每个 chunk 的文本片段。"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.config.temperature),
            "stream": True,
        }
        if self.config.extra_body:
            params["extra_body"] = self.config.extra_body
        params.update(kwargs)

        stream = self._client.chat.completions.create(**params)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


def _qwen_config() -> LLMConfig:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY not set in .env")
    return LLMConfig(
        provider="qwen",
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        temperature=0.3,
    )


def _MiniMax_config() -> LLMConfig:
    api_key = os.getenv("MiniMax_API_KEY")
    base_url = os.getenv("MiniMax_BASE_URL", "https://api.minimaxi.com/v1")
    if not api_key:
        raise ValueError("MiniMax_API_KEY not set in .env")
    return LLMConfig(
        provider="minimax",
        api_key=api_key,
        base_url=base_url,
        model="MiniMax-M2.7",
        extra_body={"reasoning_split": True},
        temperature=0.3,
    )


def _deepseek_config() -> LLMConfig:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not set in .env")
    return LLMConfig(
        provider="deepseek",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
        extra_body={"thinking": {"type": "enabled"}},
        temperature=0.3,
    )


_REGISTRY = {
    "qwen": _qwen_config,
    "minimax": _MiniMax_config,
    "deepseek": _deepseek_config,
}


def get_llm(provider: str = "deepseek") -> ChatClient:
    """工厂入口: provider ∈ {qwen, minimax, deepseek}."""
    if provider not in _REGISTRY:
        raise ValueError(f"unknown provider '{provider}', choose from {list(_REGISTRY)}")
    return ChatClient(_REGISTRY[provider]())


def list_providers() -> list[str]:
    return list(_REGISTRY.keys())
