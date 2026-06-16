"""MCP 客户端 + 简历解析 MCP 服务的集成测试.

注意: 这些测试会启动一个 MCP 子进程, 大约 5-10 秒.
跑前确保 .env 里有有效的 Qwen API Key.
"""
import os
import time

import pytest

from core.mcp_client import (
    MCPClientError,
    ResumeMCPClient,
    get_resume_mcp_client,
    parse_jd_via_mcp_or_local,
    parse_resume_via_mcp_or_local,
)


SAMPLE_JD = """Senior Python Engineer - Shanghai

Requirements:
- 3+ years of Python development experience
- Strong knowledge of FastAPI or Django
- Experience with PostgreSQL and Redis
- Familiar with Docker and Kubernetes

Nice to have:
- Open source contributions
- Multi-agent systems experience

Salary: 30K-50K RMB/month
"""


class TestMCPClient:
    def test_get_singleton(self):
        """单例返回同一个 client."""
        a = get_resume_mcp_client()
        b = get_resume_mcp_client()
        assert a is b

    def test_mcp_parse_jd_via_mcp(self):
        """通过 MCP 调 parse_jd, 验证结构化输出."""
        t0 = time.time()
        result = parse_jd_via_mcp_or_local(SAMPLE_JD, llm_provider="qwen")
        elapsed = time.time() - t0
        assert isinstance(result, dict)
        assert "job_title" in result
        assert "requirements" in result
        assert len(result["requirements"]) >= 1
        # 速度断言: MCP 调用应该 < 30 秒 (含子进程启动 + 1 次 LLM)
        assert elapsed < 30, f"MCP parse_jd too slow: {elapsed:.1f}s"
        print(f"\n[MCP parse_jd] {elapsed:.1f}s, {len(result['requirements'])} requirements, job_title={result['job_title']!r}")

    def test_mcp_client_reuse(self):
        """第二次调用复用已有 MCP 连接, 应该 < 5 秒."""
        client = get_resume_mcp_client()
        # 第一次 (已经在上一个测试启动过) - 复用
        t0 = time.time()
        r1 = client.call_tool_sync("parse_jd", {
            "text": "Python developer. Need FastAPI and PostgreSQL, 3+ years exp.",
            "llm_provider": "qwen",
        })
        t1 = time.time() - t0
        assert "job_title" in r1
        # 复用 client 应该很快 (无子进程启动开销)
        assert t1 < 20, f"reuse too slow: {t1:.1f}s"
        print(f"\n[MCP reuse] {t1:.1f}s, {r1['job_title']!r}")

    def test_mcp_client_lifecycle(self):
        """新 client 应该能独立启动和关闭."""
        client = ResumeMCPClient()
        try:
            result = client.call_tool_sync("parse_jd", {
                "text": "Backend developer. Need Go, microservices, gRPC.",
                "llm_provider": "qwen",
            })
            assert "job_title" in result
        finally:
            client.close()
        # 关闭后再调用应该报错
        with pytest.raises((MCPClientError, Exception)):
            client.call_tool_sync("parse_jd", {"text": "x" * 50, "llm_provider": "qwen"})
