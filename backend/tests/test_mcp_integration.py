"""MCP 集成测试 - 验证 jd_parser 和 resume_parser 走 MCP 路径.

测试 use_mcp=True (默认) 和 use_mcp=False (本地回退) 两条路径都能工作.
"""
import logging

import pytest

from agents.jd_parser import parse_jd_file, parse_jd_text
from agents.resume_parser import parse_resume_file, parse_resume_text

logger = logging.getLogger(__name__)

JD_FILE = "data/jds/backend_python_jd.txt"
RESUME_FILE = "data/resumes/01_zhang_san_strong_backend.pdf"


class TestMCPIntegration:
    def test_parse_jd_file_via_mcp(self):
        """parse_jd_file 默认走 MCP 路径, 应该返回有效 ParsedJD."""
        result = parse_jd_file(JD_FILE, provider="qwen", use_mcp=True)
        assert result.job_title
        assert len(result.requirements) >= 3
        assert any(r.is_must_have for r in result.requirements)
        print(f"\n[MCP parse_jd_file] job_title={result.job_title!r}, requirements={len(result.requirements)}")

    def test_parse_jd_text_local_only(self):
        """parse_jd_text use_mcp=False 走本地 (因为没有 file_path), 仍然应该工作."""
        # parse_jd_text 不走 MCP (没有文件), use_mcp 参数被忽略
        text = """
Senior Python Backend Engineer. Requirements: 3 years Python, FastAPI, PostgreSQL, Docker.
Nice to have: LangGraph, multi-agent systems experience.
"""
        result = parse_jd_text(text, provider="qwen", use_mcp=False)
        assert result.job_title
        assert len(result.requirements) >= 1
        print(f"\n[local parse_jd_text] job_title={result.job_title!r}")

    def test_parse_resume_file_via_mcp(self):
        """parse_resume_file 默认走 MCP 路径."""
        result = parse_resume_file(RESUME_FILE, provider="qwen", use_mcp=True)
        assert result.candidate_name
        assert len(result.skills) >= 3
        assert len(result.work_experiences) >= 1
        print(f"\n[MCP parse_resume_file] candidate={result.candidate_name!r}, skills={len(result.skills)}")

    def test_parse_resume_file_local_fallback(self):
        """parse_resume_file use_mcp=False 强制本地 (跳过 MCP 子进程)."""
        result = parse_resume_file(RESUME_FILE, provider="qwen", use_mcp=False)
        assert result.candidate_name
        assert len(result.skills) >= 3
        # 本地路径也用同样的简历文件, 关键字段应该一致
        print(f"\n[local parse_resume_file] candidate={result.candidate_name!r}, skills={len(result.skills)}")

    def test_mcp_and_local_consistency(self):
        """MCP 路径和本地路径对同一文件, 核心字段 (姓名) 应该一致 (因为模型相同)."""
        mcp_result = parse_resume_file(RESUME_FILE, provider="qwen", use_mcp=True)
        local_result = parse_resume_file(RESUME_FILE, provider="qwen", use_mcp=False)
        # 姓名应该一致 (同一份简历, 同一模型)
        assert mcp_result.candidate_name == local_result.candidate_name
        # 技能数量可能略有差异 (LLM 温度), 但至少 1 个
        assert len(mcp_result.skills) >= 1
        assert len(local_result.skills) >= 1
        print(f"\n[consistency] MCP={mcp_result.candidate_name!r} skills={len(mcp_result.skills)}, "
              f"local={local_result.candidate_name!r} skills={len(local_result.skills)}")