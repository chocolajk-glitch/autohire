"""JD / 简历解析集成测试 (重构后, 不再走 MCP, 直接函数调用).

保留文件因为:
1. 测试基础设施还在
2. 验证 parse_jd_file / parse_resume_file 真实可用
3. 跟 test_mcp_integration.py 名字混淆, 未来若需要 MCP 集成再分文件
"""
import logging

from agents.jd_parser import parse_jd_file, parse_jd_text
from agents.resume_parser import parse_resume_file, parse_resume_text

logger = logging.getLogger(__name__)

JD_FILE = "data/jds/backend_python_jd.txt"
RESUME_FILE = "data/resumes/01_zhang_san_strong_backend.pdf"


class TestParserIntegration:
    def test_parse_jd_file(self):
        """parse_jd_file 解析真实 JD 文件, 应该返回有效 ParsedJD."""
        result = parse_jd_file(JD_FILE, provider="qwen")
        assert result.job_title
        assert len(result.requirements) >= 3
        assert any(r.is_must_have for r in result.requirements)
        print(f"\n[parse_jd_file] job_title={result.job_title!r}, requirements={len(result.requirements)}")

    def test_parse_jd_text(self):
        """parse_jd_text 用纯文本输入."""
        text = """
Senior Python Backend Engineer. Requirements: 3 years Python, FastAPI, PostgreSQL, Docker.
Nice to have: LangGraph, multi-agent systems experience.
"""
        result = parse_jd_text(text, provider="qwen")
        assert result.job_title
        assert len(result.requirements) >= 1
        print(f"\n[parse_jd_text] job_title={result.job_title!r}")

    def test_parse_resume_file(self):
        """parse_resume_file 解析真实 PDF 简历."""
        result = parse_resume_file(RESUME_FILE, provider="qwen")
        assert result.candidate_name
        assert len(result.skills) >= 3
        assert len(result.work_experiences) >= 1
        print(f"\n[parse_resume_file] candidate={result.candidate_name!r}, skills={len(result.skills)}")

    def test_parse_resume_text(self):
        """parse_resume_text 用纯文本输入."""
        text = """
Jane Smith
Email: jane@example.com
Skills: Python, FastAPI, PostgreSQL, Docker, Kubernetes
Work Experience:
Senior Engineer at BigCo (2020 - Present)
- Built distributed systems serving 1M users
"""
        result = parse_resume_text(text, provider="qwen")
        assert result.candidate_name
        assert len(result.skills) >= 3
        print(f"\n[parse_resume_text] candidate={result.candidate_name!r}, skills={len(result.skills)}")