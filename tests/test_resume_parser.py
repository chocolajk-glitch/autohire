"""resume_parser 的集成测试 - 用 PyMuPDF 生成模拟简历验证."""
from pathlib import Path

import fitz
import pytest

from agents.resume_parser import parse_resume_file, parse_resume_text


def _make_sample_resume_pdf(path: Path) -> None:
    """用 PyMuPDF 现场造一份简单模拟简历 PDF."""
    doc = fitz.open()
    page = doc.new_page()

    text = """Zhang San
Email: zhangsan@example.com  |  Phone: 138-0000-0000

EDUCATION
Tsinghua University
Bachelor of Computer Science, 2018 - 2022

WORK EXPERIENCE
ByteDance (2022 - Present)
Senior Backend Engineer
- Designed and built a high-throughput ad ranking system serving 100M+ users
- Led migration of legacy Python 2 monolith to Python 3 + FastAPI microservices
- Mentored 3 junior engineers

Tencent (Summer 2021)
Backend Engineer Intern
- Built internal tool for log analysis using Flask + Elasticsearch

PROJECTS
AutoHire (Multi-Agent Recruitment System) | 2026
- Tech Stack: Python, LangGraph, AutoGen, CrewAI, FastAPI, Vue 3, ChromaDB
- Role: Solo developer
- Description: Built a multi-agent resume screening system with Planner, 5 AutoGen
  agents, and 1 CrewAI 3-role crew. Implements self-reflection and HITL.
- Duration: 3 months

SKILLS
Python, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, LangGraph, AutoGen, CrewAI,
Vue 3, TypeScript, MySQL, ChromaDB, FAISS

SELF SUMMARY
Backend engineer with 3 years of experience, focused on AI infrastructure and
multi-agent systems. Strong in Python ecosystem, LLM application development.
"""

    # 分块写入, 模拟多行
    y = 72
    for line in text.splitlines():
        if line.strip():
            page.insert_text((50, y), line)
        y += 14
    doc.save(path)
    doc.close()


SAMPLE_RESUME_TEXT = """Zhang San
Email: zhangsan@example.com  |  Phone: 138-0000-0000

EDUCATION
Tsinghua University, Bachelor of Computer Science, 2018 - 2022

WORK EXPERIENCE
ByteDance (2022 - Present)
Senior Backend Engineer - Designed high-throughput ad ranking system - Led Python 3 migration - Mentored 3 juniors

PROJECTS
AutoHire - Python, LangGraph, AutoGen, CrewAI - Multi-agent recruitment system, 3 months

SKILLS
Python, FastAPI, PostgreSQL, LangGraph, AutoGen, CrewAI, Vue 3

SUMMARY
Backend engineer with 3 years of experience in Python and AI infra.
"""


class TestParseResumeText:
    def test_basic_resume(self):
        result = parse_resume_text(SAMPLE_RESUME_TEXT, provider="deepseek")
        assert result.candidate_name
        assert result.years_of_experience >= 0
        assert isinstance(result.skills, list)
        assert len(result.skills) >= 3
        # education
        assert len(result.educations) >= 1
        assert result.educations[0].school
        # work experience
        assert len(result.work_experiences) >= 1
        assert result.work_experiences[0].company
        print(f"\n[DeepSeek] name={result.candidate_name!r}, skills={result.skills[:5]}")

    def test_qwen_provider(self):
        result = parse_resume_text(SAMPLE_RESUME_TEXT, provider="qwen")
        assert result.candidate_name
        print(f"\n[Qwen] name={result.candidate_name!r}")

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            parse_resume_text("", provider="deepseek")

    def test_too_short_rejected(self):
        with pytest.raises(ValueError):
            parse_resume_text("just a few words", provider="deepseek")


class TestParseResumeFile:
    def test_parse_generated_pdf(self, tmp_path: Path):
        pdf_path = tmp_path / "sample_resume.pdf"
        _make_sample_resume_pdf(pdf_path)
        assert pdf_path.exists()
        result = parse_resume_file(pdf_path, provider="deepseek")
        assert result.candidate_name
        assert result.years_of_experience >= 0
        # 模拟简历里有 1 个教育、1 个工作、1 个项目
        assert len(result.educations) >= 1
        assert len(result.work_experiences) >= 1
        assert len(result.projects) >= 1
        # 项目里要识别出 tech_stack
        proj = result.projects[0]
        assert len(proj.tech_stack) >= 3
        print(f"\n[PDF] name={result.candidate_name!r}, projects={len(result.projects)}, skills={len(result.skills)}")
