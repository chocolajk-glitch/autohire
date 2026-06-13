"""jd_parser 的集成测试 - 调真实 LLM, 验证输出符合 ParsedJD schema."""
import pytest

from agents.jd_parser import parse_jd_text


SAMPLE_JD = """
Senior Python Backend Engineer - Shanghai

We are looking for a senior Python backend engineer to join our platform team.

Requirements:
- 3+ years of Python development experience
- Strong knowledge of FastAPI or Django
- Experience with PostgreSQL and Redis
- Familiar with Docker and Kubernetes
- Bachelor's degree in Computer Science or related field

Nice to have:
- Experience with LangChain / LangGraph
- Open source contributions
- Experience leading a small team

Responsibilities:
- Design and implement RESTful APIs
- Mentor junior engineers
- Participate in architecture decisions

Salary: 30K-50K RMB/month
"""


class TestParseJDText:
    def test_basic_jd(self):
        result = parse_jd_text(SAMPLE_JD, provider="deepseek")
        assert result.job_title
        assert len(result.requirements) >= 3
        # must have at least one required_skill
        assert any(r.category == "required_skill" for r in result.requirements)
        # must have at least one marked as must_have
        assert any(r.is_must_have for r in result.requirements)
        assert result.experience_years_min is not None
        assert result.experience_years_min >= 3
        assert result.summary
        assert result.salary_range is not None
        # 确认 description 字段被填了
        assert all(r.description for r in result.requirements)
        print(f"\n[DeepSeek] job_title={result.job_title!r}, requirements={len(result.requirements)}, exp_min={result.experience_years_min}")

    def test_qwen_provider(self):
        result = parse_jd_text(SAMPLE_JD, provider="qwen")
        assert result.job_title
        assert len(result.requirements) >= 1
        print(f"\n[Qwen] job_title={result.job_title!r}")

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError):
            parse_jd_text("", provider="deepseek")

    def test_too_short_rejected(self):
        with pytest.raises(ValueError):
            parse_jd_text("hi", provider="deepseek")
