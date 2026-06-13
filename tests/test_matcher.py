"""matcher 的集成测试 - 用真实 LLM 跑完整流程."""
import pytest

from agents.jd_parser import parse_jd_text
from agents.matcher import MatcherConfig, match_resume_to_jd
from agents.resume_parser import parse_resume_text
from tests.test_jd_parser import SAMPLE_JD
from tests.test_resume_parser import SAMPLE_RESUME_TEXT


@pytest.fixture(scope="module")
def parsed_jd():
    return parse_jd_text(SAMPLE_JD, provider="deepseek")


@pytest.fixture(scope="module")
def parsed_resume():
    return parse_resume_text(SAMPLE_RESUME_TEXT, provider="deepseek")


class TestMatch:
    def test_match_with_reflection(self, parsed_jd, parsed_resume):
        result = match_resume_to_jd(
            parsed_jd,
            parsed_resume,
            config=MatcherConfig(initial_provider="deepseek", reflect_provider="qwen", enable_reflection=True),
        )
        # 简历里明确有 Python/FastAPI/AutoGen 经验, JD 要求 Python 3 年+ FastAPI
        # 整体应该是个中上分数
        assert 0 <= result.overall_score <= 100
        assert result.confidence in ("low", "medium", "high")
        assert len(result.dimensions) >= 1
        assert result.dimensions[0].score >= 0
        # 反思应该有结论
        assert result.reflection_note is not None
        print(f"\n[with reflection] overall={result.overall_score} confidence={result.confidence}")
        print(f"  note: {result.reflection_note[:150]}")
        print(f"  strengths: {result.strengths[:3]}")
        print(f"  weaknesses: {result.weaknesses[:3]}")

    def test_match_without_reflection(self, parsed_jd, parsed_resume):
        result = match_resume_to_jd(
            parsed_jd,
            parsed_resume,
            config=MatcherConfig(enable_reflection=False),
        )
        # 不开反思时, reflection_note 应该为 None
        assert result.reflection_note is None
        assert 0 <= result.overall_score <= 100
        print(f"\n[no reflection] overall={result.overall_score} note={result.reflection_note}")

    def test_weak_match_scenario(self, parsed_jd):
        """故意给一个不匹配的简历, 验证能打低分."""
        weak_resume_text = """
John Smith
Email: john@example.com
SUMMARY
Frontend developer with 1 year of experience.
SKILLS
HTML, CSS, JavaScript, React, Vue
WORK EXPERIENCE
Tiny Startup (2024 - Present)
Frontend Engineer
- Built simple landing pages
"""
        weak_resume = parse_resume_text(weak_resume_text, provider="deepseek")
        result = match_resume_to_jd(
            parsed_jd,
            weak_resume,
            config=MatcherConfig(enable_reflection=False),
        )
        # 前端 + 1 年经验, JD 要 Python 后端 3 年+, 分数应该明显低
        assert result.overall_score < 50, f"weak candidate should score < 50, got {result.overall_score}"
        print(f"\n[weak match] overall={result.overall_score} (expected < 50)")
