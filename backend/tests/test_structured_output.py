"""structured_output 的单元测试 - 不调 LLM, 只测 JSON 提取和错误处理."""
import pytest
from pydantic import BaseModel, Field

from core.structured_output import _extract_json, _format_validation_error


class Sample(BaseModel):
    name: str
    age: int = Field(ge=0)


class TestExtractJson:
    def test_fenced_json(self):
        text = '```json\n{"name": "x", "age": 1}\n```'
        assert _extract_json(text) == {"name": "x", "age": 1}

    def test_fenced_no_lang(self):
        text = '```\n{"name": "x", "age": 1}\n```'
        assert _extract_json(text) == {"name": "x", "age": 1}

    def test_bare_json(self):
        text = 'some intro\n{"name": "x", "age": 1}\nthanks'
        assert _extract_json(text) == {"name": "x", "age": 1}

    def test_array(self):
        text = '```json\n[{"name": "x", "age": 1}]\n```'
        assert _extract_json(text) == [{"name": "x", "age": 1}]

    def test_no_json(self):
        with pytest.raises(ValueError):
            _extract_json("just plain text with no json at all")


class TestFormatValidationError:
    def test_returns_short_string(self):
        from pydantic import ValidationError as PydValErr

        try:
            Sample.model_validate({"name": "x", "age": -1})
        except PydValErr as e:
            msg = _format_validation_error(e)
            assert "age" in msg
            assert "greater_than_equal" in msg or "Input should be" in msg
