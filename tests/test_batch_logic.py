"""batch.py 的纯逻辑测试 - 不调 LLM, 只验证 _score_bucket / _median / 统计逻辑."""
from agents.batch import BatchSummary, _median, _score_bucket


class TestScoreBucket:
    def test_buckets(self):
        assert _score_bucket(100) == "85-100"
        assert _score_bucket(90) == "85-100"
        assert _score_bucket(85) == "85-100"
        assert _score_bucket(84) == "70-84"
        assert _score_bucket(70) == "70-84"
        assert _score_bucket(69) == "50-69"
        assert _score_bucket(50) == "50-69"
        assert _score_bucket(49) == "30-49"
        assert _score_bucket(30) == "30-49"
        assert _score_bucket(29) == "0-29"
        assert _score_bucket(0) == "0-29"


class TestMedian:
    def test_odd(self):
        assert _median([1, 2, 3]) == 2.0
        assert _median([5, 1, 3]) == 3.0

    def test_even(self):
        assert _median([1, 2, 3, 4]) == 2.5
        assert _median([10, 20]) == 15.0

    def test_empty(self):
        assert _median([]) == 0.0

    def test_single(self):
        assert _median([42]) == 42.0


class TestBatchSummary:
    def test_to_dict(self):
        s = BatchSummary(
            job_title="Backend Engineer",
            total=10,
            succeeded=8,
            failed=2,
            hitl_count=3,
            score_distribution={"85-100": 2, "70-84": 3, "50-69": 2, "30-49": 1, "0-29": 0},
            avg_score=68.5,
            median_score=72.0,
            recommendations={"recommend": 3, "neutral": 4, "not_recommend": 1},
            top_n=[{"candidate": "A", "score": 95, "recommendation": "strong_recommend"}],
            duration_seconds=180.5,
        )
        d = s.to_dict()
        assert d["job_title"] == "Backend Engineer"
        assert d["total"] == 10
        assert d["avg_score"] == 68.5
        assert d["median_score"] == 72.0
        assert d["duration_seconds"] == 180.5
        assert d["top_n"][0]["score"] == 95
