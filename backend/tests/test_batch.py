"""批量评估集成测试 - 用 3 份简历 + 1 个 JD 跑通, 验证排行榜 + 统计.

注意: 3 份简历 x 1 个 JD = 3 次端到端 pipeline, 约 3-5 分钟.
默认不开反思 / 不出题, 每次约 1 分钟.
"""
from pathlib import Path

import pytest

from agents.batch import run_batch, submit_batch_to_hitl


DATA_DIR = Path(__file__).parent.parent / "data"
RESUMES_DIR = DATA_DIR / "resumes"
JDS_DIR = DATA_DIR / "jds"


class TestBatchSmall:
    def test_three_resumes_one_jd(self, tmp_path):
        # 选 3 份: 强匹配 / 中等 / 弱
        resume_files = [
            RESUMES_DIR / "01_zhang_san_strong_backend.pdf",  # 强后端 -> 90
            RESUMES_DIR / "05_zhou_qi_fullstack.pdf",         # 中等 -> 60
            RESUMES_DIR / "10_xu_shiyong_weak.pdf",           # 弱 -> 5
        ]
        jd_path = JDS_DIR / "backend_python_jd.txt"

        # 用临时 DB, 不污染 data/autohire.db
        db_path = tmp_path / "test_hitl.db"

        batch_report, summary = run_batch(
            jd_path=jd_path,
            resume_paths=resume_files,
            enable_reflection=False,
            run_interview_questions=False,
            llm_provider="deepseek",
        )

        # 基础断言
        assert summary.total == 3
        assert summary.succeeded == 3
        assert summary.failed == 0
        assert summary.avg_score > 0
        assert summary.median_score > 0
        assert len(summary.top_n) == 3
        # 强匹配应该在第一
        assert summary.top_n[0]["candidate"] == "Wang Lei"
        # 弱匹配应该在最后
        assert summary.top_n[-1]["candidate"] == "Xu Shi"

        # 排行榜
        assert batch_report.ranking[0] == "Wang Lei"
        assert batch_report.ranking[-1] == "Xu Shi"

        # 推荐分布
        assert sum(summary.recommendations.values()) == 3

        # 提交 HITL
        n = submit_batch_to_hitl(batch_report, db_path=str(db_path))
        # 边界分数 (Xu Shi 5 分) + 可能的 Wang Lei 都会触发 HITL
        assert n >= 1
        print(f"\n[batch] {summary.to_dict()}")
        print(f"[batch] ranking: {batch_report.ranking}")
        print(f"[batch] hitl submitted: {n}")
