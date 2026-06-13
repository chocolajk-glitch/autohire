"""HR 人工协同 (HITL - Human-in-the-Loop).

职责:
- 列出需要 HR 复核的候选人
- 接收 HR 的修改/确认 (override match score, 改 recommendation, 加备注)
- 持久化 HR 决策
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.schemas import CandidateReport, MatchResult, ParsedJD, ParsedResume

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/autohire.db"


def _get_conn(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """建表."""
    with _get_conn(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS hr_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_name TEXT NOT NULL,
                job_title TEXT NOT NULL,
                original_score INTEGER NOT NULL,
                hr_adjusted_score INTEGER,
                original_recommendation TEXT NOT NULL,
                hr_recommendation TEXT,
                hr_note TEXT,
                reviewed_at TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(candidate_name, job_title)
            );
            """
        )


@dataclass
class HRReview:
    candidate_name: str
    job_title: str
    original_score: int
    original_recommendation: str
    hr_adjusted_score: int | None = None
    hr_recommendation: str | None = None
    hr_note: str | None = None
    reviewed_at: str | None = None


def list_pending_reviews(
    job_title: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """列出所有'待 HR 复核'的候选人 (即已生成报告且 needs_human_review=True 但还没被 HR 审)."""
    init_db(db_path)
    with _get_conn(db_path) as conn:
        if job_title:
            rows = conn.execute(
                """SELECT * FROM hr_reviews
                   WHERE job_title = ? AND hr_recommendation IS NULL
                   ORDER BY created_at DESC""",
                (job_title,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM hr_reviews
                   WHERE hr_recommendation IS NULL
                   ORDER BY created_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]


def submit_for_review(
    report: CandidateReport,
    job_title: str,
    db_path: str = DEFAULT_DB_PATH,
) -> None:
    """把一份需要 HR 复核的报告写到数据库."""
    init_db(db_path)
    now = datetime.utcnow().isoformat()
    with _get_conn(db_path) as conn:
        conn.execute(
            """INSERT INTO hr_reviews
               (candidate_name, job_title, original_score, original_recommendation,
                hr_note, created_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(candidate_name, job_title) DO UPDATE SET
                 original_score = excluded.original_score,
                 original_recommendation = excluded.original_recommendation,
                 hr_note = excluded.hr_note,
                 created_at = excluded.created_at""",
            (
                report.candidate_name,
                job_title,
                report.match.overall_score,
                report.recommendation,
                report.human_review_reason,
                now,
            ),
        )
    logger.info("submitted for review: %s (job=%s)", report.candidate_name, job_title)


def submit_hr_decision(
    candidate_name: str,
    job_title: str,
    *,
    adjusted_score: int | None = None,
    recommendation: str | None = None,
    note: str | None = None,
    db_path: str = DEFAULT_DB_PATH,
) -> HRReview:
    """HR 提交决策 (可只填部分字段, 其余保留原值)."""
    init_db(db_path)
    if recommendation is not None:
        allowed = {"strong_recommend", "recommend", "neutral", "not_recommend"}
        if recommendation not in allowed:
            raise ValueError(f"invalid recommendation {recommendation!r}, must be one of {allowed}")
    if adjusted_score is not None and not (0 <= adjusted_score <= 100):
        raise ValueError(f"adjusted_score must be 0-100, got {adjusted_score}")
    now = datetime.utcnow().isoformat()
    with _get_conn(db_path) as conn:
        # 读取原值
        row = conn.execute(
            """SELECT * FROM hr_reviews
               WHERE candidate_name = ? AND job_title = ?""",
            (candidate_name, job_title),
        ).fetchone()
        if row is None:
            raise ValueError(f"no pending review for {candidate_name} ({job_title})")
        final_score = adjusted_score if adjusted_score is not None else row["original_score"]
        final_rec = recommendation if recommendation is not None else row["original_recommendation"]
        final_note = note if note is not None else row["hr_note"]
        conn.execute(
            """UPDATE hr_reviews SET
                 hr_adjusted_score = ?, hr_recommendation = ?, hr_note = ?, reviewed_at = ?
               WHERE candidate_name = ? AND job_title = ?""",
            (final_score, final_rec, final_note, now, candidate_name, job_title),
        )
    logger.info("HR decision: %s -> score=%s rec=%s", candidate_name, final_score, final_rec)
    return HRReview(
        candidate_name=candidate_name,
        job_title=job_title,
        original_score=row["original_score"],
        original_recommendation=row["original_recommendation"],
        hr_adjusted_score=final_score,
        hr_recommendation=final_rec,
        hr_note=final_note,
        reviewed_at=now,
    )
