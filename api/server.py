"""FastAPI 后端 - AutoHire 招聘筛选系统.

端点:
- GET  /                       - 健康检查
- GET  /docs                   - OpenAPI 文档
- GET  /api/jd                 - 列出可用 JD
- POST /api/jd/upload          - 上传 JD 文本/文件
- POST /api/batch/run          - 触发批量评估 (上传 N 份简历 + 1 份 JD)
- GET  /api/batch/{job_id}/stream - SSE 流式订阅 batch 进度
- GET  /api/batch/{job_id}/result - 获取最终结果
- GET  /api/hitl/pending       - 列出待 HR 复核的候选人
- POST /api/hitl/decide        - HR 提交决策
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.batch import BatchSummary, run_batch, submit_batch_to_hitl
from agents.hr_hitl import list_pending_reviews, submit_hr_decision
from core.schemas import BatchReport

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

DATA_DIR = Path("data")
RESUMES_DIR = DATA_DIR / "resumes"
JDS_DIR = DATA_DIR / "jds"
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AutoHire API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 开发环境允许所有 origin; 生产应改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Job 管理 (in-memory 简化版)
# ============================================================
@dataclass
class JobState:
    job_id: str
    status: str = "pending"  # pending / running / done / error
    progress: float = 0.0
    current_candidate: str = ""
    result: BatchReport | None = None
    summary: BatchSummary | None = None
    error: str | None = None
    log: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    def add_log(self, event: str, **kwargs: Any) -> None:
        entry = {"event": event, "ts": time.time(), **kwargs}
        self.log.append(entry)
        logger.info("[job %s] %s %s", self.job_id, event, kwargs)


_jobs: dict[str, JobState] = {}


# ============================================================
# 健康 / 静态资源
# ============================================================
@app.get("/")
def root() -> dict:
    return {"app": "AutoHire", "version": "0.1.0", "status": "ok"}


@app.get("/api/jd")
def list_jds() -> list[str]:
    if not JDS_DIR.exists():
        return []
    return sorted([p.stem for p in JDS_DIR.glob("*.txt")])


@app.get("/api/resumes")
def list_resumes() -> list[str]:
    if not RESUMES_DIR.exists():
        return []
    return sorted([p.name for p in RESUMES_DIR.glob("*.pdf")])


# ============================================================
# JD 上传
# ============================================================
class JDUploadResponse(BaseModel):
    filename: str
    jd_text: str
    job_title: str = ""


@app.post("/api/jd/upload", response_model=JDUploadResponse)
async def upload_jd(file: UploadFile = File(...)) -> JDUploadResponse:
    """上传 JD 文件 (txt / pdf / docx). 解析后返回结构化文本."""
    from core.tools.document_parser import parse_any

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".pdf", ".docx"}:
        raise HTTPException(400, f"unsupported file type: {suffix}")
    save_path = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}_{file.filename}"
    save_path.write_bytes(await file.read())
    try:
        text = parse_any(save_path)
    except Exception as e:
        raise HTTPException(400, f"failed to parse: {e}")
    # 粗略提取 job_title (第一行非空)
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return JDUploadResponse(filename=file.filename, jd_text=text, job_title=first_line[:100])


# ============================================================
# 简历上传 (单份)
# ============================================================
@app.post("/api/resume/upload")
async def upload_resume(file: UploadFile = File(...)) -> dict:
    """上传一份简历, 保存到 RESUMES_DIR."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(400, f"unsupported file type: {suffix}")
    save_path = RESUMES_DIR / file.filename
    save_path.write_bytes(await file.read())
    return {"filename": file.filename, "path": str(save_path), "size": save_path.stat().st_size}


# ============================================================
# 批量评估 (核心)
# ============================================================
class BatchRequest(BaseModel):
    jd_filename: str | None = None       # 已有 JD 文件名 (在 JDS_DIR 里)
    jd_text: str | None = None           # 或直接传 JD 文本
    resume_filenames: list[str]          # 要评估的简历 (在 RESUMES_DIR 里)
    enable_reflection: bool = False
    run_interview_questions: bool = False
    llm_provider: str = "deepseek"
    auto_submit_hitl: bool = True        # 跑完自动把 needs_human_review 的提交到队列


@app.post("/api/batch/run")
async def trigger_batch(req: BatchRequest) -> dict:
    """触发批量评估, 返回 job_id (后续用此 id 订阅 SSE)."""
    if not req.resume_filenames:
        raise HTTPException(400, "resume_filenames is empty")
    if req.jd_filename is None and req.jd_text is None:
        raise HTTPException(400, "either jd_filename or jd_text required")

    job_id = uuid.uuid4().hex[:12]
    state = JobState(job_id=job_id, status="pending")
    _jobs[job_id] = state
    state.add_log("job_created", total=len(req.resume_filenames))

    # 后台启动
    asyncio.create_task(_run_batch_job(job_id, req))
    return {"job_id": job_id, "status": "pending"}


async def _run_batch_job(job_id: str, req: BatchRequest) -> None:
    """后台跑批量, 推进 state 进度."""
    state = _jobs[job_id]
    state.status = "running"
    state.add_log("started")

    try:
        resume_paths = [RESUMES_DIR / fn for fn in req.resume_filenames]
        for i, p in enumerate(resume_paths):
            if not p.exists():
                state.add_log("missing_resume", path=str(p))
                continue
            state.current_candidate = p.stem
            state.progress = i / len(resume_paths)
            state.add_log("processing", candidate=p.stem, idx=i + 1, total=len(resume_paths))
            await asyncio.sleep(0.05)  # 让出事件循环, 让 SSE 客户端能拿到日志

        # 真正的批量 (同步, 阻塞 1-5 分钟)
        jd_path = JDS_DIR / f"{req.jd_filename}.txt" if req.jd_filename else None
        # 包装成子线程跑, 避免阻塞事件循环
        loop = asyncio.get_running_loop()
        batch_report, summary = await loop.run_in_executor(
            None,
            lambda: run_batch(
                jd_path=jd_path,
                resume_paths=resume_paths,
                jd_text=req.jd_text,
                enable_reflection=req.enable_reflection,
                run_interview_questions=req.run_interview_questions,
                llm_provider=req.llm_provider,
            ),
        )

        state.result = batch_report
        state.summary = summary
        state.progress = 1.0
        state.status = "done"
        state.finished_at = time.time()
        state.add_log("done", avg_score=summary.avg_score, succeeded=summary.succeeded, failed=summary.failed)

        if req.auto_submit_hitl and summary.hitl_count > 0:
            n = submit_batch_to_hitl(batch_report)
            state.add_log("hitl_submitted", count=n)
    except Exception as e:
        state.status = "error"
        state.error = str(e)[:500]
        state.finished_at = time.time()
        state.add_log("error", msg=str(e)[:200])
        logger.exception("batch job %s failed", job_id)


# ============================================================
# SSE 流式订阅
# ============================================================
@app.get("/api/batch/{job_id}/stream")
def stream_batch(job_id: str) -> StreamingResponse:
    """SSE 端点, 推送 job 进度 + 日志 + 最终结果."""
    if job_id not in _jobs:
        raise HTTPException(404, f"job {job_id} not found")
    state = _jobs[job_id]

    def event_gen():
        last_log_idx = 0
        last_progress = -1.0
        last_status = ""
        # 推初始状态
        yield _sse_event("status", {
            "job_id": job_id,
            "status": state.status,
            "progress": state.progress,
            "current_candidate": state.current_candidate,
        })
        while True:
            # 增量推新日志
            while last_log_idx < len(state.log):
                yield _sse_event("log", state.log[last_log_idx])
                last_log_idx += 1
            # 状态变化时推
            if state.progress != last_progress:
                yield _sse_event("progress", {"progress": state.progress, "current_candidate": state.current_candidate})
                last_progress = state.progress
            if state.status != last_status:
                yield _sse_event("status", {
                    "status": state.status,
                    "error": state.error,
                    "summary": state.summary.to_dict() if state.summary else None,
                })
                last_status = state.status
                if state.status in ("done", "error"):
                    # 推最终结果
                    yield _sse_event("result", {
                        "summary": state.summary.to_dict() if state.summary else None,
                        "ranking": state.result.ranking if state.result else [],
                        "candidates": [c.model_dump(exclude_none=True) for c in state.result.candidates] if state.result else [],
                    })
                    yield _sse_event("end", {"status": state.status})
                    break
            time.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


def _sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


# ============================================================
# 直接查询结果 (不走 SSE)
# ============================================================
@app.get("/api/batch/{job_id}/result")
def get_batch_result(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, f"job {job_id} not found")
    state = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": state.status,
        "summary": state.summary.to_dict() if state.summary else None,
        "ranking": state.result.ranking if state.result else [],
        "candidates": [c.model_dump(exclude_none=True) for c in state.result.candidates] if state.result else [],
        "log": state.log,
        "error": state.error,
    }


# ============================================================
# HITL 端点
# ============================================================
class HRDecisionRequest(BaseModel):
    candidate_name: str
    job_title: str
    adjusted_score: int | None = None
    recommendation: str | None = None
    note: str | None = None


@app.get("/api/hitl/pending")
def hitl_pending(job_title: str | None = None) -> list[dict]:
    return list_pending_reviews(job_title=job_title)


@app.post("/api/hitl/decide")
def hitl_decide(req: HRDecisionRequest) -> dict:
    review = submit_hr_decision(
        candidate_name=req.candidate_name,
        job_title=req.job_title,
        adjusted_score=req.adjusted_score,
        recommendation=req.recommendation,
        note=req.note,
    )
    return {
        "candidate_name": review.candidate_name,
        "job_title": review.job_title,
        "original_score": review.original_score,
        "hr_adjusted_score": review.hr_adjusted_score,
        "original_recommendation": review.original_recommendation,
        "hr_recommendation": review.hr_recommendation,
        "hr_note": review.hr_note,
        "reviewed_at": review.reviewed_at,
    }
