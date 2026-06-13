"""FastAPI server 的 smoke 测试 - 不触发批量, 只验证端点注册和基础响应."""
from fastapi.testclient import TestClient

from api.server import app

client = TestClient(app)


class TestBasicEndpoints:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert body["app"] == "AutoHire"
        assert body["status"] == "ok"

    def test_list_jds(self):
        r = client.get("/api/jd")
        assert r.status_code == 200
        jds = r.json()
        assert isinstance(jds, list)
        # 我们生成了 3 个 JD
        assert "backend_python_jd" in jds
        assert "frontend_vue_jd" in jds
        assert "algo_recommendation_jd" in jds

    def test_list_resumes(self):
        r = client.get("/api/resumes")
        assert r.status_code == 200
        resumes = r.json()
        assert isinstance(resumes, list)
        assert len(resumes) >= 11  # 11 份

    def test_batch_requires_resume(self):
        r = client.post("/api/batch/run", json={
            "jd_filename": "backend_python_jd",
            "resume_filenames": [],
        })
        assert r.status_code == 400

    def test_batch_requires_jd(self):
        r = client.post("/api/batch/run", json={
            "resume_filenames": ["01_zhang_san_strong_backend.pdf"],
        })
        assert r.status_code == 400

    def test_stream_unknown_job(self):
        r = client.get("/api/batch/nonexistent_job/stream")
        assert r.status_code == 404

    def test_result_unknown_job(self):
        r = client.get("/api/batch/nonexistent_job/result")
        assert r.status_code == 404

    def test_hitl_pending(self):
        r = client.get("/api/hitl/pending")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_openapi_docs(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        assert "/" in paths
        assert "/api/batch/run" in paths
        assert "/api/batch/{job_id}/stream" in paths
