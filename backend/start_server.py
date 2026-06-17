"""启动后端 - 用 subprocess.Popen 避免被父进程 kill 链影响."""
import os
import subprocess

BACKEND_ROOT = os.path.dirname(os.path.abspath(__file__))
log_path = os.path.join(os.pardir, "data", "uvicorn.log")
err_path = os.path.join(os.pardir, "data", "uvicorn.err.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
out = open(log_path, "ab", buffering=0)
err = open(err_path, "ab", buffering=0)
# 显式传 cwd=BACKEND_ROOT, 让子进程能找到 backend/.env
subprocess.Popen(
    [r"D:\Code\multi-agent\autohire\.venv\Scripts\python.exe", "-m", "uvicorn", "api.server:app",
     "--host", "127.0.0.1", "--port", "8765", "--log-level", "info"],
    stdout=out, stderr=err,
    cwd=BACKEND_ROOT,  # 关键: 让 load_dotenv 找到 backend/.env
    creationflags=0,
)
print("backend started")

