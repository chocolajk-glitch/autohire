"""启动后端 - 用 subprocess.Popen 避免被父进程 kill 链影响."""
import os
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 现在 cwd = backend/
log_path = os.path.join(os.pardir, "data", "uvicorn.log")
err_path = os.path.join(os.pardir, "data", "uvicorn.err.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
out = open(log_path, "ab", buffering=0)  # append, unbuffered
err = open(err_path, "ab", buffering=0)
subprocess.Popen(
    [r"D:\Code\multi-agent\autohire\.venv\Scripts\python.exe", "-m", "uvicorn", "api.server:app",
     "--host", "127.0.0.1", "--port", "8765", "--log-level", "info"],
    stdout=out, stderr=err,
    creationflags=0,
)
print("backend started")

