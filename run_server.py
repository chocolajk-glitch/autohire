"""启动 uvicorn 的小脚本 - 独立工作目录调用, 避免父进程被 opencode 误杀."""
import os
import sys
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([
    sys.executable, "-m", "uvicorn", "api.server:app",
    "--host", "127.0.0.1", "--port", "8765", "--log-level", "info",
])
