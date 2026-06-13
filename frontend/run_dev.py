"""启动 Vite dev server - 独立工作目录调用."""
import os
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 现在 cwd = frontend/
log_path = os.path.join(os.pardir, "data", "vite.log")
err_path = os.path.join(os.pardir, "data", "vite.err.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)
out = open(log_path, "ab", buffering=0)
err = open(err_path, "ab", buffering=0)
subprocess.Popen(
    [os.path.join("node_modules", ".bin", "vite.cmd")],
    stdout=out, stderr=err,
    creationflags=0,
)
print("vite started")

