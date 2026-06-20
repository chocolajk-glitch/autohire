"""MCP 子进程 supervisor - 负责启动/监控/重启 N 个 MCP HTTP 子进程.

被主进程 import 调用, 在主进程里 spawn 一组 MCP 子进程, 暴露一个 Pool 接口给 client 用.

设计要点:
- 启动时: spawn N 个子进程 (transport=http, 端口 9001..9001+N-1)
- 等所有端口 listen (wait_for_port)
- 提供 get_session() 给 MCPClientPool 用
- 后台线程每 5s health check, 挂了 respawn (lazy: 下次 call 时才补)
- atexit: graceful shutdown (SIGTERM -> wait 5s -> SIGKILL)

不做的事:
- 不管 stdio 模式 (单 client 不需要 supervisor)
- 不做负载均衡 (由 MCPClientPool 的 round-robin 做)
"""
from __future__ import annotations

import atexit
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = Path(sys.executable)
SERVER_SCRIPT = BACKEND_ROOT / "mcp_servers" / "resume_server.py"


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    """等指定端口开始 listen. 返回 True 表示就绪."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            time.sleep(0.1)
    return False


@dataclass
class WorkerProcess:
    """单个 MCP 子进程状态."""
    host: str
    port: int
    proc: subprocess.Popen | None = None
    healthy: bool = False
    last_health_check: float = 0.0
    restart_count: int = 0


@dataclass
class MCPPoolSupervisor:
    """N 个 MCP HTTP 子进程的 supervisor."""
    host: str = "127.0.0.1"
    base_port: int = 9001
    pool_size: int = 3
    health_check_interval: float = 5.0
    startup_timeout: float = 15.0
    workers: list[WorkerProcess] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _health_thread: threading.Thread | None = None
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _started: bool = False

    def __post_init__(self) -> None:
        # 从环境变量读配置 (允许覆盖默认值)
        self.host = os.getenv("MCP_HTTP_HOST", self.host)
        self.base_port = int(os.getenv("MCP_HTTP_BASE_PORT", str(self.base_port)))
        self.pool_size = int(os.getenv("MCP_HTTP_POOL_SIZE", str(self.pool_size)))

    def start(self) -> None:
        """启动所有 worker + 后台 health check 线程."""
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            logger.info(
                "supervisor: starting %d MCP workers on %s:%d-%d",
                self.pool_size, self.host, self.base_port, self.base_port + self.pool_size - 1,
            )
            for i in range(self.pool_size):
                w = WorkerProcess(host=self.host, port=self.base_port + i)
                self._spawn(w)
                self.workers.append(w)

            # 等所有端口就绪
            for w in self.workers:
                ok = wait_for_port(w.host, w.port, timeout=self.startup_timeout)
                w.healthy = ok
                if not ok:
                    logger.error("supervisor: worker on port %d failed to start", w.port)
                else:
                    logger.info("supervisor: worker on port %d ready", w.port)

            # 后台 health check 线程
            self._stop_event.clear()
            self._health_thread = threading.Thread(
                target=self._health_loop, daemon=True, name="mcp-pool-health",
            )
            self._health_thread.start()
            self._started = True
            atexit.register(self.shutdown)

    def _spawn(self, w: WorkerProcess) -> None:
        """spawn 一个 MCP 子进程 (HTTP mode)."""
        # Windows 下要 CREATE_NEW_PROCESS_GROUP, 否则 SIGTERM 杀不掉
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

        cmd = [
            str(PYTHON_EXE), "-m", "mcp_servers.resume_server",
            "--transport", "http", "--host", w.host, "--port", str(w.port),
        ]
        log_dir = BACKEND_ROOT / "data" / "mcp_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"worker_{w.port}.log"

        w.proc = subprocess.Popen(
            cmd,
            cwd=str(BACKEND_ROOT),
            stdout=open(log_path, "ab", buffering=0),
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        w.last_health_check = time.time()
        logger.info("supervisor: spawned MCP worker pid=%d on port %d", w.proc.pid, w.port)

    def _health_loop(self) -> None:
        """后台线程: 定期检查所有 worker, 挂了 respawn."""
        while not self._stop_event.wait(self.health_check_interval):
            with self._lock:
                for w in self.workers:
                    if not w.healthy:
                        # lazy restart: 不立即起, 等下次 call 时再起
                        continue
                    # 用端口探测代替 HTTP ping (FastMCP 没有标准 health endpoint)
                    try:
                        with socket.create_connection((w.host, w.port), timeout=2.0):
                            w.healthy = True
                            w.last_health_check = time.time()
                    except (ConnectionRefusedError, socket.timeout, OSError):
                        logger.warning("supervisor: worker on port %d unhealthy, marking", w.port)
                        w.healthy = False

    def get_healthy_workers(self) -> list[WorkerProcess]:
        """获取当前健康的 worker 列表 (调用方负责加锁或快照)."""
        return [w for w in self.workers if w.healthy]

    def ensure_worker_healthy(self, w: WorkerProcess) -> bool:
        """确保指定 worker 健康 (不健康就 respawn). 线程安全."""
        with self._lock:
            if w.healthy:
                return True
            # 检查进程是否还活着
            if w.proc and w.proc.poll() is None:
                # 进程在跑但端口不通? 等一下
                logger.warning("supervisor: worker on port %d proc alive but port dead", w.port)
            else:
                logger.info("supervisor: respawning worker on port %d (was pid=%s)",
                            w.port, w.proc.pid if w.proc else "n/a")
                # 老进程死透了, 起新的
                try:
                    if w.proc and w.proc.poll() is None:
                        w.proc.terminate()
                        w.proc.wait(timeout=3)
                except Exception:
                    pass
                self._spawn(w)
                w.restart_count += 1
            # 等端口 ready
            ok = wait_for_port(w.host, w.port, timeout=self.startup_timeout)
            w.healthy = ok
            return ok

    def shutdown(self) -> None:
        """graceful shutdown 所有 worker."""
        with self._lock:
            if not self._started:
                return
            logger.info("supervisor: shutting down %d workers", len(self.workers))
            self._stop_event.set()
            for w in self.workers:
                if w.proc and w.proc.poll() is None:
                    try:
                        w.proc.terminate()
                        w.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning("supervisor: worker pid=%d did not exit, killing", w.proc.pid)
                        w.proc.kill()
                    except Exception as e:
                        logger.warning("supervisor: error shutting down worker: %s", e)
            self._started = False
            logger.info("supervisor: all workers stopped")


# ========== 单例 ==========
_supervisor: MCPPoolSupervisor | None = None
_supervisor_lock = threading.Lock()


def get_supervisor() -> MCPPoolSupervisor:
    """获取全局 supervisor 单例 (懒加载)."""
    global _supervisor
    with _supervisor_lock:
        if _supervisor is None:
            _supervisor = MCPPoolSupervisor()
            _supervisor.start()
    return _supervisor


def shutdown_supervisor() -> None:
    """测试用: 关闭 supervisor."""
    global _supervisor
    with _supervisor_lock:
        if _supervisor is not None:
            _supervisor.shutdown()
            _supervisor = None