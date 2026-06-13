"""SSE 客户端测试 - 监听 batch 进度, 打印事件."""
import json
import sys
import time
from urllib.parse import urlencode
import httpx


def listen_sse(job_id: str, base_url: str = "http://127.0.0.1:8765") -> None:
    url = f"{base_url}/api/batch/{job_id}/stream"
    print(f"connecting to {url} ...")
    with httpx.stream("GET", url, timeout=600) as r:
        r.raise_for_status()
        event_type = ""
        for line in r.iter_lines():
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                try:
                    data = json.loads(data_str)
                except Exception:
                    data = data_str
                print(f"[{time.strftime('%H:%M:%S')}] event={event_type}: {json.dumps(data, ensure_ascii=False, default=str)[:300]}")
                if event_type == "end":
                    break


if __name__ == "__main__":
    job_id = sys.argv[1] if len(sys.argv) > 1 else None
    if not job_id:
        print("usage: python sse_client.py <job_id>")
        sys.exit(1)
    listen_sse(job_id)
