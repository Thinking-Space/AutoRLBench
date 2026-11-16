#!/usr/bin/env python3
"""用 uvicorn 暴露一个 /metrics 接口，返回 Prometheus 最新数据。

从本地 Prometheus (http://127.0.0.1:9092) 拉取当前时刻的所有时间序列，
并按 Prometheus 文本 exposition 格式(含 # HELP / # TYPE)返回。

上游地址 PROM_UPSTREAM 可通过两种方式配置(命令行优先):
    # 环境变量(uvicorn 方式也适用)
    PROM_UPSTREAM=http://127.0.0.1:9092 uvicorn metrics_server:app --port 9092
    # 命令行参数
    python3 metrics_server.py --port 9200 --upstream http://127.0.0.1:9090

访问:
    curl http://127.0.0.1:9090/metrics
"""
import argparse
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Response

from prometheus_query import PrometheusClient

# Prometheus 文本格式的 content-type
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# 上游 Prometheus 默认地址
DEFAULT_UPSTREAM = "http://127.0.0.1:9092"

# 本地 Prometheus 抓取时盖上的标签,重渲染时丢弃,避免被外层 Prometheus 再抓时
# 撞车重命名成 exported_job / exported_instance。
DROP_LABELS = frozenset({"job", "instance"})

app = FastAPI(title="prometheus-proxy", docs_url=None, redoc_url=None)

# 懒加载并缓存 client;当上游地址(环境变量)变化时自动重建
_client: Optional[PrometheusClient] = None
_client_url: Optional[str] = None


def get_client() -> PrometheusClient:
    """按当前 PROM_UPSTREAM 环境变量返回 client,地址变化时重建。"""
    global _client, _client_url
    url = os.environ.get("PROM_UPSTREAM", DEFAULT_UPSTREAM)
    if _client is None or _client_url != url:
        _client = PrometheusClient(url)
        _client_url = url
    return _client


def _escape_label_value(value: str) -> str:
    """转义 label 值:反斜杠、双引号、换行。"""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fetch_metadata() -> Dict[str, Dict[str, str]]:
    """获取每个指标的 TYPE/HELP,拿不到就返回空(不影响主流程)。"""
    try:
        meta = get_client()._get("/api/v1/metadata")
    except Exception:
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for name, entries in meta.items():
        if entries:
            out[name] = {
                "type": entries[0].get("type", "untyped"),
                "help": entries[0].get("help", ""),
            }
    return out


def render_exposition(result: List[dict], metadata: Dict[str, Dict[str, str]]) -> str:
    """把 instant query 结果渲染成 Prometheus 文本格式。"""
    # 按指标名分组,保证同名指标的 # HELP/# TYPE 只出现一次
    grouped: Dict[str, List[dict]] = {}
    for item in result:
        name = item["metric"].get("__name__", "")
        grouped.setdefault(name, []).append(item)

    lines: List[str] = []
    for name in sorted(grouped):
        meta = metadata.get(name)
        if meta:
            if meta.get("help"):
                help_text = meta["help"].replace("\\", "\\\\").replace("\n", "\\n")
                lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {meta.get('type', 'untyped')}")

        for item in grouped[name]:
            labels = {
                k: v
                for k, v in item["metric"].items()
                if k != "__name__" and k not in DROP_LABELS
            }
            label_str = ",".join(
                f'{k}="{_escape_label_value(str(v))}"'
                for k, v in sorted(labels.items())
            )
            _, val = item["value"]
            if label_str:
                lines.append(f"{name}{{{label_str}}} {val}")
            else:
                lines.append(f"{name} {val}")

    return "\n".join(lines) + "\n"


@app.get("/metrics")
def metrics() -> Response:
    try:
        data: Any = get_client().query_all()
    except Exception as exc:  # 上游 Prometheus 不可用等
        msg = f"# upstream error: {type(exc).__name__}: {exc}\n"
        return Response(content=msg, media_type=CONTENT_TYPE, status_code=503)
    result = data.get("result", [])
    metadata = _fetch_metadata()
    body = render_exposition(result, metadata)
    return Response(content=body, media_type=CONTENT_TYPE)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    cli = get_client()
    try:
        ok = cli.healthy()
    except Exception as exc:
        return {"status": "unhealthy", "upstream": cli.base_url, "error": str(exc)}
    return {"status": "ok" if ok else "unhealthy", "upstream": cli.base_url}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prometheus /metrics 代理服务")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=9200, help="监听端口")
    parser.add_argument(
        "--upstream",
        default=os.environ.get("PROM_UPSTREAM", DEFAULT_UPSTREAM),
        help="上游 Prometheus 地址 (也可用环境变量 PROM_UPSTREAM)",
    )
    args = parser.parse_args()

    # 命令行参数写回环境变量,供 get_client() 统一读取
    os.environ["PROM_UPSTREAM"] = args.upstream

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()