#!/usr/bin/env python3
"""查询本地 Prometheus (http://127.0.0.1:9092) 的实时数据。

关键点:本机环境设置了 http_proxy，直连 127.0.0.1 会被代理拦截返回 502，
所以这里显式禁用代理 (trust_env=False / proxies 置空)。
"""
import argparse
import json
import time
from typing import Any, Optional

import requests

PROM_URL = "http://127.0.0.1:9092"


class PrometheusClient:
    def __init__(self, base_url: str = PROM_URL, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        # 绕过代理，避免访问 127.0.0.1 时被 http_proxy 拦截
        self.session.trust_env = False
        self.session.proxies = {"http": None, "https": None}

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = self.session.get(
            f"{self.base_url}{path}", params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError(f"query failed: {data}")
        return data["data"]

    def healthy(self) -> bool:
        resp = self.session.get(f"{self.base_url}/-/healthy", timeout=self.timeout)
        return resp.status_code == 200

    def query(self, promql: str, when: Optional[float] = None) -> Any:
        """瞬时查询 (instant query)。"""
        params = {"query": promql}
        if when is not None:
            params["time"] = when
        return self._get("/api/v1/query", params)

    def query_all(self, when: Optional[float] = None) -> Any:
        """获取当前时刻(或指定时刻)所有指标的全部时间序列。"""
        return self.query('{__name__=~".+"}', when=when)

    def query_range(
        self, promql: str, start: float, end: float, step: str = "15s"
    ) -> Any:
        """区间查询 (range query)。"""
        params = {"query": promql, "start": start, "end": end, "step": step}
        return self._get("/api/v1/query_range", params)

    def targets(self, state: str = "active") -> Any:
        return self._get("/api/v1/targets", {"state": state})

    def label_values(self, label: str = "__name__") -> Any:
        """列出某个 label 的所有取值，比如所有指标名。"""
        return self._get(f"/api/v1/label/{label}/values")


def _series_to_str(item: dict) -> tuple:
    """把一条 vector 结果拆成 (指标名, labels字符串, 值, 时间戳)。"""
    metric = dict(item["metric"])
    name = metric.pop("__name__", "")
    labels = ",".join(f'{k}="{v}"' for k, v in sorted(metric.items()))
    ts, val = item["value"]
    return name, labels, val, ts


def _print_instant(result: list) -> None:
    if not result:
        print("  (无数据)")
        return
    for item in result:
        name, labels, val, _ = _series_to_str(item)
        print(f"  {name}{{{labels}}} = {val}")


def _dump_json(result: list, path: str, ts: float) -> None:
    payload = {"timestamp": ts, "count": len(result), "series": result}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[已保存] {len(result)} 条序列 -> {path}")


def _dump_csv(result: list, path: str) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "labels", "value", "timestamp"])
        for item in result:
            name, labels, val, ts = _series_to_str(item)
            writer.writerow([name, labels, val, ts])
    print(f"[已保存] {len(result)} 条序列 -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="查询本地 Prometheus 实时数据")
    parser.add_argument("--url", default=PROM_URL, help="Prometheus 地址")
    parser.add_argument("-q", "--query", default="up", help="PromQL 查询语句")
    parser.add_argument("--all", action="store_true", help="获取当前时刻所有指标数据")
    parser.add_argument("--range", action="store_true", help="使用区间查询(最近5分钟)")
    parser.add_argument("--step", default="15s", help="区间查询步长")
    parser.add_argument("-o", "--output", help="导出文件路径 (.json 或 .csv)")
    parser.add_argument("--raw", action="store_true", help="打印原始 JSON")
    args = parser.parse_args()

    client = PrometheusClient(args.url)

    print(f"[健康检查] {'OK' if client.healthy() else '异常'}  ({args.url})")

    # 获取当前时刻全部数据
    if args.all:
        data = client.query_all()
        result = data["result"]
        print(f"\n[全部指标] 当前时刻共 {len(result)} 条时间序列")
        if args.output:
            ts = time.time()
            if args.output.endswith(".csv"):
                _dump_csv(result, args.output)
            else:
                _dump_json(result, args.output, ts)
        elif args.raw:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            _print_instant(result)
        return

    if args.range:
        end = time.time()
        start = end - 300
        data = client.query_range(args.query, start, end, args.step)
    else:
        data = client.query(args.query)

    if args.raw:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return

    print(f"\n[查询] {args.query}")
    if data.get("resultType") == "vector":
        _print_instant(data["result"])
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
