# E2B Sandbox 高并发压测报告

- **日期**: 2026-03-02
- **E2B Domain**: ap-beijing.tencentags.com
- **代理**: http://whitelist-proxy.cybertron.svc.cluster.local:7891
- **并发梯度**: 128 → 256 → 512 → 1024 → 2048 → 4096
- **每个 sandbox 操作**: create → run_code(`print(42)`) → kill
- **超时设置**: create 30s, exec 25s, kill 15s, 单轮总超时 180s

---

## 1. 压测结果

### N=128, concurrency=128

```
wall=31.4s  tput=4.1/s
create: 125/128 ok | p50=2.26s p90=4.14s p99=7.85s max=10.20s | errs={'ConnectError': 2, 'TimeoutError': 1}
exec:   125/125 ok | p50=1.19s p90=2.72s p99=4.38s max=7.34s
kill:   123/125 ok | p50=0.48s p90=1.64s p99=3.55s max=7.50s | errs={'ConnectError': 2}
```

### N=256, concurrency=256

```
wall=18.2s  tput=14.1/s
create: 254/256 ok | p50=3.72s p90=5.62s p99=8.35s max=10.12s | errs={'ConnectError': 2}
exec:   254/254 ok | p50=2.21s p90=3.96s p99=5.26s max=6.19s
kill:   252/254 ok | p50=2.03s p90=3.27s p99=5.56s max=10.21s | errs={'ConnectError': 2}
```

### N=512, concurrency=512

```
wall=37.0s  tput=13.8/s
create: 501/512 ok | p50=6.34s p90=9.50s p99=13.77s max=17.37s | errs={'ConnectError': 8, 'TimeoutError': 3}
exec:   498/501 ok | p50=5.46s p90=7.93s p99=9.82s max=11.08s | errs={'ConnectError': 2, 'TimeoutError': 1}
kill:   497/501 ok | p50=4.83s p90=7.58s p99=9.60s max=13.65s | errs={'ConnectError': 3, 'TimeoutError': 1}
```

### N=1024, concurrency=1024

```
wall=47.9s  tput=21.4/s
create:  998/1024 ok | p50=11.57s p90=18.54s p99=23.34s max=31.02s | errs={'ConnectError': 23, 'TimeoutError': 3}
exec:    993/998  ok | p50=11.55s p90=17.35s p99=18.55s max=18.96s | errs={'ConnectError': 5}
kill:    780/998  ok | p50=8.38s  p90=14.01s p99=15.09s max=15.32s | errs={'ConnectError': 7, 'TimeoutError': 211}
```

### N=2048, concurrency=2048

```
wall=84.5s  tput=24.2/s
create: 1522/2048 ok | p50=17.71s p90=27.26s p99=29.95s max=36.47s | errs={'TimeoutError': 512, 'ConnectError': 14}
exec:   1220/1522 ok | p50=15.39s p90=24.31s p99=26.39s max=27.09s | errs={'ConnectError': 6, 'TimeoutError': 296}
kill:    864/1522 ok | p50=9.84s  p90=20.46s p99=28.14s max=36.76s | errs={'TimeoutError': 651, 'ConnectError': 7}
```

### N=4096, concurrency=4096

```
wall=126.1s  tput=32.5/s
create: 1405/4096 ok | p50=19.35s p90=28.41s p99=30.11s max=33.39s | errs={'TimeoutError': 2691}
exec:   1278/1405 ok | p50=15.50s p90=24.11s p99=25.86s max=26.61s | errs={'ConnectError': 3, 'TimeoutError': 124}
kill:    914/1405 ok | p50=9.99s  p90=21.79s p99=25.04s max=42.26s | errs={'TimeoutError': 486, 'ConnectError': 5}
```

---

## 2. 汇总表

| 并发数 | Create 成功率 | Create p50 | Create p90 | Create max | Exec 成功率 | Kill 成功率 | 吞吐量 (sandbox/s) |
|-------:|:------------:|:----------:|:----------:|:----------:|:-----------:|:-----------:|:------------------:|
| 128 | 97.7% (125/128) | 2.26s | 4.14s | 10.20s | 100% (125/125) | 98.4% (123/125) | 4.1 |
| 256 | 99.2% (254/256) | 3.72s | 5.62s | 10.12s | 100% (254/254) | 99.2% (252/254) | 14.1 |
| 512 | 97.9% (501/512) | 6.34s | 9.50s | 17.37s | 99.4% (498/501) | 99.2% (497/501) | 13.8 |
| 1024 | 97.5% (998/1024) | 11.57s | 18.54s | 31.02s | 99.5% (993/998) | 78.2% (780/998) | 21.4 |
| 2048 | 74.3% (1522/2048) | 17.71s | 27.26s | 36.47s | 80.2% (1220/1522) | 56.8% (864/1522) | 24.2 |
| 4096 | 34.3% (1405/4096) | 19.35s | 28.41s | 33.39s | 90.9% (1278/1405) | 65.1% (914/1405) | 32.5 |

---

## 3. 分析与结论

### 3.1 稳定并发上限: ~512

- 在 128-512 并发范围内，create 成功率稳定在 97%+ ，exec 接近 100%，kill 在 98%+。
- 延迟合理：create p90 < 10s，exec p90 < 8s。
- 错误以零星 `ConnectError`（代理层网络抖动）为主，retry 可覆盖。

### 3.2 1024 并发: kill 开始退化

- Create 仍有 97.5% 成功率，exec 99.5%，但 kill 骤降至 78.2%（211 个 TimeoutError）。
- 原因：E2B API 的 kill 端点在高并发下响应变慢，15s 超时内无法完成。
- **影响有限**：kill 失败不影响训练正确性，sandbox 到期会自动回收。

### 3.3 2048+ 并发: 不可用

- Create 成功率 74.3%（2048）→ 34.3%（4096），主要错误为 `TimeoutError`。
- 瓶颈在代理连接池和 E2B API 的并发处理能力——大量请求排队导致超过 30s create 超时。
- exec 和 kill 也因连接争用大幅退化。

### 3.4 错误模式

| 并发范围 | 主要错误类型 | 根因 |
|---------|------------|------|
| ≤512 | `ConnectError` (零星) | 代理层偶发网络抖动 |
| 1024 | `ConnectError` + `TimeoutError` (kill) | 连接池压力 + API 响应变慢 |
| ≥2048 | `TimeoutError` (大量) | 请求排队超时，连接池饱和 |

### 3.5 对训练的建议

- 当前训练脚本 `--generate-e2b-max-concurrency 64`，远低于 512 稳定上限，**E2B 不会成为瓶颈**。
- 若未来 rollout batch size 增大，可安全提升至 **256-512** 并发。
- 建议保留 `create_sandbox` 的 3 次 retry 机制以应对零星 `ConnectError`。
- 超过 512 并发时需考虑：加大 create timeout、使用连接池限流、或部署多代理分流。

---

## 4. 压测脚本

```python
"""E2B high-concurrency stress test with unbuffered output and per-test timeout."""
import asyncio, os, time, sys
from dataclasses import dataclass, field

from e2b_code_interpreter import AsyncSandbox
import logging
logging.disable(logging.CRITICAL)

T = os.getenv("E2B_TEMPLATE")
K = os.getenv("E2B_API_KEY")
D = os.getenv("E2B_DOMAIN")
P = os.getenv("http_proxy")
print(f"E2B: domain={D}, key=...{K[-6:]}", flush=True)

@dataclass
class R:
    ok: bool; lat: float; err: str = ""

async def one(sem, creates, execs, kills):
    async with sem:
        sbx = None
        t0 = time.time()
        try:
            sbx = await asyncio.wait_for(
                AsyncSandbox.create(
                    template=T, api_key=K, domain=D, proxy=P, timeout=3600,
                ),
                timeout=30,
            )
            creates.append(R(True, time.time() - t0))
        except Exception as e:
            creates.append(R(False, time.time() - t0, f"{type(e).__name__}"))
            return

        t0 = time.time()
        try:
            await asyncio.wait_for(sbx.run_code("print(42)", timeout=20), timeout=25)
            execs.append(R(True, time.time() - t0))
        except Exception as e:
            execs.append(R(False, time.time() - t0, f"{type(e).__name__}"))

        t0 = time.time()
        try:
            await asyncio.wait_for(sbx.kill(), timeout=15)
            kills.append(R(True, time.time() - t0))
        except Exception as e:
            kills.append(R(False, time.time() - t0, f"{type(e).__name__}"))


def report(name, items):
    ok = [r for r in items if r.ok]
    fail = [r for r in items if not r.ok]
    n = len(ok)
    if n:
        l = sorted(r.lat for r in ok)
        s = (
            f"p50={l[n//2]:.2f}s p90={l[int(n*0.9)]:.2f}s "
            f"p99={l[min(int(n*0.99),n-1)]:.2f}s max={l[-1]:.2f}s"
        )
    else:
        s = "N/A"
    line = f"  {name}: {len(ok)}/{len(items)} ok | {s}"
    if fail:
        errs = {}
        for r in fail:
            errs[r.err] = errs.get(r.err, 0) + 1
        line += f" | errs={errs}"
    print(line, flush=True)


async def test(conc, count):
    creates, execs, kills = [], [], []
    sem = asyncio.Semaphore(conc)
    print(f"\n--- N={count} conc={conc} ---", flush=True)
    t0 = time.time()
    await asyncio.gather(*[one(sem, creates, execs, kills) for _ in range(count)])
    wall = time.time() - t0
    print(f"  wall={wall:.1f}s  tput={count/wall:.1f}/s", flush=True)
    report("create", creates)
    report("exec", execs)
    report("kill", kills)


async def main():
    levels = [
        (128, 128), (256, 256), (512, 512),
        (1024, 1024), (2048, 2048), (4096, 4096),
    ]
    for conc, count in levels:
        try:
            await asyncio.wait_for(test(conc, count), timeout=180)
        except asyncio.TimeoutError:
            print(f"  *** TEST TIMED OUT (180s) ***", flush=True)


asyncio.run(main())
```
