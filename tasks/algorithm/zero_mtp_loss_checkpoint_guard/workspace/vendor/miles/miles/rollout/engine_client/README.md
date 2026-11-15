# EngineClient —— token 级 partial rollout 网关

## 解决什么问题

sglang 在 `--pause-generation-mode abort` 下，`update_weights` 期间会打断正在
生成的请求，调用方原本会看到一个 `finish_reason=abort` 的半成品。
`EngineClient` 自动用"已生成 token 当新前缀"重新发起请求（resume），**调用方
永远只看到一个完整的逻辑 turn**，同时正确合并 MoE 场景下
`routed_experts`/`indexer_topk` 在多段续推之间的 replay 数据。

## 覆盖的两条链路

- **白盒**（`generate_hub/single_turn.py`/`multi_turn.py`）：请求经
  `transport.py` 的可插拔开关，打到 `/sessions/{id}/generate`。
- **黑盒**（`SessionCore.chat_completions`，harbor TITO）：`SessionServer`
  进程内直接共享同一个 `EngineClient` 实例，无额外网络跳。

两条链路复用同一个 `EngineClient` 续推内核，只是请求/响应的协议形状
（`generate` vs OpenAI `chat_completions`）不同。

## 代码结构

| 文件 | 职责 |
|---|---|
| `types.py` | `ReplaySlice`/`GenerationSegment`/`TurnRecord`，三个只读 dataclass |
| `replay.py` | `ReplayLedger`（先写者保留区间合并）+ dtype 自适应编解码 |
| `meta_info.py` | `parse_meta_info()`：按 `path` 解析 `generate`/`chat_completions` 两种响应形状 |
| `client.py` | `EngineClient`（abort→resend 续推循环）+ `create_control_router` + `create_engine_client_router` + `push_engine_client_stats` |
| `client_sdk.py` | `TurnRecord` → 响应渲染（`render_generate_response`/`render_chat_completions_response`） |
| `transport.py` | `load_engine_transport`/`create_engine_transport`：白盒/黑盒各自的传输层开关点 |
| `args.py` | CLI flag 注册 |

测试替身 `FakeEngine`/`ScriptedResponse` 在 `miles/utils/test_utils/fake_engine.py`
（生产代码树里的测试专用文件，和 `mock_sglang_server.py` 同一惯例）。

## 核心机制

- **续推循环**：`abort` → 累积 `input_ids`、递减剩余 `max_new_tokens` → 重发，
  直到 `stop`/`length` 或预算耗尽。`rollout_partial_mode=off`（默认）下
  `EngineClient` 退化成转发一次、原样返回的薄代理。
- **ReplayLedger 先写者保留**：`routed_experts`/`indexer_topk` 每次响应都是
  "从 0 开始的整段覆盖"，不是增量。一个 turn 续推 N 次就会收到 N+1 份互相
  重叠的数据——**更早提交的区间保留**，因为它对应未被打断的原始生成；
  后续续推虽然范围更宽，但引擎状态已经历 abort/resend 重建，不能假定和原始
  生成完全一致。区间之间有空洞时抛 `ReplayGapError`，不静默返回不完整数据。
- **单段透传**：`chat_completions` 路径下，若一个 turn 只有一个 segment
  （从未被 abort），直接原样返回该次真实引擎响应，不做 tokenizer 重新
  decode + tool_call 重新解析——这条重建路径只有多段拼接时才必要，且本身
  有精度近似（见下）。
- **header 过滤**：转发前去掉 `content-length`/`transfer-encoding`/`host`
  （`http_utils.filter_forwarded_headers`），`do_proxy` 和 `EngineClient`
  共享同一份实现。
- **max_tokens 字段名保留**：写回剩余预算时沿用调用方原始用的字段名
  （`max_tokens` 或 `max_completion_tokens`），不强制替换——保证未发生 abort
  的首次请求和 `do_proxy` 逐字节一致。
- **续推可观测性写回响应**：每个响应的 `meta_info` 都会附加
  `gateway_segments`/`gateway_resumed`/`weight_versions`/`gateway_processing_time`
  （字段名保留旧的 `gateway_*` 前缀，兼容既有 `Sample` 解析和 dashboard）——
  这是下游唯一能判断"这个 turn 是否真的被 abort 过"的途径。

## 配置开关

| Flag | 默认 | 作用 |
|---|---|---|
| `--rollout-gateway-transport` | `local` | 白盒是否走 EngineClient；`local` 下请求路径和改动前完全一致 |
| `--rollout-gateway-url` | — | `remote` 时 EngineClient 所在的地址 |
| `--rollout-partial-mode` | `off` | 是否真正续推。白盒仅在 `transport=remote` 时生效；黑盒进程内共享，不受 transport 开关影响 |
| `--engine-client-timeout` | 600.0 | EngineClient 自身请求的超时 |

## 黑盒接入要点

- `SessionServer.__init__` 构造共享 `EngineClient` 实例；`SessionCore.
  chat_completions` 按 `rollout_partial_mode` 分流到它或既有 `do_proxy`。
- `LinearTrajectory` 新增单调计数器 `generation`，替代 `num_assistant` 做
  Phase 3 的乐观并发检查——`num_assistant` 在"回滚后又长回同一个值"时会漏检
  竞态（ABA 问题），`generation` 不会。
- `EngineClientDrainingError` 在黑盒路径映射为 `EngineDrainingError`（503）。

## 已知限制

- 续推假设真实后端按 `input_ids` 生成而非重渲染 `messages`；
  `mock_sglang_server.py` 不满足这个假设，只有单测验证过，接真实后端前需
  单独确认。
- 黑盒 `token` 模式下，本地 tool_call 重新解析用 `skip_special_tokens=True`
  近似 `no_stop_trim=False` 的裁剪效果，只有 stop 序列是 special token 时
  精确等价。

## 指标监控

- `GET /control/stats`：`calls_total`/`segments_total`/`resumed_total`/
  `avg_segments_per_turn` 实时快照。
- 若训练侧打开 `--use-prometheus`，同一份数据以 `rollout/gateway/*`
  为 key 推送（未打开时 no-op，`_METRIC_PREFIX` 保留旧名以兼容既有面板）。
- `POST /control/drain`：优雅下线，之后的 abort 不再续推，直接失败。

## 测试

```bash
cd /user/huangyinhong/miles_partial_abort/miles
python3 -m pytest tests/fast/rollout/engine_client/ tests/fast/rollout/session/ -q
```
