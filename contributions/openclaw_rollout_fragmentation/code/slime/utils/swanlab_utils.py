import logging
import re
from html import escape
from types import ModuleType

from .tracking_config import compute_config_for_logging, compute_secondary_config_for_logging

logger = logging.getLogger(__name__)

_SWANLAB_WRITABLE_PROJECT_PERMISSIONS = {"WRITE", "MANAGE"}


def _organized_metric_name(name: str) -> str:
    """Put OpenClaw/Advisor metrics into readable SwanLab subgroups.

    The reward code keeps its stable metric names for other backends. This
    projection is SwanLab-only, where slash-separated names form the visible
    hierarchy in the metric browser.
    """
    if name.startswith("openclaw/cumulative_"):
        return "openclaw/cumulative/" + name.removeprefix("openclaw/cumulative_")

    if name.startswith("openclaw/advisor_calls_"):
        match = re.fullmatch(r"openclaw/advisor_calls_(0|1|2|3plus)_(.+)", name)
        if match:
            bucket, suffix = match.groups()
            suffix = {
                "rollouts": "rollouts",
                "generated_tokens_mean": "generated_tokens/mean",
                "trained_response_tokens_mean": "trained_response_tokens/mean",
                "max_context_tokens_mean": "max_context_tokens/mean",
            }.get(suffix, suffix)
            return f"openclaw/advisor_usage/calls_{bucket}/{suffix}"

    if name.startswith("openclaw/episode_generated_tokens_"):
        return "openclaw/tokens/generated/" + name.removeprefix("openclaw/episode_generated_tokens_")
    if name.startswith("openclaw/episode_elapsed_sec_"):
        return "openclaw/perf/episode_elapsed_sec/" + name.removeprefix("openclaw/episode_elapsed_sec_")
    if name.startswith("openclaw/episode_max_context_tokens_"):
        return "openclaw/tokens/max_context/" + name.removeprefix("openclaw/episode_max_context_tokens_")
    if name.startswith("openclaw/episode_trained_response_tokens_"):
        return "openclaw/tokens/trained_response/" + name.removeprefix("openclaw/episode_trained_response_tokens_")
    if name == "openclaw/episode_response_region_tokens_mean":
        return "openclaw/tokens/response_region/mean"
    if name == "openclaw/episode_trained_token_ratio":
        return "openclaw/tokens/trained_fraction"
    if name == "openclaw/episode_replay_inflation_ratio":
        return "openclaw/trajectory/replay_inflation_ratio"
    if name.startswith("openclaw/episode_reward_"):
        return "openclaw/reward/" + name.removeprefix("openclaw/episode_reward_")
    if name.startswith("openclaw/agent_turns_"):
        return "openclaw/behavior/turns_" + name.removeprefix("openclaw/agent_turns_")
    if name.startswith("openclaw/agent_tool_calls_"):
        return "openclaw/behavior/tool_calls_" + name.removeprefix("openclaw/agent_tool_calls_")
    if name in {"openclaw/fragments_per_rollout", "openclaw/forked_rollout_rate"}:
        return "openclaw/trajectory/" + name.removeprefix("openclaw/")
    if name.startswith("openclaw/dangling_"):
        return "openclaw/trajectory/dangling/" + name.removeprefix("openclaw/dangling_")
    if name.startswith("openclaw/"):
        health_prefixes = (
            "terminal_rollouts",
            "excluded_rollouts",
            "excluded_rollout_rate",
            "rollout_aborts",
            "rollout_abort_rate",
            "agent_truncations",
            "agent_truncation_rate",
            "agent_process_failures",
            "agent_process_failure_rate",
            "verifier_failures",
            "verifier_failure_rate",
        )
        suffix = name.removeprefix("openclaw/")
        if suffix in health_prefixes:
            return "openclaw/health/" + suffix

    if name.startswith("advisor/cumulative_cloud_failure_status/"):
        return "advisor/cumulative/reliability/failure_status/" + name.removeprefix(
            "advisor/cumulative_cloud_failure_status/"
        )
    if name.startswith("advisor/cloud_failure_status/"):
        return "advisor/reliability/failure_status/" + name.removeprefix("advisor/cloud_failure_status/")
    if name.startswith("advisor/cumulative_"):
        return "advisor/cumulative/" + name.removeprefix("advisor/cumulative_")
    if name.startswith("advisor/"):
        suffix = name.removeprefix("advisor/")
        if suffix.startswith(("cloud_rollouts", "cloud_rollout_rate", "cloud_calls")) or suffix == "rollouts":
            return "advisor/usage/" + suffix
        if suffix.startswith(("cloud_tokens", "cloud_cost_rmb", "full_cloud_reference_cost_rmb", "cost_")):
            return "advisor/cost/" + suffix
        if suffix.startswith(
            ("cloud_observed_calls", "cloud_success", "cloud_failed_calls", "cloud_attempts", "cloud_retry")
        ):
            return "advisor/reliability/" + suffix
        if suffix.startswith("cloud_latency"):
            return "advisor/performance/" + suffix
        if suffix.startswith(("cloud_response", "cloud_responses")):
            return "advisor/trace/" + suffix.removeprefix("cloud_")
        if suffix.startswith(("committed_tool_result_calls", "tool_return_without_committed_result")):
            return "advisor/transcript/" + suffix
        if suffix.startswith("tool_"):
            return "advisor/tool/" + suffix.removeprefix("tool_")
        if suffix.startswith("pending_tool_calls"):
            return "advisor/tool/" + suffix
        if suffix.startswith("trace_"):
            return "advisor/trace/" + suffix.removeprefix("trace_")
        if suffix.startswith("transcript_"):
            return "advisor/transcript/" + suffix.removeprefix("transcript_")
        if suffix.startswith("followup_model_"):
            return "advisor/model/followup_" + suffix.removeprefix("followup_model_")
        if suffix.startswith("model_"):
            return "advisor/model/" + suffix.removeprefix("model_")
        if suffix.startswith(("task_advantage", "cost_advantage", "total_advantage")):
            return "advisor/reward/" + suffix

    return name


def _organize_metrics_for_swanlab(metrics: dict) -> dict:
    organized = {}
    for name, value in metrics.items():
        organized_name = _organized_metric_name(name)
        if organized_name in organized and organized_name != name:
            raise RuntimeError(f"SwanLab metric name collision after organization: {organized_name!r}")
        organized[organized_name] = value
    return organized


def _metric_guide_sections() -> dict[str, tuple[str, list[tuple[str, str]]]]:
    """Metric dictionary grouped exactly like the SwanLab metric namespaces."""
    return {
        "train": (
            "train/*",
            [
                ("train/step", "Actor optimizer step；train 曲线横轴。"),
                ("train/loss", "最终优化目标：policy loss、entropy bonus 和可选 KL penalty 的组合。"),
                ("train/pg_loss", "Policy-gradient loss；已经按 rollout/sample 聚合。"),
                ("train/entropy_loss", "策略输出 entropy 的均值；entropy_coef 非零时作为探索奖励进入总 loss。"),
                ("train/pg_clipfrac", "Importance ratio 触发 PPO/GRPO clipping 的训练 token 比例。"),
                ("train/ppo_kl", "当前 actor 与 rollout policy 在训练 token 上的近似 KL。"),
                ("train/kl_loss", "当前 actor 与 reference model 的 KL penalty；仅 use_kl_loss 开启时上报。"),
                ("train/train_rollout_logprob_abs_diff", "训练重算 logprob 与 rollout 时 logprob 的平均绝对差，用于检查 train/infer mismatch。"),
                ("train/grad_norm", "当前 optimizer step 的全局梯度范数。"),
                ("train/lr-pg_<N>", "Optimizer 第 N 个 parameter group 的当前学习率。"),
                ("train/global_batch_size", "当前 optimizer step 实际包含的完整 rollout 数；异步/不均匀切分时可能变化。"),
                ("train/mtp_loss", "Multi-token prediction 辅助 loss；仅 MTP 训练开启时上报。"),
                ("train/ois", "Off-policy importance sampling ratio 的聚合值；仅 mismatch/TIS 路径开启时上报。"),
                ("train/<TIS或mismatch指标>", "TIS、rejection sampling 或 train/infer mismatch 路径返回的附加训练指标。"),
                ("train/<role>-*", "非 actor 角色使用 role 前缀，例如 critic-grad_norm；含义与同名 actor 指标一致。"),
            ],
        ),
        "rollout": (
            "rollout/*",
            [
                ("rollout/step", "Rollout 逻辑步；rollout、perf、OpenClaw 和 advisor 曲线横轴。"),
                ("rollout/rewards", "进入训练的最终 reward 均值。"),
                ("rollout/raw_reward", "Reward shaping 前的原始任务 reward 均值。"),
                ("rollout/response_lengths", "训练 Sample/fragment 的 response region token 数均值。"),
                ("rollout/total_lengths", "训练 Sample/fragment 的 prompt + response 总 token 数均值；用于 packing、DP 调度和显存负载。"),
                ("rollout/truncated", "训练数据中 truncated 标记的均值，即截断比例。"),
                ("rollout/round_number", "通用多轮 Sample 的轮数均值。"),
                ("rollout/log_probs", "Actor 在训练侧重算的 response token logprob 聚合值。"),
                ("rollout/ref_log_probs", "Reference model 的 response token logprob 聚合值。"),
                ("rollout/rollout_log_probs", "Rollout/SGLang 生成时保存的 response token logprob 聚合值。"),
                ("rollout/entropy", "Actor response token 的 entropy 聚合值。"),
                ("rollout/kl", "Actor 与 reference model 的 token-level KL 聚合值。"),
                ("rollout/returns", "训练 Sample 的 return 聚合值。"),
                ("rollout/advantages", "训练 Sample 的 advantage 聚合值。"),
                ("rollout/values", "Critic value 预测聚合值；仅 critic/PPO 路径上报。"),
                ("rollout/teacher_log_probs", "Teacher model logprob；仅蒸馏路径上报。"),
                ("rollout/opd_reverse_kl", "On-policy distillation 的 reverse KL；仅 OPD 路径上报。"),
                ("rollout/response_len/{mean,median,min,max}", "当前 batch 的有效 response 长度统计，即 loss_mask=1 token；Sample/fragment 级。"),
                ("rollout/repetition_frac", "Response 被重复检测规则判为重复的 Sample 比例。"),
                ("rollout/truncated_ratio", "状态为 TRUNCATED 的 Sample 比例。"),
                ("rollout/zero_std/count_<reward>", "同一 prompt group 内 reward 完全相同的组数，按共同 reward 分桶。"),
                ("rollout/spec_accept_rate", "Speculative decoding draft token 接受率；启用 speculative decoding 时上报。"),
                ("rollout/spec_accept_length", "Speculative decoding 每次接受的平均 token 数。"),
                ("rollout/prefix_cache_hit_rate", "Prompt token 的 prefix-cache 命中比例。"),
                ("rollout/avg_cached_tokens_per_sample", "每个 Sample 平均命中的 prefix-cache token 数。"),
                ("rollout/top_p_kept_vocab_per_token", "每个训练 token 在 top-p nucleus 中平均保留的候选词表大小。"),
                ("rollout/error_cat/<类别>", "按配置的 reward error category 统计 Sample 比例。"),
                ("rollout/dynamic_filter/drop_<原因>", "动态过滤按原因丢弃的 Sample 数。"),
                ("multi_turn/*", "通用多轮 rollout 的 raw/observed response 长度和 round number 统计。"),
                ("passrate/pass@<K>", "按 prompt group 估计的 pass@K。"),
                ("eval/*", "评测数据集 reward、response 长度、截断比例和 pass@K。"),
            ],
        ),
        "perf": (
            "perf/*",
            [
                ("perf/rollout_time", "当前 rollout step 收齐目标样本所用墙钟时间（秒）；不是单条 episode 耗时。"),
                ("perf/non_generation_time/{mean,median,min,max}", "Sample 中工具执行、环境等待等非模型生成耗时统计（秒）。"),
                ("perf/tokens_per_gpu_per_sec", "原始 response token 的 rollout 吞吐（token/GPU/秒）。"),
                ("perf/effective_tokens_per_gpu_per_sec", "loss_mask=1 token 的 rollout 吞吐（token/GPU/秒）。"),
                ("perf/longest_sample_tokens_per_sec", "当前 batch 最长原始 response Sample 的等效生成速度。"),
                ("perf/longest_effective_sample_tokens_per_sec", "当前 batch 最长有效 response Sample 的等效生成速度。"),
                ("perf/longest_{effective_}sample_non_generation_time", "最长 Sample 的平均非生成耗时（秒）。"),
                ("perf/longest_{effective_}sample_tokens_per_sec_without_non_generation", "扣除最长 Sample 非生成耗时后的生成速度。"),
                ("perf/request/e2e_latency/{mean,median,min,max}", "SGLang 单次 generate 请求端到端延迟统计。"),
                ("perf/request/queue_time/{mean,median,min,max}", "SGLang 单次 generate 请求排队时间统计。"),
                ("perf/decode/throughput/{mean,median,min,max}", "SGLang decode throughput 统计。"),
                ("perf/prefill/*/{mean,median,min,max}", "PD prefill 的排队、分配、forward、传输耗时/速率统计。"),
                ("perf/decode/*/{mean,median,min,max}", "PD decode 的 prealloc、bootstrap、分配、传输和 forward 耗时统计。"),
                ("perf/actor_train_time", "Actor 完成当前训练 step 的墙钟时间（秒）。"),
                ("perf/log_probs_time", "Actor 重算 rollout logprob 的墙钟时间（秒）。"),
                ("perf/ref_log_probs_time", "Reference model 计算 logprob 的墙钟时间（秒）。"),
                ("perf/update_weights_time", "将最新 actor 权重同步到 rollout engine 的时间（秒）。"),
                ("perf/train_wait_time", "训练侧等待 rollout 数据或同步条件的时间（秒）。"),
                ("perf/train_time", "训练侧当前 step 总计时（秒）。"),
                ("perf/step_time", "train_wait_time + train_time（秒）。"),
                ("perf/wait_time_ratio", "train_wait_time / step_time；越高说明训练侧等待越多。"),
                ("perf/actor_train_tok_per_s", "Actor 训练吞吐（训练 token/秒）。"),
                ("perf/{actor_train,log_probs,ref_log_probs}_tflops", "对应阶段估算计算吞吐（TFLOPS）。"),
                ("perf/update_weights_{density,wire_bytes,flushes_per_rank,...}", "Delta weight sync 的密度、传输字节数、flush 次数和压缩率；仅 delta sync 路径上报。"),
            ],
        ),
        "openclaw/perf": (
            "openclaw/perf/*",
            [("openclaw/perf/episode_elapsed_sec/{mean,p50,p90,p95,p99,max}", "完整 episode 端到端耗时（秒），包含 sandbox 准备、agent、reward 和清理；trajectory 级。")],
        ),
        "openclaw/tokens": (
            "openclaw/tokens/*",
            [
                ("openclaw/tokens/generated/{mean,p50,max}", "完整 episode 中模型实际生成 token 总数；trajectory 级。"),
                ("openclaw/tokens/max_context/{mean,p50,max}", "每条 episode 内单次模型请求所见最大 context，再在 batch 内统计；trajectory 级。"),
                ("openclaw/tokens/trained_response/{mean,p50,max}", "完整 episode 中 loss_mask=1、实际进入梯度计算的 response token 数。"),
                ("openclaw/tokens/response_region/mean", "episode 所有 fragment 的 response region token 数，可能包含 replay 重复 token。"),
                ("openclaw/tokens/trained_fraction", "实际训练 token / 模型实际生成 token。"),
            ],
        ),
        "openclaw/trajectory": (
            "openclaw/trajectory/*",
            [
                ("openclaw/trajectory/replay_inflation_ratio", "fragment response region token / episode 实际生成 token；大于 1 表示 replay/fork 重复。"),
                ("openclaw/trajectory/fragments_per_rollout", "每条完整 episode 平均产生的训练 Sample/fragment 数。"),
                ("openclaw/trajectory/forked_rollout_rate", "产生两个及以上 fragment 的 episode 比例。"),
                ("openclaw/trajectory/dangling/final_turn_rate", "最后停在未返回结果 tool call 的 episode 比例。"),
                ("openclaw/trajectory/dangling/masked_tokens_mean", "每条 episode 因 dangling final tool call 被 mask 的平均 token 数。"),
                ("openclaw/trajectory/dangling/masked_tokens_total", "当前 batch 因 dangling final tool call 被 mask 的 token 总数。"),
            ],
        ),
        "openclaw/behavior": (
            "openclaw/behavior/*",
            [
                ("openclaw/behavior/turns_per_rollout", "每条 episode 平均模型轮次。"),
                ("openclaw/behavior/tool_calls_per_rollout", "每条 episode 平均工具调用数。"),
                ("openclaw/behavior/tool_calls_per_turn", "每个模型轮次平均工具调用数。"),
            ],
        ),
        "openclaw/health": (
            "openclaw/health/*",
            [
                ("openclaw/health/terminal_rollouts", "进入终态的 episode 数。"),
                ("openclaw/health/excluded_rollouts{,_rate}", "被排除出 reward/advantage 聚合的 episode 数及比例。"),
                ("openclaw/health/rollout_aborts{,_rate}", "异常终止 episode 数及比例。"),
                ("openclaw/health/agent_truncations{,_rate}", "达到时间/长度边界但保留轨迹的 episode 数及比例。"),
                ("openclaw/health/agent_process_failures{,_rate}", "OpenClaw agent 进程失败 episode 数及比例。"),
                ("openclaw/health/verifier_failures{,_rate}", "Reward verifier 非零退出 episode 数及比例。"),
                ("openclaw/cumulative/*", "上述健康度指标从当前 run 开始累计的版本。"),
            ],
        ),
        "openclaw/reward": (
            "openclaw/reward/*",
            [("openclaw/reward/{mean,p50,max}", "完整 episode 的任务 reward 批次统计；trajectory 级。")],
        ),
        "openclaw/advisor_usage": (
            "openclaw/advisor_usage/*",
            [("openclaw/advisor_usage/calls_{0,1,2,3plus}/*", "按每条 episode 的 cloud_assist 调用次数分桶后的 episode 数和长度统计。")],
        ),
        "advisor/usage": (
            "advisor/usage/*",
            [
                ("advisor/usage/rollouts", "当前批次有效 episode 数。"),
                ("advisor/usage/cloud_rollouts{,_rate}", "使用过云 advisor 的 episode 数及比例。"),
                ("advisor/usage/cloud_calls{,_per_rollout}", "云调用次数及每 episode 平均值。"),
            ],
        ),
        "advisor/cost": (
            "advisor/cost/*",
            [
                ("advisor/cost/cloud_tokens{,_per_call,_per_rollout}", "云调用 token 数及平均值。"),
                ("advisor/cost/cloud_cost_rmb{,_per_call,_per_rollout}", "云调用人民币 cost 及平均值。"),
                ("advisor/cost/full_cloud_reference_cost_rmb", "当前批次任务的 full-cloud 参考 cost。"),
                ("advisor/cost/cost_usage_ratio", "实际云 cost / full-cloud 参考 cost。"),
                ("advisor/cost/{cost_budget_ratio,cost_lambda_before,cost_lambda_after}", "成本预算比例及 dual lambda 更新前后值。"),
            ],
        ),
        "advisor/reliability": (
            "advisor/reliability/*",
            [
                ("advisor/reliability/cloud_{observed_calls,successful_calls,failed_calls,success_rate}", "实际观测云请求的成功失败统计。"),
                ("advisor/reliability/cloud_{attempts,retry_count,attempts_per_call,retries_per_call}", "云请求 attempt 和 retry 统计。"),
                ("advisor/reliability/failure_status/*", "按 HTTP 状态或错误标签拆分的失败次数。"),
            ],
        ),
        "advisor/performance": (
            "advisor/performance/*",
            [("advisor/performance/cloud_latency_ms_{total,per_call,max}", "云请求总延迟、平均延迟和最大延迟（毫秒）。")],
        ),
        "advisor/trace": (
            "advisor trace / tool / model / transcript",
            [
                ("advisor/tool/*", "cloud_assist 工具调用、返回和 pending 数量。"),
                ("advisor/model/*", "cloud_assist 前后模型请求/响应数量。"),
                ("advisor/trace/*", "云 attempt、响应和错误在 trace 中的计数。"),
                ("advisor/transcript/*", "工具调用、结果和 follow-up 在 transcript 中的完整性统计。"),
            ],
        ),
        "advisor/reward": (
            "advisor/reward/*",
            [("advisor/reward/{task_advantage,cost_advantage,total_advantage}_{mean,std}", "成本约束 reward 的 task、cost 和最终 advantage 统计。")],
        ),
        "advisor/cumulative": (
            "advisor/cumulative/*",
            [("advisor/cumulative/*", "Advisor 使用量、cost、可靠性和延迟从当前 run 开始累计的版本。")],
        ),
    }


def _metric_guide_html(title: str, rows: list[tuple[str, str]]) -> str:
    table_rows = "".join(
        "<tr><td><code>"
        + escape(metric)
        + "</code></td><td>"
        + escape(description)
        + "</td></tr>"
        for metric, description in rows
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin: 0; padding: 20px; font: 14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color: #1f2937; background: #ffffff; }}
  h1 {{ margin: 0 0 6px; font-size: 21px; letter-spacing: 0; }}
  .note {{ margin: 0 0 18px; color: #64748b; }}
  table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
  th {{ position: sticky; top: 0; z-index: 1; padding: 10px 12px; text-align: left; background: #f1f5f9; border: 1px solid #dbe3ec; }}
  td {{ padding: 10px 12px; vertical-align: top; border: 1px solid #dbe3ec; overflow-wrap: anywhere; }}
  tr:nth-child(even) td {{ background: #f8fafc; }}
  th:first-child, td:first-child {{ width: 40%; }}
  code {{ font: 12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; color: #075985; background: #e0f2fe; padding: 2px 5px; border-radius: 4px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ color: #e5e7eb; background: #111827; }}
    .note {{ color: #94a3b8; }}
    th {{ background: #1f2937; border-color: #374151; }}
    td {{ border-color: #374151; }}
    tr:nth-child(even) td {{ background: #172033; }}
    code {{ color: #bae6fd; background: #0c4a6e; }}
  }}
</style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p class="note">按 SwanLab 指标前缀归类；每项会注明单位和 Sample、trajectory 或 step 等统计层级。</p>
  <table>
    <thead><tr><th>指标</th><th>含义 / 单位 / 统计层级</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>"""


def _log_metric_guides(swanlab) -> None:
    payload = {}
    for key, (title, rows) in _metric_guide_sections().items():
        payload[f"system/metric_guide/{key}"] = swanlab.Html(
            _metric_guide_html(title, rows),
            caption=f"{title}指标说明",
        )
    swanlab.log(payload, step=0)


def _get_swanlab():
    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "SwanLab tracking requires swanlab>=0.9.1. Install the project requirements before using --use-swanlab."
        ) from exc
    return swanlab


def _shared_project_get_or_create(original, get_project):
    """Use an existing writable project when workspace-level creation is denied.

    SwanLab 0.9.x calls a workspace-level create-or-get endpoint before it
    starts a run. Shared-project administrators can have WRITE/MANAGE access to
    the target project while that endpoint still reports workspace READ-only.
    """

    def get_or_create(*, username, name, public):
        try:
            return original(username=username, name=name, public=public)
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code != 403 or not username:
                raise

            project = get_project(username=username, name=name)
            permission = str(project.get("permission", "")).upper()
            if permission not in _SWANLAB_WRITABLE_PROJECT_PERMISSIONS:
                raise

            logger.warning(
                "SwanLab workspace project lookup was denied, but existing project %s/%s grants %s; "
                "continuing with the existing shared project.",
                username,
                name,
                permission,
            )
            return project

    get_or_create._slime_shared_project_compat = True
    return get_or_create


def _install_shared_project_compat(swanlab):
    """Install the SwanLab 0.9.x shared-project fallback once per process."""
    if not isinstance(swanlab, ModuleType) or swanlab.__name__ != "swanlab":
        return

    try:
        from swanlab.sdk.internal.core_python import client as core_client
        from swanlab.sdk.internal.core_python import utils as core_utils
    except ImportError:
        logger.debug("SwanLab shared-project compatibility is unavailable in this SDK version.")
        return

    current = core_utils.get_or_create_project
    if getattr(current, "_slime_shared_project_compat", False):
        return

    def get_project(*, username, name):
        return core_client.get(f"/project/{username}/{name}").data

    core_utils.get_or_create_project = _shared_project_get_or_create(current, get_project)


def _init_swanlab(args, config):
    swanlab = _get_swanlab()
    _install_shared_project_compat(swanlab)
    return swanlab.init(**_init_kwargs(args, config))


def _init_kwargs(args, config):
    return {
        "project": args.swanlab_project,
        "workspace": args.swanlab_workspace,
        "name": args.swanlab_experiment_name,
        "group": args.swanlab_group,
        "mode": args.swanlab_mode,
        "log_dir": args.swanlab_log_dir,
        "id": args.swanlab_run_id,
        "resume": "allow",
        "parallel": "shared",
        "config": config,
    }


def init_swanlab_primary(args):
    if not args.use_swanlab:
        args.swanlab_run_id = None
        return

    run = _init_swanlab(args, compute_config_for_logging(args))
    args.swanlab_run_id = run.id
    swanlab = _get_swanlab()
    _log_metric_guides(swanlab)
    logger.info(
        "SwanLab primary run initialized (project=%s, name=%s, id=%s).",
        args.swanlab_project,
        args.swanlab_experiment_name,
        args.swanlab_run_id,
    )


def init_swanlab_secondary(args, role=None):
    if not args.use_swanlab or args.swanlab_run_id is None:
        return

    _init_swanlab(args, compute_secondary_config_for_logging(args, role=role))
    logger.info("SwanLab secondary process joined shared run %s.", args.swanlab_run_id)


def log(args, metrics, step_key):
    if args.use_swanlab:
        _get_swanlab().log(_organize_metrics_for_swanlab(metrics), step=int(metrics[step_key]))


def finish(args):
    if not args.use_swanlab:
        return
    swanlab = _get_swanlab()
    if swanlab.has_run():
        swanlab.finish()
