# Historical evidence: derivation of the metric files

The metric files in this directory are recomputed from the sanitized console archives by
`python3 derive_metrics.py .` (stdlib only; `--check` verifies the shipped files).
The run configurations are listed in `run_comparison.json`.

## 1. Effective timestamp and windows

`derive_metrics.py` reads `master.log.gz` then `worker.log.gz` of a run in record order. For every record the
*effective timestamp* is the first `[YYYY-MM-DD HH:MM:SS` found in `msg`; records without one inherit the last
effective timestamp seen earlier in the same archive (carry-forward). Records before the first embedded
timestamp have none and are only counted in whole-archive aggregates and in reference windows that have no lower
bound. 5-minute windows are `[HH:MM, HH:MM+5)` buckets of the effective timestamp (minute floored to a multiple
of 5). Reference windows (`reference_windows.json`) are half-open `[from, to)`; `null` means unbounded.

## 2. Metric definitions (all line-based; regexes are in `derive_metrics.py`)

| field | source lines | definition |
|---|---|---|
| `prefix_hit_rate_pct` | `... Prefill batch, #new-seq: S, #new-token: N, #cached-token: C, ...` | `100 * sum(C) / sum(N + C)` over the printed lines in the window (token-weighted, **not** the mean of per-line ratios; per-line ratios are biased by `chunked_prefill_size=8192` chunking, where only the first chunk of a long prompt can hit). |
| `prefill_new_ktokens`, `prefill_cached_ktokens` | same | `sum(N) / 1000`, `sum(C) / 1000` (3 decimals, i.e. exact token counts expressed in thousands). |
| `prompt_len_p50` | same | median of `N + C` over printed Prefill lines (a chunked long prompt contributes several lines). |
| `running_req_mean_per_engine` / `running_req_max` | `... Decode batch, #running-req: R, #token: K, token usage: U, ... gen throughput (token/s): G, ...` | mean / max of `R` over printed Decode lines, pooled across engines. Each Decode line is one engine's scheduler snapshot. With TP>1 only the TP0 scheduler prints Decode lines. |
| `total_concurrency_est` | same | `running_req_mean_per_engine * rollout_engines`, where `rollout_engines` is the configured engine count from `run_comparison.json`. |
| `sum_of_engine_mean_running_req` | same | group Decode lines by the engine actor pid in the Ray prefix, average `R` per engine, sum over engines seen. Alternative estimate; differs from the above when engines print unevenly or are missing from a window. |
| `engines_seen_in_decode_lines` | same | number of distinct engine pids that printed a Decode line. |
| `gen_throughput_mean_tok_s_per_engine` | same | mean of `G` over printed Decode lines (per-engine decode throughput as SGLang reports it). |
| `kv_token_usage_p50` | same | median of `U` (fraction of the engine's KV token pool in use). |
| `sem_wait_lines`, `sem_wait_p50_s`, `sem_wait_p90_s`, `sem_wait_max_s` | `sandbox setup waited X s for semaphore` | count and linear-interpolated percentiles of `X` over printed lines. The warning is only emitted for waits > 5 s. |
| `sem_wait_events_lower_bound` | same | `sum over lines of (N if the line carries "[repeated Nx ...]" else 1)`: Ray folds same-pattern lines from the following ~5 s into one summary line whose N counts the folded occurrences; their values are not printed, so percentiles use printed values only. |
| `slow_generate_*` | `slow agentic generate: turn=T elapsed=E s` | percentiles/max of `E` and `T` over printed lines. The line is only logged for slow LLM calls, so `turn` is the turn index of slow calls (a proxy for trajectory depth). |
| `dropped_group_*` | `[fully-async] Dynamic filter dropped group I (REASON); ...` | `dropped_group_lines` = printed lines; `dropped_groups_distinct_ids_by_reason` = distinct `I` per REASON. Each drop is printed twice in these logs (once per Ray prefix form), hence distinct ids. A dropped group is a *completed* group (all samples finished) that was filtered out; there is no log line for accepted groups, so this is a lower bound on completed groups. |
| `train_steps_ft_distinct`, `first_train_step_ts` | `ft op=train_step rollout=R step=S ...` | distinct `(R, S)` pairs and the earliest effective timestamp. |
| `train_steps_logged_distinct` | `log_utils.py:<n> - step S: {...}` | distinct `S` . |
| `collected_max` | `Warning: No progress for ...s. Queue size: q, Collected: c/n` | max `c` . |
| `train_wait_end_elapsed_s` | `Timer train_wait end (elapsed: X s)` | values of `X` (how long the trainer waited for the first batch). |
| `active_5min_window_spread` | 5-minute windows | min / p50 / max of hit rate, total concurrency and gen throughput over windows with >=10 Prefill and >=10 Decode lines, skipping the first such window (engine warm-up). |

Percentiles use linear interpolation between order statistics (numpy's default "linear" method).

### Caveats that apply to every number

* **Ray de-duplication.** Ray prints the first occurrence of a line pattern (numbers ignored) immediately and
  folds later same-pattern lines from other processes into one `[repeated Nx across cluster]` line. Every
  SGLang Prefill/Decode metric above is therefore computed over a *sample* of scheduler lines (the printed
  ones), not over all scheduler iterations. Weighting by N is not possible for token sums because folded lines'
  numbers are not printed. The engine pid in a folded line's prefix is that of the representative line only.
* Ray also prints some lines with and without the `ip=` part of the prefix; these are different lines unless
  the full text is identical (only the dropped-group lines are systematically doubled).

## 3. Re-deriving

```
python3 -I derive_metrics.py .            # rewrites metrics_summary.json, metrics_windows_5min.{json,csv}
python3 -I derive_metrics.py . --check    # verifies the shipped metric files are reproduced byte-for-byte
```
Stdlib only (Python >= 3.8). `run_comparison.json` supplies the configured engine counts;
`reference_windows.json` the named windows.
