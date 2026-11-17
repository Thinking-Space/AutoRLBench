# Derivation notes: multi_engine_prefix_affinity_collapse (historical evidence)

Runs (see `run_comparison.json`; labels are neutral and carry no external identifiers):

* `run-a` baseline: standalone SGLang router, `cache_aware` policy, 8 engines.
* `run-b` miles router (least in-flight requests, ignores the routing-key header), session gateway on, 4 engines x TP2.
* `run-c` same as `run-b` with 8 engines x TP1.
* `run-d` routing fixed (standalone SGLang router, `cache_aware`, gateway off), 8 engines; the 64-slot sandbox gate is
  still present (it is the subject of the sibling `sandbox_admission_starvation` task).

## 0. Headline numbers: documented vs derived

"documented" = the original 2026-09 investigation write-up. "full" = whole archive. "ref" = the named reference
window in `reference_windows.json`.

| run | metric | documented | derived (full archive) | derived (ref window) | verdict |
|---|---|---|---|---|---|
| run-a | prefix hit rate | 97.3% | **97.3%** | (= full) | reproduced exactly |
| run-a | prompt length p50 | 17986 | **17986** | | reproduced exactly |
| run-a | KV token usage p50 | 0.10 | **0.10** | | reproduced exactly |
| run-a | running-req mean/engine; total concurrency (x8) | 4.78; 38.3 | 5.35; 42.8 | | not reproduced (+12%), see note A |
| run-a | gen throughput / engine | 340 | 397.5 | | not reproduced (+17%), see note A |
| run-b | prefix hit rate | 7.6% | 20.7% | 7.6% (`best_fit_window` 14:18-14:30, searched) | see note B; 5-min window spread 2.5-57.2%, p50 12.7% |
| run-b | total concurrency (x4) | 5.2 | 8.4 | 7.7 | see note B; window p50 7.75 |
| run-c | prefix hit rate | 20.8% | 30.9% | 21.2% (`best_fit_window` 14:52-15:02, searched) | see note B; window spread 10.3-51.6%, p50 27.6% |
| run-c | running-req mean/engine; total concurrency (x8) | 1.12; 8.9 | 1.07; 8.6 | 1.13; 9.0 | consistent (window p50 8.1) |
| run-c | gen throughput / engine | 151 | 145.2 | 151.5 | consistent |
| run-c | prompt length p50 | 5250 | 5650 | 4883.5 | within window variation |
| run-d | prefix hit rate | 82.2% | 82.3% | 82.1% (`snapshot_until_1733`) | reproduced (+-0.1) |
| run-d | running-req mean/engine; max; total (x8) | 1.20; 9; 9.5 | 1.62; 16; 13.0 | **1.20; 9; 9.6** | reproduced in the snapshot window, note C |
| run-d | gen throughput / engine | 155 | 187.7 | **155.0** | reproduced in the snapshot window |
| run-d | semaphore waits n / p50 / p90 / max | 67 / 190 / 432 / 449 s | **67 / 190.3 / 432.4 / 449.4** | same | reproduced exactly |

Notes:

* **A (run-a concurrency / throughput).** The archive is the same 2000-line pod-log page the documented hit rate,
  prompt p50 and KV usage were computed from (all three reproduce exactly), but no aggregation of its Decode lines
  tried (pooled mean, per-engine means, repeat-weighted mean, Prefill-line `#running-req`, any sub-window) yields
  4.78 / 340 together; the closest sub-window (14:30-14:38) gives 4.74 / 366. The documented baseline concurrency
  is therefore not reproducible from surviving logs; the derived value (42.8, window spread 32.9-66.3) is *higher*,
  so the qualitative conclusion (baseline concurrency ~5x the miles-router runs) is unaffected.
* **B (run-b / run-c hit rate).** The documented values came from live log pages whose boundaries were not recorded.
  The token-weighted hit rate of these runs fluctuates strongly between 5-minute windows (run-b 2.5-57%, run-c
  10-52%), so a ~15-minute page can land anywhere in that range. The `best_fit_window` entries were found by searching
  for the documented values and are shown only to demonstrate that the documented numbers are attainable windows of
  these logs; they are not independent confirmation. Whole-archive values (20.7%, 30.9%) and every 5-minute window
  stay far below the cache-aware runs (run-a 96.5-98.6%, run-d 68.5-86.9%), which is the robust finding.
  For run-b the documented total concurrency 5.2 could not be matched jointly with 7.6% in any single window; the
  run's concurrency rises over time (4.4 early, up to 16 at 14:15-14:20) and its window p50 is 7.75.
* **C (run-d).** The documented snapshot was taken while the run was live, around 17:33 on the log clock; the
  archive continues to 17:41, and the late windows (after the sandbox queue drained) have higher concurrency. The
  documented "9.5" is 8 x the per-engine mean (1.20 x 8 = 9.6 here; the write-up presumably used an unrounded mean just below 1.20).
* `run-b`/`run-c` also log sandbox semaphore waits in their first ~15 minutes (both have the 64-slot gate:
  run-b n=85 p50 224 s, run-c n=83 p50 205 s); see the sandbox task for that mechanism.

## 1. Archives

Each `run-<label>/` holds `master.log.gz` and `worker.log.gz`: gzip-compressed JSON Lines, one object per console
line, `{"t": <ISO-8601 UTC capture time or null>, "pod": "master"|"worker", "msg": <sanitized line>}`.

* `t` is set only for lines that came from the in-process console capture written by the experiment tracker;
  lines that came from the platform's archived pod log have `t = null` (that archive carries no per-line time).
  **No metric uses `t`.** All metrics use the cluster-local clock embedded in the message text
  (e.g. `[2026-09-28 14:39:14]`, UTC+8).
* `[sanitized: N line(s) removed]` records mark where N consecutive source lines were dropped.
* Ray's `[repeated Nx across cluster]` markers and the Ray actor prefix `(<Actor> pid=<pid>[, ip=<node-x>])` are kept
  verbatim (IPs replaced by per-run node labels).
* `manifest.json` lists, per archive, record counts, how many source lines were dropped, coverage, SHA-256 and
  which source(s) the archive was assembled from. Some archives are only the last 2000 lines of the pod console
  (the platform's archive limit); this is stated per archive.

Sanitization never edits the fields that the metrics read (the numbers in SGLang `Prefill batch` / `Decode batch`
lines, the semaphore-wait seconds, the slow-generate turn/elapsed values, dropped-group ids/reasons, train-step
markers). Metrics computed on the unsanitized source lines and on these archives are identical.

## 2. Effective timestamp and windows

`derive_metrics.py` reads `master.log.gz` then `worker.log.gz` of a run in record order. For every record the
*effective timestamp* is the first `[YYYY-MM-DD HH:MM:SS` found in `msg`; records without one inherit the last
effective timestamp seen earlier in the same archive (carry-forward). Records before the first embedded
timestamp have none and are only counted in whole-archive aggregates and in reference windows that have no lower
bound. 5-minute windows are `[HH:MM, HH:MM+5)` buckets of the effective timestamp (minute floored to a multiple
of 5). Reference windows (`reference_windows.json`) are half-open `[from, to)`; `null` means unbounded.

## 3. Metric definitions (all line-based; regexes are in `derive_metrics.py`)

| field | source lines | definition |
|---|---|---|
| `prefix_hit_rate_pct` | `... Prefill batch, #new-seq: S, #new-token: N, #cached-token: C, ...` | `100 * sum(C) / sum(N + C)` over the printed lines in the window (token-weighted, **not** the mean of per-line ratios; per-line ratios are biased by `chunked_prefill_size=8192` chunking, where only the first chunk of a long prompt can hit). |
| `prefill_new_ktokens`, `prefill_cached_ktokens` | same | `sum(N) / 1000`, `sum(C) / 1000` (3 decimals, i.e. exact token counts expressed in thousands). |
| `prompt_len_p50` | same | median of `N + C` over printed Prefill lines (a chunked long prompt contributes several lines). |
| `running_req_mean_per_engine` / `running_req_max` | `... Decode batch, #running-req: R, #token: K, token usage: U, ... gen throughput (token/s): G, ...` | mean / max of `R` over printed Decode lines, pooled across engines. Each Decode line is one engine's scheduler snapshot. With TP>1 only the TP0 scheduler prints Decode lines. |
| `total_concurrency_est` | same | `running_req_mean_per_engine * rollout_engines`, where `rollout_engines` is the configured engine count from `run_comparison.json`. This is the "total concurrency" convention of the original write-up. |
| `sum_of_engine_mean_running_req` | same | group Decode lines by the engine actor pid in the Ray prefix, average `R` per engine, sum over engines seen. Alternative estimate; differs from the above when engines print unevenly or are missing from a window. |
| `engines_seen_in_decode_lines` | same | number of distinct engine pids that printed a Decode line. |
| `gen_throughput_mean_tok_s_per_engine` | same | mean of `G` over printed Decode lines (per-engine decode throughput as SGLang reports it). |
| `kv_token_usage_p50` | same | median of `U` (fraction of the engine's KV token pool in use). |
| `sem_wait_lines`, `sem_wait_p50_s`, `sem_wait_p90_s`, `sem_wait_max_s` | `sandbox setup waited X s for semaphore` | count and linear-interpolated percentiles of `X` over printed lines. The warning is only emitted for waits > 5 s. |
| `sem_wait_events_lower_bound` | same | `sum over lines of (N if the line carries "[repeated Nx ...]" else 1)`: Ray folds same-pattern lines from the following ~5 s into one summary line whose N counts the folded occurrences; their values are not printed, so percentiles use printed values only. |
| `slow_generate_*` | `slow agentic generate: turn=T elapsed=E s` | percentiles/max of `E` and `T` over printed lines. The line is only logged for slow LLM calls, so `turn` is the turn index of slow calls (a proxy for trajectory depth). |
| `dropped_group_*` | `[fully-async] Dynamic filter dropped group I (REASON); ...` | `dropped_group_lines` = printed lines; `dropped_groups_distinct_ids_by_reason` = distinct `I` per REASON. Each drop is printed twice in these logs (once per Ray prefix form), hence distinct ids. A dropped group is a *completed* group (all samples finished) that was filtered out; there is no log line for accepted groups, so this is a lower bound on completed groups. |
| `train_steps_ft_distinct`, `first_train_step_ts` | `ft op=train_step rollout=R step=S ...` | distinct `(R, S)` pairs and the earliest effective timestamp. |
| `train_steps_logged_distinct` | `log_utils.py:<n> - step S: {...}` | distinct `S` (older logger format used by the baseline). |
| `collected_max` | `Warning: No progress for ...s. Queue size: q, Collected: c/n` | max `c` (baseline only). |
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
* Reproduction tolerance: the original write-up computed its numbers from *live* 2000-line pod-log pages
  (about 15-25 minutes of console each) whose exact boundaries were not recorded. Where those pages can no
  longer be reconstructed exactly, this directory reports the whole-archive value, the 5-minute window spread,
  and (labelled as such) a window chosen to match the documented value.

## 4. Re-deriving

```
python3 -I derive_metrics.py .            # rewrites metrics_summary.json, metrics_windows_5min.{json,csv}
python3 -I derive_metrics.py . --check    # verifies the shipped metric files are reproduced byte-for-byte
```
Stdlib only (Python >= 3.8). `run_comparison.json` supplies the configured engine counts;
`reference_windows.json` the named windows.
