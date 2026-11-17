# Derivation notes: sandbox_admission_starvation (historical evidence)

Runs (see `run_comparison.json`; labels are neutral and carry no external identifiers):

* `run-a` routing already fixed (standalone SGLang router, `cache_aware`), sandbox setup gate = 64 slots for 256
  in-flight trajectories, sandbox HTTP client pool at defaults. (Same archives as `run-d` of the routing task.)
* `run-b` identical except gate = 256 and a larger sandbox HTTP pool (2000 connections / 2048 keep-alive / 120 s).

## 0. Headline numbers: documented vs derived

"documented" = the original 2026-09 investigation write-up. "full" = whole archive. "ref" = named window in
`reference_windows.json`.

| run | metric | documented | derived (full archive) | derived (ref window) | verdict |
|---|---|---|---|---|---|
| run-a | semaphore waits n / p50 / p90 / max | 67 / 190 / 432 / 449 s | **67 / 190.3 / 432.4 / 449.4** | same (`snapshot_until_1733`) | reproduced exactly |
| run-a | semaphore waits incl. Ray-folded repeats | - | >= 378 events | | new (lower bound) |
| run-a | slow-generate turn p50 / max | 2 / 9 | 3 / 14 | **2 / 9** | reproduced in snapshot window |
| run-a | slow-generate elapsed p50 / max | 91.8 / 391 s | 92.6 / 525.1 | 90.9 / **391.2** | max exact, p50 within 1 s |
| run-a | running-req mean / max per engine; total (x8) | 1.20 / 9; 9.5 | 1.62 / 16; 13.0 | **1.20 / 9; 9.6** | reproduced in snapshot window |
| run-a | gen throughput / engine | 155 | 187.7 | **155.0** | reproduced in snapshot window |
| run-a | KV token usage p50 | 0.010 | 0.02 | **0.01** | reproduced in snapshot window |
| run-a | prefix hit rate | 82.2% | 82.3% | 82.1% | reproduced (+-0.1) |
| run-a | prompt length p50 | 4442 | 5511 | 4343 | within 2% in snapshot window |
| run-a | completed groups | 0 | 0 dropped-group ids, no train step | 0 | consistent |
| run-b | semaphore waits (> 5 s) | 0 | **0** | 0 | reproduced |
| run-b | running-req mean / max per engine; total (x8) | 5.12 / 16; 41.0 | 8.16 / 16; 65.3 | 5.36 / **16**; 42.9 (`documented_window_1755_1807`) | within 5% in the documented window, note A |
| run-b | slow-generate elapsed p50 / max | 34.6 / 57.3 s | 39.2 / 328.7 | 34.8 / **57.3** | reproduced in documented window |
| run-b | prefix hit rate | 80.7% | 81.7% | 82.0% | within 1.3 pp |
| run-b | KV token usage p50 | 0.030 | 0.29 | 0.035 | close in documented window |
| run-b | gen throughput / engine | 237 | 539.8 | 365.2 | not reproduced, note A |
| run-b | prompt length p50 | 6088 | 10301 | 6867 | +13% in documented window, note A |
| run-b | first train step | "not reached yet" (as of 18:06) | **18:36:19**, 8 train steps (rollout 0) by 18:40 | | new: the archive extends past the write-up |

Notes:

* **A (run-b documented window).** The write-up states its window as 17:55-18:06 but its numbers came from a live
  2000-line page whose exact start is unknown; the first ~10 minutes of this run are dominated by sandbox creation
  (256 sandboxes at once) and engine warm-up, so small boundary shifts move throughput-type averages a lot
  (e.g. total concurrency is 19.3 in the 18:00-18:05 window and 78.4 in 18:05-18:10; 17:55-18:00 has no Decode lines). The concurrency, max running requests,
  slow-generate and semaphore conclusions reproduce; the documented 237 tok/s/engine does not (365 in the stated
  window). After 18:07 the run sustains a total concurrency of ~60-70 (reference window 18:07-18:33: 8.8 running
  requests per engine, 70.7 total; 5-minute window p50 62.1). In that window 40 of 310 Decode snapshots sit at the
  16-request per-engine cap and 50 show a non-empty SGLang queue (`#queue-req` > 0): the per-engine
  `max_running_requests=16` starts to bind intermittently; admission no longer does.
* `run-b`'s trainer waited 2159 s (`Timer train_wait end`) for its first batch (32 accepted groups); 50 distinct groups
  were dropped by the dynamic filter over the whole archive (to 18:53), 23 of them in 18:07-18:33
  (`dropped_groups_distinct_ids_by_reason`).
* The routing task's miles-router runs (not included here because they confound two causes) show the same gate
  signature in their first ~15 minutes (n=85 p50 224 s; n=83 p50 205 s).

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
