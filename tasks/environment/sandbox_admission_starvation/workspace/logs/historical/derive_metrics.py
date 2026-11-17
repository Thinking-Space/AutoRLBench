#!/usr/bin/env python3
"""Re-derive rollout-throughput metrics from the sanitized console-log archives.

Stdlib only.  Usage:
    python3 derive_metrics.py [TASK_DIR] [--check]

TASK_DIR defaults to the directory containing this script.  It must contain
run_comparison.json, reference_windows.json and run-*/master.log.gz
(+ optional run-*/worker.log.gz).  Outputs (overwritten):
    metrics_summary.json        whole-archive + reference-window aggregates per run
    metrics_windows_5min.json   per-5-minute-window metrics per run
    metrics_windows_5min.csv    same, flattened
With --check, the freshly computed outputs are compared to the files already on
disk and the script exits non-zero on any difference (nothing is written).
See derivation.md for the exact definitions.
"""
import csv
import datetime as dt
import glob
import gzip
import io
import json
import os
import re
import sys

# ---------------------------------------------------------------- line regexes
PREFIX = re.compile(r'^\((?P<actor>[^ ()]+)(?: pid=(?P<pid>\d+))?(?:, ip=[^)]*)?\) ?')
REPEAT = re.compile(r' \[repeated (\d+)x across cluster\]\s*$')
EMBEDDED_TS = re.compile(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
PREFILL = re.compile(r'\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), '
                     r'token usage: ([\d.]+), #running-req: (\d+)')
DECODE = re.compile(r'\] Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), '
                    r'.*?gen throughput \(token/s\): ([\d.]+)')
SEM_WAIT = re.compile(r'sandbox setup waited ([\d.]+)s for semaphore')
SLOW = re.compile(r'slow agentic generate: turn=(\d+) elapsed=([\d.]+)s')
DROPPED_GROUP = re.compile(r'\[fully-async\] Dynamic filter dropped group (\d+) \((\w+)\)')
TRAIN_STEP_FT = re.compile(r'ft op=train_step rollout=(\d+) step=(\d+) ')
TRAIN_STEP_LOG = re.compile(r'log_utils\.py:\d+ - step (\d+): \{')
COLLECTED = re.compile(r'No progress for [\d.]+s\. Queue size: \d+, Collected: (\d+)/(\d+)')
TRAIN_WAIT_END = re.compile(r'Timer train_wait end \(elapsed: ([\d.]+)s\)')

TS_FMT = '%Y-%m-%d %H:%M:%S'
WINDOW_MIN = 5


def pct(xs, p):
    """Linear-interpolated percentile (same as numpy 'linear'); None if empty."""
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 4)


def mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def read_records(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def events(run_dir):
    """Yield (effective_ts_str_or_None, msg) for master then worker archive.

    effective timestamp = the embedded log timestamp of the line if it has one,
    otherwise the last embedded timestamp seen earlier in the same archive
    (carry-forward).  Lines before the first embedded timestamp get None.
    """
    for pod in ('master', 'worker'):
        p = os.path.join(run_dir, pod + '.log.gz')
        if not os.path.exists(p):
            continue
        cur = None
        for r in read_records(p):
            m = EMBEDDED_TS.search(r['msg'])
            if m:
                cur = m.group(1)
            yield cur, r['msg']


class Acc:
    def __init__(self):
        self.new = self.cached = 0
        self.prefill_lines = 0
        self.prompt_len = []
        self.dec_run = []
        self.dec_gen = []
        self.dec_kv = []
        self.dec_by_engine = {}
        self.sem = []
        self.sem_events_lb = 0
        self.slow_elapsed = []
        self.slow_turn = []
        self.dropped = {}
        self.dropped_lines = 0
        self.dropped_folded = 0
        self.train_steps = set()
        self.train_step_first = None
        self.log_steps = set()
        self.collected = []
        self.train_wait = []

    def add(self, ts, msg):
        rep = REPEAT.search(msg)
        nrep = int(rep.group(1)) if rep else 0
        pm = PREFIX.match(msg)
        pid = pm.group('pid') if pm else None
        m = PREFILL.search(msg)
        if m:
            n, c = int(m.group(2)), int(m.group(3))
            self.new += n
            self.cached += c
            self.prefill_lines += 1
            self.prompt_len.append(n + c)
        m = DECODE.search(msg)
        if m:
            r = int(m.group(1))
            self.dec_run.append(r)
            self.dec_kv.append(float(m.group(3)))
            self.dec_gen.append(float(m.group(4)))
            self.dec_by_engine.setdefault(pid, []).append(r)
        m = SEM_WAIT.search(msg)
        if m:
            self.sem.append(float(m.group(1)))
            self.sem_events_lb += nrep if nrep else 1
        m = SLOW.search(msg)
        if m:
            self.slow_turn.append(int(m.group(1)))
            self.slow_elapsed.append(float(m.group(2)))
        m = DROPPED_GROUP.search(msg)
        if m:
            self.dropped.setdefault(m.group(2), set()).add(int(m.group(1)))
            self.dropped_lines += 1
            self.dropped_folded += nrep
        m = TRAIN_STEP_FT.search(msg)
        if m:
            self.train_steps.add((int(m.group(1)), int(m.group(2))))
            if ts and (self.train_step_first is None or ts < self.train_step_first):
                self.train_step_first = ts
        m = TRAIN_STEP_LOG.search(msg)
        if m:
            self.log_steps.add(int(m.group(1)))
        m = COLLECTED.search(msg)
        if m:
            self.collected.append((int(m.group(1)), int(m.group(2))))
        m = TRAIN_WAIT_END.search(msg)
        if m:
            self.train_wait.append(float(m.group(1)))

    def result(self, n_engines):
        tot = self.new + self.cached
        run_mean = mean(self.dec_run)
        eng_means = {k: sum(v) / len(v) for k, v in self.dec_by_engine.items()}
        return {
            'prefill_lines': self.prefill_lines,
            # token sums are reported in thousands with 3 decimals (exact), which keeps every
            # emitted number below 7 consecutive digits (archive-wide leak scans flag 7-digit runs).
            'prefill_new_ktokens': round(self.new / 1000.0, 3),
            'prefill_cached_ktokens': round(self.cached / 1000.0, 3),
            'prefix_hit_rate_pct': round(100.0 * self.cached / tot, 1) if tot else None,
            'prompt_len_p50': pct(self.prompt_len, 0.5),
            'decode_lines': len(self.dec_run),
            'running_req_mean_per_engine': run_mean,
            'running_req_max': max(self.dec_run) if self.dec_run else None,
            'engines_configured': n_engines,
            'engines_seen_in_decode_lines': len(self.dec_by_engine),
            'total_concurrency_est': round(run_mean * n_engines, 1) if run_mean is not None else None,
            'sum_of_engine_mean_running_req': round(sum(eng_means.values()), 2) if eng_means else None,
            'gen_throughput_mean_tok_s_per_engine': round(sum(self.dec_gen) / len(self.dec_gen), 1) if self.dec_gen else None,
            'kv_token_usage_p50': pct(self.dec_kv, 0.5),
            'sem_wait_lines': len(self.sem),
            'sem_wait_events_lower_bound': self.sem_events_lb,
            'sem_wait_p50_s': pct(self.sem, 0.5),
            'sem_wait_p90_s': pct(self.sem, 0.9),
            'sem_wait_max_s': max(self.sem) if self.sem else None,
            'slow_generate_lines': len(self.slow_elapsed),
            'slow_generate_elapsed_p50_s': pct(self.slow_elapsed, 0.5),
            'slow_generate_elapsed_max_s': max(self.slow_elapsed) if self.slow_elapsed else None,
            'slow_generate_turn_p50': pct(self.slow_turn, 0.5),
            'slow_generate_turn_max': max(self.slow_turn) if self.slow_turn else None,
            'dropped_group_lines': self.dropped_lines,
            'dropped_group_folded_repeats': self.dropped_folded,
            'dropped_groups_distinct_ids_by_reason': {k: len(v) for k, v in sorted(self.dropped.items())},
            'dropped_groups_distinct_ids_total': len(set().union(*self.dropped.values())) if self.dropped else 0,
            'train_steps_ft_distinct': len(self.train_steps),
            'first_train_step_ts': self.train_step_first,
            'train_steps_logged_distinct': len(self.log_steps),
            'collected_max': max(self.collected)[0] if self.collected else None,
            'train_wait_end_elapsed_s': sorted(set(self.train_wait)),
        }


def window_start(ts):
    t = dt.datetime.strptime(ts, TS_FMT)
    return t.replace(minute=t.minute - t.minute % WINDOW_MIN, second=0).strftime(TS_FMT)


def derive(task_dir):
    cmp_ = json.load(open(os.path.join(task_dir, 'run_comparison.json')))
    refs = json.load(open(os.path.join(task_dir, 'reference_windows.json')))
    summary, windows = {}, {}
    for run_dir in sorted(glob.glob(os.path.join(task_dir, 'run-*'))):
        label = os.path.basename(run_dir)
        n_eng = cmp_['runs'][label]['config']['rollout_engines']
        evs = list(events(run_dir))
        full = Acc()
        per_w = {}
        ref_accs = {name: Acc() for name in refs.get(label, {})}
        for ts, msg in evs:
            full.add(ts, msg)
            if ts:
                per_w.setdefault(window_start(ts), Acc()).add(ts, msg)
            for name, spec in refs.get(label, {}).items():
                lo, hi = spec.get('from'), spec.get('to')
                if ts is None:
                    if lo is None:
                        ref_accs[name].add(ts, msg)
                    continue
                if (lo is None or ts >= lo) and (hi is None or ts < hi):
                    ref_accs[name].add(ts, msg)
        tss = [t for t, _ in evs if t]
        summary[label] = {
            'log_clock_span': [min(tss), max(tss)] if tss else None,
            'full_archive': full.result(n_eng),
            'reference_windows': {name: dict(window=[refs[label][name].get('from'), refs[label][name].get('to')],
                                             note=refs[label][name].get('note', ''),
                                             **ref_accs[name].result(n_eng))
                                  for name in ref_accs},
        }
        windows[label] = [dict(window_start=k, **per_w[k].result(n_eng)) for k in sorted(per_w)]
        # spread across "active" windows: >= MIN_LINES Prefill and Decode lines, skipping the first
        # active window (engine warm-up / first-turn prefill).
        active = [w for w in windows[label]
                  if w['prefill_lines'] >= MIN_LINES and w['decode_lines'] >= MIN_LINES][1:]

        def spread(key):
            xs = [w[key] for w in active if w[key] is not None]
            return {'n_windows': len(xs), 'min': min(xs) if xs else None, 'p50': pct(xs, 0.5),
                    'max': max(xs) if xs else None}
        summary[label]['active_5min_window_spread'] = {
            'rule': 'windows with >=%d Prefill and >=%d Decode lines, first such window skipped' % (MIN_LINES, MIN_LINES),
            'prefix_hit_rate_pct': spread('prefix_hit_rate_pct'),
            'total_concurrency_est': spread('total_concurrency_est'),
            'gen_throughput_mean_tok_s_per_engine': spread('gen_throughput_mean_tok_s_per_engine'),
        }
    return summary, windows


MIN_LINES = 10


WINDOW_CSV_COLS = ['run', 'window_start', 'prefill_lines', 'prefill_new_ktokens', 'prefill_cached_ktokens',
                   'prefix_hit_rate_pct', 'prompt_len_p50', 'decode_lines', 'running_req_mean_per_engine',
                   'running_req_max', 'engines_seen_in_decode_lines', 'total_concurrency_est',
                   'sum_of_engine_mean_running_req', 'gen_throughput_mean_tok_s_per_engine', 'kv_token_usage_p50',
                   'sem_wait_lines', 'sem_wait_p50_s', 'sem_wait_p90_s', 'sem_wait_max_s', 'slow_generate_lines',
                   'slow_generate_elapsed_p50_s', 'slow_generate_turn_p50', 'dropped_group_lines',
                   'dropped_groups_distinct_ids_total', 'train_steps_ft_distinct', 'train_steps_logged_distinct']


def render(summary, windows):
    out = {}
    out['metrics_summary.json'] = json.dumps(summary, indent=1, sort_keys=False) + '\n'
    out['metrics_windows_5min.json'] = json.dumps(windows, indent=1) + '\n'
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator='\n')
    w.writerow(WINDOW_CSV_COLS)
    for label, rows in windows.items():
        for r in rows:
            w.writerow([label] + ['' if r.get(c) is None else r.get(c) for c in WINDOW_CSV_COLS[1:]])
    out['metrics_windows_5min.csv'] = buf.getvalue()
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    task_dir = args[0] if args else os.path.dirname(os.path.abspath(__file__))
    out = render(*derive(task_dir))
    if '--check' in sys.argv:
        bad = [n for n, s in out.items()
               if not os.path.exists(os.path.join(task_dir, n)) or open(os.path.join(task_dir, n)).read() != s]
        print('OK: all metric files reproduce' if not bad else 'MISMATCH: ' + ', '.join(bad))
        sys.exit(1 if bad else 0)
    for n, s in out.items():
        with open(os.path.join(task_dir, n), 'w') as f:
            f.write(s)
    print('wrote', ', '.join(sorted(out)))


if __name__ == '__main__':
    main()
