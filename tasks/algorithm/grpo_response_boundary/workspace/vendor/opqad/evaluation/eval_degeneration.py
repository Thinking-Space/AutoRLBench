#!/usr/bin/env python3
"""Measure output degeneration in eval shards, because accuracy saturates.

Why this exists
---------------
The off-policy 1.5B arm reads `accuracy 0.0000` before training and `0.0000`
after 150 steps, which looks like "the objective did nothing".  It is not what
happened: both runs degenerate, and measuring *when* they degenerate separates
them cleanly.  Accuracy could not show it because it was already floored at 0,
so a saturated metric hid a real regression.

Two detectors, because the two arms fail in DIFFERENT ways
----------------------------------------------------------
A non-Latin-junk test alone is not enough, and using it alone gives a wrong
answer.  The untrained 1.5B degenerates into repeated *English* ("find the time
in time time total plus the time plus including") and only drifts into CJK much
later; the trained one goes straight to `的的的的` plus random multilingual
tokens.  Scoring both with the non-Latin test alone reports 52% vs 99% and an
onset ratio of 14x -- an artifact of measuring a different failure mode in each
arm.  With both detectors it is 100% vs 99% and 2.5x.

    junk:    a 200-char window >=25% characters outside Latin/ASCII
    repeat:  a 200-char window whose word 4-grams are <=55% distinct

Control that the detectors are not just firing on chain-of-thought style: the
4B at the IDENTICAL bit rate, recipe and eval path scores 0% junk and 12%
repeat, with a median clean fraction of 1.000, before and after training.

Reporting
---------
Onset is CENSORED: a sample that never degenerates is counted at its full
length, not dropped.  Dropping it biases the median towards whichever arm
degenerates less often -- exactly backwards.  Quantiles are over all samples.

`--paired` matches samples by problem `id` across arms and reports on how many
problems arm 2 collapses earlier than arm 1.  This is the number to quote: it
is immune to the two arms having different sample counts and different response
length distributions.

Token position is approximated by scaling the character offset by
`response_tokens` -- good to maybe 10%, far finer than the effects measured.
It is a heuristic, not a definition of "broken".  Use it as a regression signal
between comparable runs, not as an absolute quality gate.  `extraction_failures`
does NOT catch any of this: it was 0 on every one of these runs, because the
extractor regex-matches a number out of the wreckage and reports success.

Usage
-----
    python opqad/evaluation/eval_degeneration.py <glob> [<glob> ...]
    python opqad/evaluation/eval_degeneration.py --paired <base-glob> <trained-glob>
"""

from __future__ import annotations

import glob as globmod
import json
import statistics as st
import sys

WINDOW = 200
STRIDE = 50
JUNK_FRACTION = 0.25
REPEAT_DISTINCT = 0.55
MIN_WORDS = 12


def _is_junk(ch: str) -> bool:
    o = ord(ch)
    return o > 0x2E00 or (0x0370 <= o <= 0x1FFF)


def junk_onset(text: str) -> int | None:
    """First char offset of a sustained non-Latin run, or None."""
    for i in range(0, max(len(text) - WINDOW, 0), STRIDE):
        w = text[i : i + WINDOW]
        if sum(map(_is_junk, w)) / WINDOW >= JUNK_FRACTION:
            return i
    return None


def repeat_onset(text: str) -> int | None:
    """First char offset of a low-diversity run, or None.

    Catches degeneration into repeated English, which junk_onset misses.
    """
    for i in range(0, max(len(text) - WINDOW, 0), STRIDE):
        w = text[i : i + WINDOW].split()
        if len(w) < MIN_WORDS:
            continue
        grams = [tuple(w[j : j + 4]) for j in range(len(w) - 3)]
        if len(set(grams)) / len(grams) <= REPEAT_DISTINCT:
            return i
    return None


def quantile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = p * (len(xs) - 1)
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def scan(files: list[str]) -> dict:
    """Per-sample rows plus the shard-level bookkeeping we cross-check."""
    rows = []
    correct = n_samples = 0
    bits: set[float] = set()
    skips: set = set()
    for f in files:
        d = json.load(open(f))
        correct += d.get("correct", 0)
        n_samples += d.get("total_samples", 0)
        o = d.get("oscar") or {}
        bits.add(round(o.get("effective_bits", 0.0), 4))
        skips.add(o.get("skip_layer_count"))
        for i, pred in enumerate(d.get("predictions", [])):
            key = str(pred.get("id", pred.get("index", i)))
            for s in pred.get("samples", []):
                text = s.get("response", "")
                tokens = s.get("response_tokens", 0) or 0
                scale = tokens / max(len(text), 1)
                j = junk_onset(text)
                r = repeat_onset(text)
                first = min([x for x in (j, r) if x is not None], default=None)
                rows.append(
                    {
                        "key": key,
                        "tokens": tokens,
                        "junk": None if j is None else j * scale,
                        "repeat": None if r is None else r * scale,
                        # censored: clean samples counted at their full length
                        "onset": tokens if first is None else first * scale,
                        "degenerate": first is not None,
                    }
                )
    return {
        "rows": rows,
        "shards": len(files),
        "accuracy": correct / n_samples if n_samples else float("nan"),
        "n": n_samples,
        "bits": sorted(bits),
        "skip": sorted(x for x in skips if x is not None),
    }


def report(label: str, res: dict) -> None:
    rows = res["rows"]
    if not rows:
        print(f"{label}: no samples")
        return
    tokens = [r["tokens"] for r in rows]
    onset = [r["onset"] for r in rows]
    n = len(rows)
    nj = sum(1 for r in rows if r["junk"] is not None)
    nr = sum(1 for r in rows if r["repeat"] is not None)
    nd = sum(1 for r in rows if r["degenerate"])
    clean_frac = [r["onset"] / max(r["tokens"], 1) for r in rows]
    print(f"\n{label}")
    print(
        f"  shards {res['shards']}  accuracy {res['accuracy']:.4f} (n={res['n']})  "
        f"bits {res['bits']}  skip {res['skip']}"
    )
    print(
        f"  response_tokens  p50 {quantile(tokens, 0.5):.0f}  "
        f"p90 {quantile(tokens, 0.9):.0f}  mean {st.mean(tokens):.0f}"
    )
    print(
        f"  degenerate {nd}/{n} ({100 * nd / n:.0f}%)   "
        f"[non-latin {100 * nj / n:.0f}%, repeat {100 * nr / n:.0f}%]"
    )
    qs = "  ".join(f"p{int(p * 100):02d} {quantile(onset, p):.0f}" for p in (0.1, 0.25, 0.5, 0.75, 0.9))
    print(f"  censored onset (tok)  {qs}")
    print(
        f"  clean fraction of output  p10 {quantile(clean_frac, 0.1):.3f}  "
        f"p50 {quantile(clean_frac, 0.5):.3f}  p90 {quantile(clean_frac, 0.9):.3f}"
    )


def paired(a: dict, b: dict) -> None:
    """Median onset per problem, arm 1 vs arm 2, over the problems both ran."""

    def by_key(res):
        d: dict[str, list[float]] = {}
        for r in res["rows"]:
            d.setdefault(r["key"], []).append(r["onset"])
        return {k: st.median(v) for k, v in d.items()}

    da, db = by_key(a), by_key(b)
    common = sorted(set(da) & set(db), key=lambda k: (len(k), k))
    if not common:
        print("\npaired: no problem ids in common")
        return
    earlier = sum(1 for k in common if db[k] < da[k])
    ratio = st.median([da[k] / max(db[k], 1) for k in common])
    print(f"\npaired by problem ({len(common)} common)")
    print(f"  arm2 collapses EARLIER on {earlier}/{len(common)} problems")
    print(f"  median per-problem onset ratio arm1/arm2  {ratio:.2f}x")
    for k in common:
        mark = "<<" if db[k] < da[k] else "  "
        print(f"    id {k:>4}  {da[k]:8.0f} -> {db[k]:8.0f}  {mark}")


def main(argv: list[str]) -> int:
    args = argv[1:]
    do_pair = "--paired" in args
    args = [a for a in args if a != "--paired"]
    if not args:
        print(__doc__.strip().split("Usage")[-1], file=sys.stderr)
        return 2
    results = []
    for pattern in args:
        files = sorted(globmod.glob(pattern))
        label = pattern.split("/")[-1]
        if not files:
            print(f"{label}: no files matched")
            continue
        res = scan(files)
        report(label, res)
        results.append(res)
    if do_pair:
        if len(results) != 2:
            print("--paired needs exactly two globs that matched", file=sys.stderr)
            return 2
        paired(results[0], results[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
