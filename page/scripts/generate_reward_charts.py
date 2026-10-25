#!/usr/bin/env python3
"""
Generate one wandb-style reward curve (SVG) per task into page/images/rewards/.

Each chart is a single line panel in the wandb idiom: the metric name rotated
on the y axis, hairline horizontal gridlines only, a step axis, and a 2px line
over a 10% area wash with an end marker. No title, no caption, no numeric
annotations - the page copy around the figure carries the context. Every chart is built from that
task's own committed run artifacts under tasks/<domain>/<id>/:

  algorithm       workspace/logs/metrics_*.jsonl (issue run)
  reward          workspace/outputs/metrics.json eval points
  data            "step:N - k:v - ..." lines inside workspace/logs/*.log
  environment     workspace/results/baseline32/metrics_summary.json
  infrastructure  workspace/outputs/metrics.json eval points
  configuration   workspace/autorlbench_case/original_eval/notes/n32_epoch_breakdown.txt

Run from anywhere:  python3 page/scripts/generate_reward_charts.py
Output:             page/images/rewards/<domain>-<task>.svg
"""

import glob
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "page" / "images" / "rewards"

# Series colors and inks are the validated reference palette (checked against
# this surface #fbfaf8).
BLUE = "#2a78d6"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fbfaf8"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

W, H = 560, 280  # logical viewBox; the page scales it to the figure column

ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------- utilities

def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def nice_ticks(lo, hi, n=4):
    """Clean tick values covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = 10 * mag
    for mult in (1, 2, 2.5, 5, 10):
        if mult * mag >= raw:
            step = mult * mag
            break
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + 1e-9:
        if v >= lo - 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks, step


def fmt_tick(v, step):
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    return f"{v:.{decimals}f}"


def fmt_val(v):
    a = abs(v)
    if a >= 1000:
        s = f"{v / 1000:.1f}K"
    elif a >= 100:
        s = f"{v:.0f}"
    elif a >= 1:
        s = f"{v:.2f}"
    else:
        s = f"{v:.3f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def parse_step_lines(path):
    """Yield (step, {key: value}) from verl trainer logs' `step:N - k:v` lines."""
    for line in open(path, errors="replace"):
        line = ANSI.sub("", line)
        m = re.search(r"step:(\d+) - (.*)$", line)
        if not m:
            continue
        data = {}
        for kv in m.group(2).split(" - "):
            if ":" in kv:
                key, val = kv.split(":", 1)
                try:
                    data[key] = float(val)
                except ValueError:
                    pass
        if data:
            yield int(m.group(1)), data


def read_jsonl_series(paths, key):
    steps, vals = [], []
    for path in paths:
        for line in open(path):
            rec = json.loads(line)
            if key in rec["data"]:
                steps.append(rec["step"])
                vals.append(rec["data"][key])
    return steps, vals


# ---------------------------------------------------------------- rendering

def render_chart(spec):
    """A wandb-style line chart. spec: series[{steps,vals,color,name}], y_label,
    x_label. One series needs no legend (the y axis names it); two get one."""
    series = spec["series"]
    L, R = 60, W - 18
    # the bottom stack (ticks at B+14, axis label at B+30) must fit inside
    # the viewBox with descenders, so B sits 42px above the edge
    T, B = (34 if len(series) > 1 else 18), H - 42
    out = []

    # the metric name, rotated along the y axis
    ymid = (T + B) / 2
    out.append(f'<text x="13" y="{ymid:.1f}" font-family="{FONT}" font-size="11" '
               f'fill="{INK2}" letter-spacing="0.2" text-anchor="middle" '
               f'dominant-baseline="middle" '
               f'transform="rotate(-90 13 {ymid:.1f})">{esc(spec["y_label"])}</text>')

    xmin = min(s["steps"][0] for s in series)
    xmax = max(s["steps"][-1] for s in series)
    all_vals = [v for s in series for v in s["vals"]]
    ymin, ymax = min(all_vals), max(all_vals)
    span = (ymax - ymin) or abs(ymax) or 1.0
    lo = ymin - span * 0.06  # wandb auto-ranges to the data, not to zero
    hi = ymax + span * 0.06

    # a denser grid: ~7 y-gridlines, every other one carrying a label, so the
    # hairlines multiply without the tick text getting crowded
    yticks, ystep = nice_ticks(lo, hi, 7)
    for i, t in enumerate(yticks):
        py = B - (t - lo) / (hi - lo) * (B - T)
        out.append(f'<line x1="{L:.1f}" y1="{py:.1f}" x2="{R:.1f}" y2="{py:.1f}" '
                   f'stroke="{GRID}" stroke-width="1"/>')
        if i % 2 == 0:
            out.append(f'<text x="{L - 8}" y="{py:.1f}" font-family="{FONT}" font-size="10" '
                       f'fill="{MUTED}" text-anchor="end" dominant-baseline="middle">'
                       f'{esc(fmt_tick(t, ystep))}</text>')
    out.append(f'<line x1="{L:.1f}" y1="{B:.1f}" x2="{R:.1f}" y2="{B:.1f}" '
               f'stroke="{AXIS}" stroke-width="1"/>')

    for t in nice_ticks(xmin, xmax, 6)[0]:
        px = L + (t - xmin) / (xmax - xmin) * (R - L)
        out.append(f'<text x="{px:.1f}" y="{B + 14}" font-family="{FONT}" '
                   f'font-size="10" fill="{MUTED}" text-anchor="middle" '
                   f'dominant-baseline="hanging">{esc(fmt_val(t))}</text>')
    if spec.get("x_label"):
        out.append(f'<text x="{(L + R) / 2:.1f}" y="{B + 30}" font-family="{FONT}" '
                   f'font-size="10" fill="{MUTED}" text-anchor="middle" '
                   f'dominant-baseline="hanging">{esc(spec["x_label"])}</text>')

    if len(series) > 1:  # legend: line-key + name, right-aligned on the top row
        cx = R
        for s in reversed(series):
            label_w = 6.3 * len(s["name"])
            out.append(f'<line x1="{cx - 16 - label_w:.1f}" y1="18" x2="{cx - 5 - label_w:.1f}" '
                       f'y2="18" stroke="{s["color"]}" stroke-width="2" '
                       f'stroke-linecap="round"/>')
            out.append(f'<text x="{cx - label_w:.1f}" y="18" font-family="{FONT}" '
                       f'font-size="11" fill="{INK2}" dominant-baseline="middle" '
                       f'text-anchor="start">{esc(s["name"])}</text>')
            cx -= 16 + label_w + 12

    def px_of(v):
        return L + (v - xmin) / (xmax - xmin) * (R - L)

    def py_of(v):
        return B - (v - lo) / (hi - lo) * (B - T)

    for s in series:
        if len(s["steps"]) > 1:
            # a soft wash under the line (series hue at 10% opacity) grounds
            # the curve without adding data ink
            area = (f'M{px_of(s["steps"][0]):.1f},{B:.1f} '
                    + " ".join(f'L{px_of(st):.1f},{py_of(v):.1f}' for st, v in zip(s["steps"], s["vals"]))
                    + f' L{px_of(s["steps"][-1]):.1f},{B:.1f} Z')
            out.append(f'<path d="{area}" fill="{s["color"]}" fill-opacity="0.1"/>')
        pts = " ".join(f"{px_of(st):.1f},{py_of(v):.1f}" for st, v in zip(s["steps"], s["vals"]))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" '
                   f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
        if len(s["steps"]) <= 12:  # sparse eval schedules get per-point markers
            for st, v in zip(s["steps"], s["vals"]):
                out.append(f'<circle cx="{px_of(st):.1f}" cy="{py_of(v):.1f}" r="4.5" '
                           f'fill="{s["color"]}" stroke="{SURFACE}" stroke-width="2"/>')
        lx, ly = px_of(s["steps"][-1]), py_of(s["vals"][-1])  # end marker only
        out.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="4.5" '
                   f'fill="{s["color"]}" stroke="{SURFACE}" stroke-width="2"/>')

    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
        f'aria-labelledby="d">',
        f'<desc id="d">{esc(spec["desc"])}</desc>',
        f'<rect width="{W}" height="{H}" rx="10" fill="{SURFACE}"/>',
        "".join(out),
        "</svg>",
    ]) + "\n"


# ------------------------------------------------------------- task specs

def task_dir(task_id):
    return ROOT / "tasks" / task_id


def spec_algorithm():
    d = task_dir("algorithm/verl_grpo_uid_after_repeat")
    key = "critic/rewards/mean"
    steps, vals = read_jsonl_series(glob.glob(str(d / "workspace/logs/metrics_*.jsonl")), key)
    return {
        "y_label": key,
        "series": [
            {"steps": steps, "vals": vals, "color": BLUE, "name": "issue run"},
        ],
        "x_label": "Step",
        "desc": "Per-step mean training reward of the recorded issue run. The "
                "reward collapses where the repeat penalty starts double-counting "
                "samples.",
    }


def spec_reward():
    d = task_dir("reward/slime_retool_length_reward_bonus")
    pts = json.load(open(d / "workspace/outputs/metrics.json"))["metrics"]["eval_points"]
    steps = [p["step"] for p in pts]
    return {
        "y_label": "eval/aime",
        "series": [{"steps": steps, "vals": [p["eval_score"] for p in pts],
                    "color": BLUE, "name": "logged score"}],
        "x_label": "Step",
        "desc": "The run's logged eval score at each checkpoint. It includes the "
                "unconditional length bonus, so it rises while generations lengthen "
                "rather than while answers get more correct.",
    }


def spec_data():
    d = task_dir("data/verl_taco_difficulty_mismatch")
    log = glob.glob(str(d / "workspace/logs/*.log"))[0]
    steps, rewards = [], []
    for st, data in parse_step_lines(log):
        if "critic/score/mean" in data:
            steps.append(st)
            rewards.append(data["critic/score/mean"])
    return {
        "y_label": "critic/score/mean",
        "series": [{"steps": steps, "vals": rewards, "color": BLUE, "name": "reward"}],
        "x_label": "Step",
        "desc": "Per-step mean training reward of the recorded run on the TACO "
                "split. Reward stays flat because the difficulty mismatch leaves "
                "the model with nothing learnable at this level.",
    }


def spec_environment():
    d = task_dir("environment/search_r1_zero_query_embedding")
    pts = json.load(open(d / "workspace/results/baseline32/metrics_summary.json"))["points"]
    steps = [p["step"] for p in pts]
    return {
        "y_label": "val/em",
        "series": [{"steps": steps, "vals": [p["val/em"] for p in pts],
                    "color": BLUE, "name": "EM"}],
        "x_label": "Step",
        "desc": "Exact match on the fixed held-out question set, evaluated at the "
                "recorded checkpoints of the run.",
    }


def spec_infrastructure():
    d = task_dir("infrastructure/verl_missing_flash_attn_patch")
    pts = json.load(open(d / "workspace/outputs/metrics.json"))["metrics"]["eval_points"]
    steps = [p["step"] for p in pts]
    key = pts[0].get("eval_key", "eval_score")
    return {
        "y_label": key,
        "series": [{"steps": steps, "vals": [p["eval_score"] for p in pts],
                    "color": BLUE, "name": "accuracy"}],
        "x_label": "Step",
        "desc": "Held-out validation accuracy at the recorded eval points of the "
                "same run.",
    }


def spec_configuration():
    d = task_dir("configuration/verl_aime25_single_sample_protocol")
    txt = open(d / "workspace/autorlbench_case/original_eval/notes/n32_epoch_breakdown.txt").read()
    epochs = [(int(a), float(b)) for a, b in
              re.findall(r"epoch=\s*(\d+)\s+correct=\d+/30\s+accuracy=([\d.]+)", txt)]
    return {
        "y_label": "accuracy",
        "series": [{"steps": [e for e, _ in epochs], "vals": [v for _, v in epochs],
                    "color": BLUE, "name": "per-repeat accuracy"}],
        "x_label": "evaluation",
        "desc": "Accuracy of each of the 32 evaluation repeats of the same "
                "checkpoint - the spread a single-sample protocol hides.",
    }


SPECS = [
    ("algorithm/verl_grpo_uid_after_repeat", spec_algorithm),
    ("reward/slime_retool_length_reward_bonus", spec_reward),
    ("data/verl_taco_difficulty_mismatch", spec_data),
    ("environment/search_r1_zero_query_embedding", spec_environment),
    ("infrastructure/verl_missing_flash_attn_patch", spec_infrastructure),
    ("configuration/verl_aime25_single_sample_protocol", spec_configuration),
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task_id, builder in SPECS:
        spec = builder()
        svg = render_chart(spec)
        out = OUT_DIR / f"{task_id.replace('/', '-')}.svg"
        out.write_text(svg)
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
