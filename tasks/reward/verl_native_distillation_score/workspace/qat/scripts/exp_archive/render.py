"""Turn harvested records into ``QAT/experiments/``.

One directory per day, and **nothing below it except figures**::

    experiments/
      README.md              the overall story, hand-written on top of FACTS.md
      index.csv              one row per experiment, machine-readable
      report.html            self-contained: every figure inlined as base64
      figures/               cross-day overview figures
      code/                  snapshot of the scripts that produced all of this
      2026-09-08/
        README.md            what happened that day and what it meant
        FACTS.md             machine-generated ground truth the README rests on
        figures/*.png
        configs.json         every training config that day, keyed by job id
        kv_verification.json every rollout KV proof that day, keyed by job id
        evals.json           every evaluator payload that day, minus generations
        evals.csv            the same as a flat table
        curves-j12392.csv    per-step metrics, one file per substantive run
        POINTERS.md          absolute paths to everything not copied

An earlier version gave each job its own subdirectory.  That produced eleven
near-identically-named folders for a single day, ten of which were two-minute
crashes, and buried the two runs that mattered.  Jobs are not the unit a reader
cares about; a *day of work* is.  So runs that never reached
``_SUBSTANTIVE_STEPS`` steps get one line in a table and nothing else -- they
are real history, but they are footnotes, not chapters.

Nothing large is copied.  Generations (~2 GB) and checkpoints (~270 GB) stay on
/shared behind pointers.
"""

from __future__ import annotations

import base64
import csv
import datetime as _dt
import hashlib
import json
from pathlib import Path
import shutil

from .harvest import CURVE_KEYS, EvalRecord, HarvestError, TrainRecord

# Bulk keys: real results, but far too large to version and never read by eye.
_BULK_EVAL_KEYS = ("predictions", "data")

# Below this many logged steps a run is a smoke test, a config probe or a crash.
# 20 steps is roughly where the block-align loss and the OPD reward stop moving
# monotonically off their initialisation, i.e. where a curve starts saying
# something.
_SUBSTANTIVE_STEPS = 20

# The scripts that actually produce the experiments.  Snapshotted verbatim so
# the archive stays readable after the working tree moves on.  Listed rather
# than globbed for the same reason EVAL_ROOTS is: scripts/ also holds .bak
# files and one-off diagnostics that would only add noise.
_CODE_FILES = (
    "scripts/preflight_mopd.py",
    "scripts/run_mopd_verl.sh",
    "scripts/slurm_qat_train.sbatch",
    "scripts/slurm_qat_gate.sbatch",
    "scripts/eval_vllm_turboquant.py",
    "scripts/eval_aime.py",
    "scripts/eval_math_benchmarks.py",
    "scripts/eval_longbench_v2.py",
    "scripts/parity_int4_ptkh.py",
    "qat_kv/turboquant_fake_quant.py",
    "qat_kv/int4_ptkh_fake_quant.py",
    "qat_kv/turboquant_attn_hook.py",
    "qat_kv/tq_scheme.py",
)

# Curves worth a panel.  Missing keys are skipped: an off-policy arm has no
# pg_loss, and that absence is information rather than an error.
_PANELS = (
    ("actor/block_align_loss", "NMSE block-align loss"),
    ("actor/pg_loss", "OPD policy-gradient loss"),
    ("critic/score/mean", "top-k OPD reward"),
    ("val-topk/overlap_ratio", "top-k overlap vs teacher"),
    ("actor/grad_norm", "grad norm"),
    ("response_length/mean", "response length"),
)

# Config keys that distinguish one arm from another.  A day README that lists
# every one of the ~200 settings keys is as unreadable as one that lists none.
_ARM_KEYS = (
    "kv_scheme",
    "block_align_coef",
    "block_align_metric",
    "kv_align_coef",
    "reward_top_k",
    "reward_top_p_scope",
    "objective",
    "updates",
    "teacher_routing",
    "teacher_kv_cache_dtype",
)


def _fig_backend():
    """matplotlib, or a clear explanation of which interpreter to use.

    Deliberately not optional: an archive that silently skips its figures when
    the plotting library is absent looks complete and is not.
    """

    try:
        import matplotlib
    except ModuleNotFoundError as exc:  # pragma: no cover - environment issue
        raise HarvestError(
            "matplotlib is required to render the archive figures and is not "
            "installed in this interpreter.  Run the archive builder with the "
            "plotting venv, e.g. $QAT_ROOT/tools/plotenv/bin/python."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_substantive(rec: TrainRecord) -> bool:
    return (rec.last_step or 0) >= _SUBSTANTIVE_STEPS


def _arm_label(rec: TrainRecord) -> str:
    """The short human name of an arm, as we actually talk about it.

    Taken from the free-text tag the launcher bakes into the run directory
    (``...-k512-union-1p5b-nmseopd-pl512-j12392`` -> ``1p5b-nmseopd``), because
    that is the name these arms are referred to by in the logs and in
    conversation.  Falls back to the job id, never to a guess about what the
    arm was for.
    """

    name = rec.path.name
    for marker in ("-union-", "-only_stu-"):
        if marker in name:
            tail = name.split(marker, 1)[1]
            tail = tail.rsplit("-pl512-", 1)[0].rsplit(f"-j{rec.job_id}", 1)[0]
            if tail:
                return tail
    return f"j{rec.job_id}"


# --------------------------------------------------------------------------
# per-day data files


def _write_day_data(day_dir: Path, evals: list[EvalRecord], trains: list[TrainRecord]) -> None:
    substantive = [t for t in trains if _is_substantive(t)]

    if trains:
        (day_dir / "configs.json").write_text(
            json.dumps(
                {f"j{t.job_id}": t.settings for t in sorted(trains, key=lambda t: t.job_id)},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        proofs = {
            f"j{t.job_id}": t.kv_verification
            for t in sorted(trains, key=lambda t: t.job_id)
            if t.kv_verification is not None
        }
        if proofs:
            (day_dir / "kv_verification.json").write_text(
                json.dumps(proofs, indent=2, sort_keys=True) + "\n"
            )

    # Curves only for the runs whose curves say something.  A 2-step crash
    # produces a two-row CSV that is noise in a directory listing.
    for rec in substantive:
        keys = [k for k in CURVE_KEYS if rec.curves.get(k)]
        if not keys:
            continue
        steps = sorted({s for k in keys for s, _ in rec.curves[k]})
        table = {k: dict(rec.curves[k]) for k in keys}
        with (day_dir / f"curves-j{rec.job_id}.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", *keys])
            for step in steps:
                writer.writerow([step, *[table[k].get(step, "") for k in keys]])

    if evals:
        (day_dir / "evals.json").write_text(
            json.dumps(
                {
                    f"{e.group}/{e.label}": {
                        k: v for k, v in e.raw.items() if k not in _BULK_EVAL_KEYS
                    }
                    for e in sorted(evals, key=lambda e: (e.group, e.label))
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        with (day_dir / "evals.csv").open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "timestamp",
                    "benchmark",
                    "kv_scheme",
                    "accuracy",
                    "count",
                    "total_samples",
                    "n",
                    "status",
                    "source",
                ]
            )
            for e in sorted(evals, key=lambda e: (e.benchmark, -(e.accuracy or -1))):
                writer.writerow(
                    [
                        e.timestamp,
                        e.benchmark,
                        e.kv_scheme,
                        "" if e.accuracy is None else f"{e.accuracy:.6f}",
                        e.count,
                        e.total_samples,
                        e.n_samples,
                        e.status,
                        f"{e.group}/{e.label}",
                    ]
                )


def _write_day_pointers(day_dir: Path, evals: list[EvalRecord], trains: list[TrainRecord]) -> None:
    lines = [
        "# Pointers (not copied into the archive)",
        "",
        "Checkpoints, merged weights, swanlab media and per-sample generations "
        "stay on `/shared`.  Absolute paths:",
        "",
    ]
    if trains:
        lines.append("## Training runs")
        lines.append("")
        for t in sorted(trains, key=lambda t: t.job_id):
            lines.append(f"* **j{t.job_id}** run dir: `{t.path}`")
            if t.stdout:
                lines.append(f"  * stdout: `{t.stdout}`")
            if t.stderr:
                lines.append(f"  * stderr: `{t.stderr}`")
        lines.append("")
    if evals:
        lines.append("## Evaluations")
        lines.append("")
        for e in sorted(evals, key=lambda e: (e.group, e.label)):
            lines.append(f"* `{e.group}/{e.label}`: `{e.path}`")
        lines.append("")
    (day_dir / "POINTERS.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# per-day figures


def _plot_day_training(plt, out: Path, date: str, trains: list[TrainRecord]) -> Path | None:
    """One figure for the day: every substantive arm overlaid, per metric.

    Overlaid rather than one figure per job -- the question a reader has on a
    day with several arms is how they differ, and separate axes make that
    comparison impossible to do by eye.
    """

    arms = [t for t in trains if _is_substantive(t)]
    panels = [(k, t) for k, t in _PANELS if any(a.curves.get(k) for a in arms)]
    if not arms or not panels:
        return None
    cols = min(3, len(panels))
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.2 * rows), squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, (key, title) in zip(flat, panels):
        for arm in sorted(arms, key=lambda a: a.job_id):
            series = arm.curves.get(key)
            if not series:
                continue
            ax.plot(
                [s for s, _ in series],
                [v for _, v in series],
                lw=1.2,
                label=f"j{arm.job_id} {_arm_label(arm)}",
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    for ax in flat[len(panels) :]:
        ax.axis("off")
    flat[0].legend(fontsize=7, loc="best")
    fig.suptitle(f"training on {date}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / "training.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _plot_day_evals(plt, out: Path, date: str, evals: list[EvalRecord]) -> Path | None:
    scored = [e for e in evals if e.accuracy is not None]
    if len(scored) < 2:
        return None
    benchmarks = sorted({e.benchmark for e in scored})
    heights = [sum(1 for e in scored if e.benchmark == b) for b in benchmarks]
    fig, axes = plt.subplots(
        1,
        len(benchmarks),
        figsize=(max(5.0, 3.6 * len(benchmarks)), max(3.4, 0.28 * max(heights) + 1.6)),
        squeeze=False,
    )
    for ax, bench in zip(axes[0], benchmarks):
        rows = sorted(
            [e for e in scored if e.benchmark == bench], key=lambda e: e.accuracy, reverse=True
        )
        ax.barh(range(len(rows)), [e.accuracy for e in rows], color="#3b6ea5")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{e.kv_scheme} · {e.label}"[:52] for e in rows], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("accuracy", fontsize=8)
        ax.set_title(bench, fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"evaluations on {date}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "evals.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# cross-day figures


def _plot_scheme_overview(plt, out: Path, evals: list[EvalRecord]) -> Path | None:
    scored = [e for e in evals if e.accuracy is not None and e.benchmark == "aime_math"]
    if not scored:
        return None
    by_scheme: dict[str, list[float]] = {}
    for e in scored:
        by_scheme.setdefault(e.kv_scheme, []).append(e.accuracy)
    order = sorted(by_scheme, key=lambda k: max(by_scheme[k]), reverse=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(order) + 1.8))
    for i, scheme in enumerate(order):
        vals = by_scheme[scheme]
        ax.scatter(vals, [i] * len(vals), s=22, color="#3b6ea5", alpha=0.75)
        ax.scatter([max(vals)], [i], s=60, marker="|", color="#c1440e")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AIME accuracy (every recorded evaluation; red bar = best)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("AIME accuracy by KV scheme, all checkpoints and all dates", fontsize=11)
    fig.tight_layout()
    path = out / "aime_by_scheme.png"
    fig.savefig(path, dpi=130)
    fig.savefig(out / "aime_by_scheme.svg")
    plt.close(fig)
    return path


def _plot_arm_comparison(plt, out: Path, trains: list[TrainRecord]) -> Path | None:
    arms = [t for t in trains if _is_substantive(t)]
    if not arms:
        return None
    keys = [(k, t) for k, t in _PANELS if any(a.curves.get(k) for a in arms)][:4]
    if not keys:
        return None
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 3.6), squeeze=False)
    for ax, (key, title) in zip(axes[0], keys):
        for arm in sorted(arms, key=lambda a: a.job_id):
            series = arm.curves.get(key)
            if not series:
                continue
            ax.plot(
                [s for s, _ in series],
                [v for _, v in series],
                lw=1.1,
                label=f"j{arm.job_id} {arm.kv_scheme}",
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=6, loc="best")
    fig.suptitle(f"every training arm with ≥{_SUBSTANTIVE_STEPS} logged steps", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out / "arm_comparison.png"
    fig.savefig(path, dpi=130)
    fig.savefig(out / "arm_comparison.svg")
    plt.close(fig)
    return path


def _plot_timeline(plt, out: Path, days: dict[str, tuple[list, list]]) -> Path:
    dates = sorted(days)
    evals = [len(days[d][0]) for d in dates]
    trains = [len(days[d][1]) for d in dates]
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(dates)), 3.6))
    ax.bar(dates, trains, label="training jobs", color="#c1440e")
    ax.bar(dates, evals, bottom=trains, label="evaluations", color="#3b6ea5")
    ax.set_ylabel("experiments")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    ax.legend(fontsize=8)
    ax.set_title("experiments per day", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out / "timeline.png"
    fig.savefig(path, dpi=130)
    fig.savefig(out / "timeline.svg")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# FACTS.md -- the machine-generated ground truth each day README rests on


def _facts_md(date: str, evals: list[EvalRecord], trains: list[TrainRecord], figs: list[str]) -> str:
    substantive = sorted(
        (t for t in trains if _is_substantive(t)), key=lambda t: t.job_id
    )
    minor = sorted((t for t in trains if not _is_substantive(t)), key=lambda t: t.job_id)

    lines = [
        f"# {date} — facts",
        "",
        "Generated by `scripts/build_experiment_archive.py`; do not edit.  "
        "`README.md` is the human account written on top of this.",
        "",
        f"{len(trains)} training job(s) ({len(substantive)} substantive, {len(minor)} "
        f"smoke/aborted), {len(evals)} evaluation(s).",
        "",
    ]

    if substantive:
        lines += ["## Substantive training runs", ""]
        for t in substantive:
            proof = t.kv_verification or {}
            lines += [
                f"### j{t.job_id} — {_arm_label(t)}",
                "",
                f"* run dir: `{t.path.name}`",
                f"* started (config written): {t.timestamp}",
                f"* KV scheme: `{t.kv_scheme}`  ·  GPUs: {t.gpus}  ·  profile: `{t.profile}`",
                f"* steps logged: **{t.last_step}**"
                + (f" of {t.updates} planned" if t.updates else ""),
                "* rollout KV proof: "
                + (
                    "none recorded"
                    if not proof
                    else f"`rollout_kv_verified={proof.get('rollout_kv_verified')}`, "
                    f"runtime dtype `{proof.get('runtime_kv_cache_dtype')}`, "
                    f"skip layers {proof.get('runtime_skip_layers')}"
                ),
                "* arm config: "
                + ", ".join(
                    f"`{k}={t.settings.get(k)!r}`" for k in _ARM_KEYS if k in t.settings
                ),
                "",
            ]
            keys = [k for k in CURVE_KEYS if t.curves.get(k)]
            if keys:
                lines += ["| metric | first | last | step |", "| --- | --- | --- | --- |"]
                for key in keys:
                    series = t.curves[key]
                    lines.append(
                        f"| `{key}` | {series[0][1]:.6g} | {series[-1][1]:.6g} "
                        f"| {series[-1][0]} |"
                    )
                lines += ["", f"Full series: `curves-j{t.job_id}.csv`.", ""]

    if minor:
        lines += [
            "## Smoke tests, config probes and crashes",
            "",
            "Kept for the record; no curves are archived for these.",
            "",
            "| job | KV scheme | steps | run dir |",
            "| --- | --- | --- | --- |",
        ]
        for t in minor:
            lines.append(
                f"| j{t.job_id} | `{t.kv_scheme}` | "
                f"{t.last_step if t.last_step is not None else 'none'} | `{t.path.name}` |"
            )
        lines.append("")

    if evals:
        scored = [e for e in evals if e.accuracy is not None]
        failed = [e for e in evals if e.accuracy is None]
        lines += [
            "## Evaluations",
            "",
            f"{len(scored)} scored, {len(failed)} failed.  Full payloads in "
            "`evals.json`, flat table in `evals.csv`.",
            "",
            "| benchmark | KV scheme | accuracy | n | samples | source |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for e in sorted(scored, key=lambda e: (e.benchmark, -e.accuracy)):
            lines.append(
                f"| {e.benchmark} | `{e.kv_scheme}` | {e.accuracy:.4f} | {e.n_samples or '—'} "
                f"| {e.count}/{e.total_samples} | `{e.group}/{e.label}` |"
            )
        lines.append("")
        if failed:
            lines += ["Failed evaluations:", ""]
            for e in failed:
                err = str(e.raw.get("error", "")).splitlines()[:1]
                lines.append(
                    f"* `{e.group}/{e.label}` — {err[0] if err else 'no error recorded'}"
                )
            lines.append("")

    if figs:
        lines += ["## Figures", ""] + [f"* `figures/{f}`" for f in figs] + [""]
    lines += ["Pointers to everything not copied: `POINTERS.md`.", ""]
    return "\n".join(lines)


def _placeholder_readme(date: str, facts_present: bool) -> str:
    """Written only when no hand-authored README exists yet.

    Says plainly that it is unwritten rather than dressing FACTS.md up as an
    account -- a generated table that reads like analysis is worse than an
    admitted gap.
    """

    return (
        f"# {date}\n\n"
        "*(Narrative not yet written.)*\n\n"
        + ("See [`FACTS.md`](FACTS.md) for the machine-generated record of this day.\n"
           if facts_present else "")
    )


# --------------------------------------------------------------------------
# code snapshot / index / report


def _snapshot_code(repo_root: Path, dest: Path, git_rev: str) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rel in _CODE_FILES:
        src = repo_root / rel
        if not src.is_file():
            raise HarvestError(
                f"{src} is listed in the code snapshot but does not exist; either it "
                "was renamed (update _CODE_FILES) or the archive would ship an "
                "incomplete picture of how the experiments were run"
            )
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        manifest.append({"path": rel, "sha256": _sha256(src), "bytes": src.stat().st_size})
    (dest / "MANIFEST.json").write_text(
        json.dumps({"git_revision": git_rev, "files": manifest}, indent=2) + "\n"
    )
    (dest / "README.md").write_text(
        "# Code snapshot\n\n"
        f"Copied verbatim from the working tree at git revision `{git_rev}`.\n"
        "These are the scripts that launched, quantized and scored every experiment "
        "in this archive; `MANIFEST.json` carries a sha256 for each so a later "
        "reader can tell whether the live copy has since changed.\n\n"
        "This is a snapshot, not the source of truth — edit the real files under "
        "`QAT/scripts/` and `QAT/qat_kv/`, then rebuild the archive.\n"
    )


def _write_index_csv(path: Path, evals: list[EvalRecord], trains: list[TrainRecord]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "date",
                "timestamp",
                "kind",
                "id",
                "kv_scheme",
                "benchmark_or_arm",
                "accuracy_or_last_step",
                "status",
                "source_path",
            ]
        )
        for e in sorted(evals, key=lambda e: (e.date, e.timestamp)):
            writer.writerow(
                [
                    e.date,
                    e.timestamp,
                    "eval",
                    f"{e.group}/{e.label}",
                    e.kv_scheme,
                    e.benchmark,
                    "" if e.accuracy is None else f"{e.accuracy:.6f}",
                    e.status,
                    str(e.path),
                ]
            )
        for t in sorted(trains, key=lambda t: (t.date, t.timestamp)):
            writer.writerow(
                [
                    t.date,
                    t.timestamp,
                    "train",
                    f"j{t.job_id}",
                    t.kv_scheme,
                    _arm_label(t),
                    "" if t.last_step is None else t.last_step,
                    "substantive" if _is_substantive(t) else "smoke-or-aborted",
                    str(t.path),
                ]
            )


def _write_report_html(
    path: Path,
    days: dict[str, tuple[list[EvalRecord], list[TrainRecord]]],
    out_dir: Path,
    overview: list[Path],
    day_figs: dict[str, list[Path]],
    git_rev: str,
    built_at: str,
) -> None:
    def img(p: Path) -> str:
        data = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="{p.stem}">'

    def markdown_lite(text: str) -> str:
        """Enough markdown for the day narratives: headings, bold, code, lists.

        Not a general renderer -- the day READMEs are written to a known shape,
        and pulling in a markdown dependency for four constructs would be a
        worse trade than a hundred lines that handle exactly those four.
        """

        import html
        import re

        out: list[str] = []
        in_list = False
        in_table = False
        for raw in text.splitlines():
            line = html.escape(raw.rstrip())
            line = re.sub(r"`([^`]+)`", r"<code>\1</code>", line)
            line = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", line)
            if line.startswith("|") and line.endswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if set("".join(cells)) <= set("-: "):
                    continue
                if not in_table:
                    out.append("<table>")
                    in_table = True
                tag = "td"
                out.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
                continue
            if in_table:
                out.append("</table>")
                in_table = False
            if line.startswith("* "):
                if not in_list:
                    out.append("<ul>")
                    in_list = True
                out.append(f"<li>{line[2:]}</li>")
                continue
            if in_list:
                out.append("</ul>")
                in_list = False
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                out.append(f"<h{min(level + 1, 6)}>{line.lstrip('# ')}</h{min(level + 1, 6)}>")
            elif line.strip():
                out.append(f"<p>{line}</p>")
        if in_list:
            out.append("</ul>")
        if in_table:
            out.append("</table>")
        return "\n".join(out)

    total_e = sum(len(v[0]) for v in days.values())
    total_t = sum(len(v[1]) for v in days.values())
    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>QAT experiment archive</title><style>",
        "body{font:15px/1.6 -apple-system,Segoe UI,Helvetica,sans-serif;max-width:1000px;"
        "margin:2rem auto;padding:0 1rem;color:#222}",
        "h1{border-bottom:2px solid #333}h2{margin-top:3rem;border-bottom:1px solid #ccc}",
        "img{max-width:100%;border:1px solid #ddd;border-radius:4px;margin:.6rem 0}",
        "table{border-collapse:collapse;font-size:13px;margin:.6rem 0}",
        "td,th{border:1px solid #ddd;padding:3px 8px}th{background:#f2f2f2;text-align:left}",
        "code{background:#f4f4f4;padding:1px 4px;border-radius:3px;font-size:.92em}",
        ".meta{color:#666;font-size:12px}",
        "</style></head><body>",
        "<h1>QAT experiment archive</h1>",
        f"<p class='meta'>git <code>{git_rev}</code> · built {built_at} · "
        f"{total_t} training jobs · {total_e} evaluations · {len(days)} days</p>",
    ]
    top_readme = out_dir / "README.md"
    if top_readme.is_file():
        out.append(markdown_lite(top_readme.read_text()))
    out.append("<h2>Overview figures</h2>")
    out += [img(p) for p in overview]
    for date in sorted(days, reverse=True):
        out.append(f"<h2>{date}</h2>")
        readme = out_dir / date / "README.md"
        if readme.is_file():
            out.append(markdown_lite(readme.read_text()))
        out += [img(p) for p in day_figs.get(date, [])]
    out.append("</body></html>")
    path.write_text("\n".join(out))


# --------------------------------------------------------------------------


def render_archive(
    evals: list[EvalRecord],
    trains: list[TrainRecord],
    out_dir: Path,
    repo_root: Path,
    git_rev: str,
    keep_narratives: bool = True,
) -> dict:
    """Write the whole archive.  Returns a small summary for the CLI to print.

    ``keep_narratives`` preserves any hand-written ``README.md`` under a day
    directory across a rebuild.  Those are the one part of the archive a
    machine did not produce, so clobbering them on every rebuild would make the
    archive strictly worse each time it is regenerated.
    """

    plt = _fig_backend()

    preserved: dict[str, str] = {}
    if keep_narratives and out_dir.exists():
        for readme in sorted(out_dir.glob("*/README.md")):
            preserved[readme.parent.name] = readme.read_text()
        top = out_dir / "README.md"
        if top.is_file():
            preserved["."] = top.read_text()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    days: dict[str, tuple[list[EvalRecord], list[TrainRecord]]] = {}
    for e in evals:
        days.setdefault(e.date, ([], []))[0].append(e)
    for t in trains:
        days.setdefault(t.date, ([], []))[1].append(t)

    day_figs: dict[str, list[Path]] = {}
    for date, (day_evals, day_trains) in days.items():
        day_dir = out_dir / date
        day_dir.mkdir(parents=True, exist_ok=True)
        _write_day_data(day_dir, day_evals, day_trains)
        _write_day_pointers(day_dir, day_evals, day_trains)

        fig_dir = day_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        figs = [
            p
            for p in (
                _plot_day_training(plt, fig_dir, date, day_trains),
                _plot_day_evals(plt, fig_dir, date, day_evals),
            )
            if p is not None
        ]
        if figs:
            day_figs[date] = figs
        else:
            fig_dir.rmdir()

        (day_dir / "FACTS.md").write_text(
            _facts_md(date, day_evals, day_trains, [p.name for p in figs])
        )
        (day_dir / "README.md").write_text(
            preserved.get(date) or _placeholder_readme(date, True)
        )

    overview_dir = out_dir / "figures"
    overview_dir.mkdir()
    overview = [
        p
        for p in (
            _plot_timeline(plt, overview_dir, days),
            _plot_scheme_overview(plt, overview_dir, evals),
            _plot_arm_comparison(plt, overview_dir, trains),
        )
        if p is not None
    ]

    _snapshot_code(repo_root, out_dir / "code", git_rev)
    _write_index_csv(out_dir / "index.csv", evals, trains)
    built_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if "." in preserved:
        (out_dir / "README.md").write_text(preserved["."])
    _write_report_html(
        out_dir / "report.html", days, out_dir, overview, day_figs, git_rev, built_at
    )

    return {
        "days": len(days),
        "evals": len(evals),
        "trains": len(trains),
        "figures": len(overview) + sum(len(v) for v in day_figs.values()),
        "narratives_preserved": len([k for k in preserved if k != "."]),
        "out_dir": str(out_dir),
    }
