#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build two public JustRL2 trees with identical instrumentation and one defect delta."""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

REVISION = "75814318f243af7e39af337c1173fc3927c9635b"
HERE = Path(__file__).resolve().parent
def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError("Pinned source anchor missing or ambiguous: " + before[:90])
    return text.replace(before, after, 1)


def common_instrumentation(root):
    actor = root / "miles/backends/megatron_utils/actor.py"
    text = "import os\n" + actor.read_text()
    text = 'from benchmark.process_group_compat import install as install_process_group_compat\n' + text
    text = replace_once(text, '        monkey_patch_torch_dist()\n',
                        '        monkey_patch_torch_dist()\n        install_process_group_compat()\n')
    text = replace_once(text, "        super().init(args, role, with_ref)\n", '''        trace_prefix = os.environ.get("TORCH_NCCL_DEBUG_INFO_TEMP_FILE")
        if trace_prefix:
            os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = trace_prefix + role + "_"
        super().init(args, role, with_ref)
''')
    text = replace_once(text, "        init(args)\n", "        init(args)\n        args.bench_role = role\n")
    actor.write_text(text)
    loss = root / "miles/backends/training_utils/loss.py"
    text = loss.read_text()
    text = "from benchmark.runtime_evidence import emit\n" + text
    # Remove answer-revealing repair commentary from BOTH sides, without changing code.
    begin = text.index("        # Collective-communication invariant: distributed_masked_whiten")
    end = text.index("        assert (", begin)
    text = text[:begin] + text[end:]
    needle = '        _diag_logger.info(\n            "[adv-diag] rank=%s enter whiten adv_numel=%s mask_numel=%s",'
    record = '''        emit(args, "whitening_enter", local_numel=all_advs.numel(),
             mask_sum=float(all_masks.sum().item()),
             response_lengths=response_lengths, total_lengths=total_lengths,
             local_advantages=all_advs.detach().float().cpu().tolist(),
             local_masks=all_masks.detach().float().cpu().tolist())
'''
    text = replace_once(text, needle, record + needle)
    text = replace_once(text, "        whitened_advs_flat = distributed_masked_whiten(\n",
                        '        emit(args, "whitening_call", local_numel=all_advs.numel())\n'
                        "        whitened_advs_flat = distributed_masked_whiten(\n")
    done = '        _diag_logger.info("[adv-diag] rank=%s whiten done", dist.get_rank())\n'
    text = replace_once(text, done, done + '''        emit(args, "whitening_complete", local_numel=whitened_advs_flat.numel(),
             normalized=whitened_advs_flat.detach().float().cpu().tolist())
''')
    text = replace_once(text, '    rollout_data["advantages"] = advantages\n',
                        '    emit(args, "advantages_ready", local_numel=sum(x.numel() for x in advantages))\n'
                        '    rollout_data["advantages"] = advantages\n')
    loss.write_text(text)
    model = root / "miles/backends/megatron_utils/model.py"
    text = "from benchmark.runtime_evidence import snapshot_optimizer, after_optimizer\n" + model.read_text()
    target = "        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()\n"
    text = replace_once(text, target, '''        bench_before = snapshot_optimizer(optimizer)
''' + target + '''        after_optimizer(args, optimizer, bench_before, update_successful,
                        grad_norm, rollout_id, step_id)
''')
    model.write_text(text)


def inject(text):
    start = text.index('        emit(args, "whitening_call",', text.index("def compute_advantages_and_returns"))
    end_marker = "        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))\n"
    end = text.index(end_marker, start) + len(end_marker)
    block = text[start:end]
    return text[:start] + "        if all_masks.numel() > 0:\n" + "".join("    " + line if line.strip() else line for line in block.splitlines(keepends=True)) + text[end:]


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def build(source, output):
    source = source.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if (source / ".git").exists():
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != REVISION:
            raise ValueError("Source revision differs from public pin")
        archive = output / "source.tar"
        subprocess.run(["git", "-C", str(source), "archive", "--format=tar", "--output", str(archive), REVISION], check=True)
        with tarfile.open(archive) as tar:
            tar.extractall(output / "healthy", filter="data")
        archive.unlink()
    else:
        shutil.copytree(source, output / "healthy", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    healthy = output / "healthy"
    raw = inventory(healthy)
    pins = json.loads((HERE / "source_files.json").read_text())
    if raw != pins["files"]:
        raise ValueError("Source file inventory differs from pinned public revision")
    helper = healthy / "benchmark"
    helper.mkdir()
    (helper / "__init__.py").write_text("")
    for name in ["runtime_evidence.py", "run_training.py", "train.sh", "replay_loader.py", "sglang_compat.py", "process_group_compat.py"]:
        shutil.copyfile(HERE / name, helper / name)
    common_instrumentation(healthy)
    shutil.copytree(healthy, output / "faulty")
    target = Path("miles/backends/training_utils/loss.py")
    faulty_file = output / "faulty" / target
    faulty_file.write_text(inject(faulty_file.read_text()))
    for root in [healthy, output / "faulty"]:
        for path in root.rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path.relative_to(root)))
    hi, fi = inventory(healthy), inventory(output / "faulty")
    changed = [name for name in hi if hi[name] != fi[name]]
    if changed != [str(target)] or set(hi) != set(fi):
        raise ValueError("Pair differs beyond the intended defect")
    before, after = (healthy / target).read_text(), faulty_file.read_text()
    diff = "".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                         fromfile="healthy/" + str(target), tofile="faulty/" + str(target)))
    (output / "defect.patch").write_text(diff)
    manifest = {"public_source": "https://github.com/merak0514/JustRL-II", "revision": REVISION,
                "case_kind": "controlled_regression_on_public_code", "training_validated": False,
                "different_files": changed, "healthy_files": hi, "faulty_files": fi,
                "common_modified_files": [name for name in raw if raw[name] != hi[name]]}
    (output / "pair_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["revision", "different_files", "common_modified_files", "training_validated"]}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output.resolve())
