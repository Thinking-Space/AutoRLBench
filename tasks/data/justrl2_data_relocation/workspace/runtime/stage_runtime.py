"""Stage pinned public source and instrument real optimizer updates only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

REVISION = '75814318f243af7e39af337c1173fc3927c9635b'
ARCHIVE_SHA = '09e7548e98bb3285526eb7582d06b5e671165cec7fab98ae7876081def5ed9b5'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Pinned source anchor missing or ambiguous: ' + before[:80])
    return text.replace(before, after, 1)


def instrument(root):
    actor = root / 'miles/backends/megatron_utils/actor.py'
    text = actor.read_text()
    text = 'from benchmark.process_group_compat import install as install_process_group_compat\n' + text
    text = replace_once(text, '        monkey_patch_torch_dist()\n',
                        '        monkey_patch_torch_dist()\n        install_process_group_compat()\n')
    text = replace_once(text, '        init(args)\n', '        init(args)\n        args.bench_role = role\n')
    actor.write_text(text)
    model = root / 'miles/backends/megatron_utils/model.py'
    text = 'from benchmark.runtime_evidence import snapshot_optimizer, after_optimizer\n' + model.read_text()
    target = '        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()\n'
    text = replace_once(text, target,
                        '        bench_before = snapshot_optimizer(optimizer)\n' + target +
                        '        after_optimizer(args, optimizer, bench_before, update_successful,\n'
                        '                        grad_norm, rollout_id, step_id)\n')
    model.write_text(text)


def upgrade_weight_api(root):
    """Honor the pinned image's explicit session protocol, including finalize.

    The pinned image implements begin(selector,sync_base)
    and end() semantics. End performs the old post_process_weights lifecycle.
    This adapter applies only to the distributed-transfer path used here.
    """
    engine = root / 'miles/backends/sglang_utils/sglang_engine.py'
    text = engine.read_text()
    anchor = '    def pause_generation(self, mode: str = "retract"):\n'
    methods = '''    def begin_weight_update(self):
        return self._make_request("begin_weight_update", {"selector": "all", "sync_base": True})

    def end_weight_update(self):
        return self._make_request("end_weight_update", {})

'''
    engine.write_text(replace_once(text, anchor, methods + anchor))
    mixin = root / 'miles/backends/megatron_utils/update_weight/update_weight_from_distributed/mixin.py'
    text = mixin.read_text()
    replacements = {
        '_pause_and_prepare_engines': '''    def _pause_and_prepare_engines(self) -> None:
        if dist.get_rank() == 0:
            mode = self.args.pause_generation_mode
            ray.get([engine.pause_generation.remote(mode=mode) for engine in self.rollout_engines])
            if mode != "in_place":
                ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            ray.get([engine.begin_weight_update.remote() for engine in self.rollout_engines])
''',
        '_finalize_and_resume_engines': '''    def _finalize_and_resume_engines(self, post_load_weights: bool = False) -> None:
        if dist.get_rank() == 0:
            ray.get([engine.end_weight_update.remote() for engine in self.rollout_engines])
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
''',
    }
    lines = text.splitlines(keepends=True)
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name in replacements]
    if len(nodes) != len(replacements):
        raise ValueError('Pinned weight updater structure changed')
    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        lines[node.lineno - 1:node.end_lineno] = [replacements[node.name]]
    text = ''.join(lines)
    ast.parse(text)
    mixin.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-archive', type=Path, required=True)
    ap.add_argument('--helpers', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if sha(args.source_archive) != ARCHIVE_SHA:
        raise ValueError('Public source archive SHA mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.source_archive) as tar:
        tar.extractall(args.output, filter='data')
    root = args.output / ('JustRL-II-' + REVISION)
    pins = json.loads((args.helpers / 'source_files.json').read_text())['files']
    actual = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
    if actual != pins:
        raise ValueError('Extracted public source inventory differs from pinned revision')
    helper = root / 'benchmark'
    helper.mkdir()
    (helper / '__init__.py').write_text('')
    for name in ['runtime_evidence.py', 'process_group_compat.py', 'sglang_compat.py']:
        shutil.copyfile(args.helpers / name, helper / name)
    instrument(root)
    upgrade_weight_api(root)
    for name, target in [('Megatron-LM', '/root/Megatron-LM'), ('sglang', '/sgl-workspace/sglang')]:
        if not Path(target).is_dir():
            raise ValueError('Pinned image framework missing: ' + target)
        (root / name).symlink_to(target, target_is_directory=True)
    receipt = {'revision': REVISION, 'source_archive_sha256': ARCHIVE_SHA,
               'original_files': pins,
               'instrumented_files': {name: sha(root / name) for name in pins},
               'helpers': {p.name: sha(p) for p in helper.iterdir()}}
    (args.output / 'source_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(root, flush=True)


if __name__ == '__main__':
    main()
