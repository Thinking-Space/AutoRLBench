"""Generate a standalone healthy control from the current task; maintainer only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


ORACLE = """# Current-implementation assessment: healthy

The expected verdict is healthy. The supplied resolver consults current storage
metadata for every record, matching both the logical ID and content hash, and
checks that the chosen backing file exists. The outer reader checks actual
content bytes, sample identity and difficulty. No code change is required.

Essential reasons:
- Current code uses catalog.json to recover current locations, including moved
  hard examples, instead of treating the original manifest path as authoritative.
- Historical missing-file errors describe an earlier resolver. They do not show
  that the supplied resolver is defective. The model solves math; it does not
  learn the application code's sample-ID-to-file-path mapping.

Evidence items:
- The supplied resolver matches both ID and SHA-256 in current catalog rows.
- All 64 original training records remain readable with original content through
  this resolver, including the 32 hard records at their relocated paths.
- Historical logs show zero admitted hard samples after relocation, but the
  current resolver recovers those files; a training-reward increase alone is
  not a current-health check.

Do not award full credit merely for repeating the historical failure. A correct
healthy assessment needs evidence about the current implementation. The solution
evaluation additionally requires fresh continuation; no new control GPU run or
blind agent evaluation is claimed by this generated package.
"""

README = """# Review a math training experiment / 审查数学训练实验

This is the healthy control generated from the data-relocation task. It uses the
same neutral instructions, step-3 checkpoint, historical logs, runtime and
verifier. Only the starting resolver and maintainer verdict differ. A correct
healthy report with unchanged code must still pass fresh steps 4–7 continuation.
The model solves math; sample-ID-to-path lookup is application code.

这是数据迁移题的健康对照，复用同一中性题面、step-3 检查点、历史日志、
运行时和评分器，只更换初始 resolver 与维护者判定。正确判断 healthy 后可以
原样提交，但仍须通过 steps 4–7 的真实续训验收。模型负责解数学题，
样本 ID 到文件路径的映射属于应用代码。

Use the Docker backend and the pinned dataset cache described in
[delivery instructions](provenance/DELIVERY.md). The copied runtime includes the
current upstream image digest, three-GPU allocation and isolated-stage checks.
See [security limits](provenance/SECURITY.md). Original GPU receipts are inherited
evidence from before hardening, not a new healthy-control GPU run. No blind LLM
evaluation or public snapshot hosting is implied by generation.

使用 Docker 后端，按[交付说明](provenance/DELIVERY.md)安装固定快照；运行时
保留上游镜像摘要、三卡配置及阶段隔离要求。[安全说明](provenance/SECURITY.md)
记录剩余限制。原 GPU 收据是加固前的历史证据，不是健康对照的新 GPU 验收；
生成题包不代表已开展 LLM 盲测，也不代表快照已有公共下载地址。
"""


def make_control(destination):
    source = Path(__file__).resolve().parents[1]
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('Control destination already exists: ' + str(destination))
    destination = destination.resolve()
    if destination.is_relative_to(source):
        raise ValueError('Control must be outside the source task')
    expected_path = Path('evaluations/solution/verifier/resources/expected_verdict.json')
    if json.loads((source / expected_path).read_text())['expected_verdict'] != 'unhealthy':
        raise ValueError('Generate the control from the original unhealthy package')
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    loader = (source / 'evaluations/solution/oracle/reference_loader.py').read_text()
    loader = loader.replace('Maintainer reference repair; kept outside the answer workspace.',
                            'Resolve a requested record to its current backing-file path.')
    (destination / 'workspace/loader.py').write_text(loader)
    (destination / expected_path).write_text(json.dumps(
        {'schema_version': 1, 'expected_verdict': 'healthy'}, indent=2) + '\n')
    (destination / 'evaluations/issue/oracle.md').write_text(ORACLE)
    (destination / 'README.md').write_text(README)
    metadata = (destination / 'task.toml').read_text()
    values = {
        'title': 'Review a math training experiment',
        'failure_type': 'healthy_control',
        'failure_reason': 'Current code resolves original samples using current storage metadata; '
                          'historical failed reads do not establish a current defect.',
    }
    for key, value in values.items():
        metadata, count = re.subn(r'^' + key + r' = .*$',
                                  key + ' = ' + json.dumps(value), metadata, flags=re.MULTILINE)
        if count != 1:
            raise ValueError('Expected exactly one metadata field: ' + key)
    (destination / 'task.toml').write_text(metadata)
    validation_path = destination / 'provenance/validation.json'
    validation = json.loads(validation_path.read_text())
    validation['expected_starting_verdict'] = 'healthy'
    validation['control_note'] = 'Generated from the current task. Inherited GPU receipts do not validate this variant.'
    validation_path.write_text(json.dumps(validation, indent=2) + '\n')
    inventory = destination / 'provenance/source_manifest.json'
    files = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(destination.rglob('*')) if p.is_file() and p != inventory}
    inventory.write_text(json.dumps({'schema_version': 3,
        'scope': 'Generated control files, excluding this inventory', 'files': files}, indent=2) + '\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    print(make_control(parser.parse_args().destination))
