"""Generate a standalone healthy control from the current task; maintainer only."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


ORACLE = """# Current-implementation assessment: healthy

The expected verdict is healthy. The supplied resolver consults the current
catalog for every record, requires a unique match for logical ID and SHA-256,
and checks that the backing file exists. The fixed reader independently checks
actual bytes, ID and difficulty. No code change is required.

The historical incident involved stale paths after a directory migration and
recovery/resampling that admitted only easy survivors. Those logs describe an
earlier resolver, not a defect in the current query. The model solved math;
sample-ID-to-file-path lookup belongs to application code.

Evidence for current health:

- The resolver matches ID and hash uniquely in current storage metadata.
- All 64 original training records remain readable with original content,
  including the 32 hard records at their current locations.
- The same query handles fresh unpredictable payload paths, a permuted catalog
  and wrong-byte decoys at stale paths without substituting any sample.
- Unknown IDs, wrong hashes on existing or missing paths, matching entries with
  absent files and ambiguous catalog matches are rejected.

The scene retains the frozen UltraData-RL-2609 Math selection: 64 training
records (32 easy + 32 hard) and 16 evaluation records (8 easy + 8 hard), with
original IDs, prompts, answers and hashes. Difficulty and historical accuracy
are author-supplied historical strata, not newly measured performance.

Do not award full credit merely for repeating the historical failure. A healthy
assessment needs evidence from the current implementation. Solution acceptance
also requires an explanation and independent fresh CPU behavior checks; it
ignores preplanted JSON and logs and does not judge prose semantics. The issue
judge separately assesses diagnosis. These finite checks do not prove correctness
for every future storage state or measure broad repository debugging ability.

No model, training, checkpoint download or GPU is needed. Historical GPU receipts
remain provenance only. No current full PPO success, reward improvement, GPU
calibration or blind LLM evaluation is claimed.
"""

README = """# Review a math data reader / 审查数学数据读取器

## 中文背景与验收

这是数据迁移题的健康对照，复用当前题包的中性题面、原始样本、历史日志、CPU
运行时与评分器，只改变起始 resolver 与维护者判定/说明。历史事故中，困难
样本迁移后旧路径失效，容错重采样让训练继续却只保留容易样本，导致奖励存在
幸存者偏差。当前查询已使用 catalog 唯一匹配 ID 与哈希，并检查文件存在。
模型负责解数学题；样本 ID 到文件路径的映射属于应用代码。

随包包含已核验历史场景中的冻结 UltraData-RL-2609 Math 数据：64 条训练样本
（32 easy + 32 hard）和 16 条评估样本（8 easy + 8 hard），原始 ID、题目、答案
与哈希不变。难度和历史准确率来自作者提供的历史分层，本次未重新测量。

正确判为 healthy 后可以原样提交，但仍须有解释并通过全部独立新运行的行为
检查：读取全部原始 64 条；将全部训练载荷移到新生成的不可预知目录、打乱
catalog、放置错误字节的旧路径诱饵，再核验同一批 ID 与内容；拒绝未知 ID、
已有/缺失路径上的错误哈希、匹配 catalog 但文件缺失及匹配不唯一的请求。
固定读取器核验 ID、实际字节哈希和难度，不能替换、漏读或修改样本。隐藏
verifier 不采信预先写入的成功 JSON 或历史日志。Solution 不语义评分解释
正文；issue judge 另行评审诊断。这是有限的数据契约题，不证明覆盖所有未来
存储状态或具备无提示的大仓库调试难度。

默认仅需 Python 标准库 CPU 检查，`gpus = 0`。Prepare 只复制随包环境和场景，
无需模型、训练、模型或数据下载，也无需 Hugging Face 托管。保留现有按 digest
固定的共享 miles 镜像；镜像可能较大，首次拉取可能需要网络。
`provenance/evaluate_submission.py` 可为外部 agent 已产出的工作区评分，需要
Docker daemon、固定镜像及尚不存在的新输出目录。Prepare 和 verifier 在不同
CPU 容器中运行，均禁用网络，verifier 输入只读；该 CLI 不启动 LLM agent。
上游默认 Harbor 自动 agent 工作流拒绝需要阶段隔离的任务，目前不支持本题。
本地 checker 示例只展示行为，不证明部署隔离。
[交付说明](provenance/DELIVERY.md)提供命令，[安全说明](provenance/SECURITY.md)
说明固定小样本、答案记忆、窄接口与可信 runner 的限制。旧 GPU 收据仅为历史
材料，不声称当前完整 PPO 成功、奖励改善、新 GPU 校准或 LLM 盲测。
可选历史完整快照可配合 `aa18e97` 与用户已有本地文件研究，无需另行托管。

## English background and acceptance

This healthy control reuses the neutral instructions, original samples,
historical logs, CPU runtime and verifier. Only the starting resolver and
maintainer verdict/description differ. In the historical incident, relocated
hard examples became unreadable through stale paths; recovery and resampling
let training proceed with easy survivors, biasing reward. The current query
already selects a unique catalog ID/hash match and checks file availability.
The model solves math; ID-to-file lookup belongs to application code.

The bundle preserves the verified historical scene's frozen UltraData-RL-2609
Math selection: 64 training records (32 easy + 32 hard) and 16 evaluation records
(8 easy + 8 hard), with original IDs, prompts, answers and hashes. Difficulty
and historical accuracy are author-supplied historical strata, not new results.

A healthy verdict with unchanged code is valid, but still requires an
explanation and every fresh behavioral check. The checker reads all original
64 training records, moves all payloads into fresh unpredictable directories,
permutes the catalog, plants wrong-byte stale-path decoys and checks the same
IDs and contents again. Probes reject unknown IDs, wrong hashes with existing
or missing paths, matching entries with missing files and ambiguous matches.
The fixed reader checks ID, actual byte hash and difficulty; no sample may be
replaced, omitted or modified. The hidden verifier reruns independently and
ignores preplanted success JSON or historical logs. Solution grading does not
assess prose semantics; the issue judge separately assesses diagnosis. These
finite checks neither exhaust future storage states nor establish broad,
unassisted repository debugging difficulty.

The default uses Python's standard library on CPU with `gpus = 0`. Prepare only
copies bundled environment and scene files. No model, training, model/data
download or Hugging Face hosting is needed. The existing shared miles image
remains digest-pinned and may still be large; its first pull may need network
access. `provenance/evaluate_submission.py` grades an external agent's existing
workspace and requires a running Docker daemon, the pinned image and a new
output directory. Prepare and verifier run in separate CPU containers with
networking disabled and read-only verifier inputs. The CLI does not launch an
LLM agent. The upstream default Harbor automatic agent workflow rejects tasks
requiring isolated stages and does not support this task. Local checker
demonstrations do not establish deployment isolation.
See [delivery instructions](provenance/DELIVERY.md) and
[security limits](provenance/SECURITY.md). Historical GPU receipts establish no
current full PPO success, reward improvement, new GPU calibration or blind LLM
evaluation. Optional historical snapshot work can use code at `aa18e97` with
the user's existing local files, without requiring hosting.
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
    validation['control_note'] = ('Generated from the current CPU data-contract task; unchanged healthy code still requires '
                                  'fresh original/relocated reads and negative probes. Inherited GPU receipts are historical only.')
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
