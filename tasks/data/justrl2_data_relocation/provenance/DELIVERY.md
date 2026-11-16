# Scene delivery / 场景交付

## 中文

本题复现一个人为注入的数据读取事故：JustRL-II 用 PPO 训练数学模型，
应用代码通过样本 ID 查找文件。困难样本的文件迁移后，旧映射仍指向原路径；
容错重采样使训练继续，却只保留容易样本，造成训练奖励虚高。
模型的任务始终是解数学题，文件路径映射不是模型的学习目标。

Agent 收到中性审查题面，先判断当前实现是否健康。历史日志仅作为调查线索。
若健康，可以保持代码不变；若不健康，修改受限的 `loader.py` 路径查询。
解法评分要求初始健康判定正确、报告有解释、不可用记录检查通过，并且固定
系统从 step 3 恢复 actor/critic 的模型、Adam、调度器和 RNG，完成 steps 4–7。
验收检查原始样本 ID、内容和每步 8 easy + 8 hard、真实更新、最终检查点和
固定评估；不要求奖励上涨。解法评分不语义评审报告正文，另有 issue 诊断评分。

### 已验证的文件

以下两个文件必须配套使用。tar 内有 88 个有效载荷文件，**不包含** `SCENE.json`。
独立清单补入解包目录后，完整场景是 89 个文件、66,739,559,968 字节。
这两份下载已完成全量 SHA-256 校验，且通过题包的场景校验。
这不等同于当前加固版本完成 GPU 端到端验收。

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `scene-v1-step3-b4886244.tar` | 66,739,527,680 | `601a3a94eb8a070ee4a15eaa5331c8ddd78046d289439db670fe3db7aa6eb420` |
| `SCENE.json` | 110,500 | `b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac` |

当前尚未核实可长期提供给评审的公共下载地址，不把临时签名链接写入仓库。
拿到上述两个文件后，从仓库根目录运行下方安装命令。
解包需要额外约 66.74 GB；运行时还会复制场景、准备模型并产生新的检查点，
不能只按压缩包大小规划磁盘。`SCENE.json` 的哈希是清单摘要，不是 tar 摘要。

### 安装 / Installation

Python 3.12+ is required. Both input files must be in `download_dir`.

```bash
download_dir="$HOME/Downloads"
datasets_dir="$HOME/autorlbench-datasets"
scene_dir="$datasets_dir/incidents/b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac"

python3 - "$download_dir" "$scene_dir" <<'PY'
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile

downloads, scene = map(Path, sys.argv[1:])
pins = {
    'scene-v1-step3-b4886244.tar': (66739527680, '601a3a94eb8a070ee4a15eaa5331c8ddd78046d289439db670fe3db7aa6eb420'),
    'SCENE.json': (110500, 'b48862441e71acd0bbbf6ad3f75c7168e6bd758255ad130e4f73c9ec1262e9ac'),
}
if scene.exists() or scene.is_symlink():
    raise FileExistsError(f'Refusing to overwrite {scene}; verify an existing scene separately')
for name, (size, expected) in pins.items():
    path = downloads / name
    if path.stat().st_size != size:
        raise ValueError(f'Wrong file size: {name}')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch: {name}')
scene.mkdir(parents=True)
with tarfile.open(downloads / 'scene-v1-step3-b4886244.tar') as archive:
    archive.extractall(scene, filter='data')
shutil.copyfile(downloads / 'SCENE.json', scene / 'SCENE.json')
print(scene)
PY

python3 -B tasks/data/justrl2_data_relocation/environment/incident_engine/scene.py verify "$scene_dir"
```

The expected verifier output includes `file_count: 88`, `checkpoint_iteration: 3`,
`resume_start: 4`, `final_step: 7`. The count excludes `SCENE.json` itself.

运行 harness 时传 `--datasets-dir "$datasets_dir"`，prepare 会自动读取
`incidents/<清单 SHA-256>/`。也可在 prepare 容器内设置 `RELOCATION_SCENE_SOURCE`
指向完整目录。**不要把上述缺少清单的 tar 直接传给该变量**；当前自动下载流程
只接受包含清单的完整归档。运行必须使用分离阶段的 Docker 后端，需三张 GPU。
上述安装和校验命令不启动训练。

### 健康对照

从当前题包生成独立健康对照，保留当前上游镜像、三卡配置和安全检查：

```bash
python3 -B tasks/data/justrl2_data_relocation/provenance/make_control.py /tmp/justrl2-control
python3 -m autorlbench validate /tmp/justrl2-control
```

生成器拒绝覆盖已有目录。对照和故障题的题面、历史日志、检查点及运行时一致，
只修改起始 resolver 与维护者判定/说明。生成目录与同一个 datasets 缓存配套使用。
`provenance/`、README、参考答案、隐藏判定不得作为 agent 可见输入。

## English

This task reproduces an intentionally injected data-reading incident. JustRL-II
trains a math model with PPO. Application code maps sample IDs to backing files.
After difficult examples move, the old mapping still points to missing paths.
Recovery resamples until training can continue on easy survivors, inflating the
conditional training reward. The model still solves math; file-location lookup
is not its learning objective.

The agent receives a neutral review instruction and assesses the current code.
Historical logs motivate investigation but do not establish a current defect.
Healthy code may be submitted unchanged; an unhealthy implementation needs a
repair through the bounded `loader.py` interface. Solution acceptance requires a
correct starting verdict, a report with an explanation, unavailable-record
checks, and fresh continuation from step 3 through steps 4–7. The fixed harness
checks model/Adam/scheduler/RNG restoration, original IDs/content, 8 easy plus
8 hard samples per step, real updates, final checkpoints and fixed evaluation.
Reward improvement is not required. Solution grading checks report presence,
not its semantic accuracy; the separate issue evaluation assesses diagnosis.

The table above pins both downloaded files. The tar contains 88 payload files
and omits `SCENE.json`; add the separately downloaded manifest at the extracted
scene root. Together they contain 89 files and 66,739,559,968 bytes. The full tar
hash, every payload hash and the task's scene verifier have been checked. This
does not constitute a new GPU validation of the hardened runtime. No durable
public download URL is verified yet; temporary signed URLs are not committed.

Use the shared installation block above from the repository root with Python
3.12+. It checks both hashes before extraction, refuses an existing destination,
adds the manifest, then validates all bytes and resume metadata. Allow another
66.74 GB for extraction; prepare copies the scene and also needs model assets
and fresh checkpoints. `file_count: 88` excludes the manifest itself. The
manifest hash is not the archive hash.

Pass `--datasets-dir "$datasets_dir"` to the harness, which locates the complete
scene under `incidents/<manifest SHA-256>/`. Alternatively set
`RELOCATION_SCENE_SOURCE` to that directory inside the prepare container. Do not
point it at the manifest-less tar: the automatic archive installer expects a
complete archive. Execution requires separate Docker stages and three GPUs;
the installation commands themselves do not train.

Generate the standalone healthy control with the `make_control.py` command
above. It uses the same instructions, historical observations, snapshot and
runtime; only starting resolver and maintainer verdict/description change.
Generation refuses an existing destination. Keep maintainer provenance,
README, oracle and expected verdict outside agent mounts. The control generator
does not imply an agent evaluation or a new healthy-control GPU run.
