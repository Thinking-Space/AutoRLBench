# Scene delivery / 场景交付

## 中文

默认交付已包含 CPU 数据契约题所需的场景与运行代码，无需下载模型、数据或
检查点，也无需 Hugging Face 托管。Prepare 只复制随包环境与场景；原约
66.7 GB 完整训练快照不再是前置条件。题包自带的提交物评分 CLI 使用现有按
digest 固定的共享 miles 镜像、CPU 和独立 Docker 阶段，`gpus = 0`。
镜像本身可能较大；首次拉取镜像可能需要网络。

数据来自已核验的历史场景与冻结的 UltraData-RL-2609 Math 选择集。随包保存
原始 64 条训练样本（32 easy + 32 hard）和 16 条评估样本（8 easy + 8 hard），
保留 ID、题目、答案与内容哈希。样本选择及哈希记录见
`provenance/selection.json`，数据来源锁定信息见同目录
`provenance/sources.lock.json`，当前场景摘要见 `environment/snapshot.json`。
场景载荷共 69 个文件、141,589 字节（不含运行时、日志及容器镜像）。
难度标签及历史准确率是作者提供的历史分层，本次未重新运行模型或测量难度。

历史事故中，困难样本迁移后旧路径失效，容错过滤和重采样让训练继续，却使批次
只剩容易样本。当前验收聚焦这条读入契约：中性题面要求判断当前实现；健康实现
无需修改。隐藏 verifier 要求初始判定正确、报告有解释，并独立执行全量读取与
负例检查；不语义评分报告正文，issue 诊断 judge 另行运行。

`check_reads.py` 先检查全部原始 64 条训练样本，再将它们移到新生成的不可预知
目录，打乱 catalog，并在旧路径写入错误字节的诱饵，重新检查同一批 ID 与内容。
负例覆盖未知 ID、已有/缺失路径上的错误哈希、匹配 catalog 但文件缺失和
匹配不唯一。固定读取器核对 ID、字节哈希及难度；禁止替换、漏读、修改样本。
隐藏 verifier 独立新运行，不以预先写好的 JSON 或历史日志评分。

## Isolated submission grading / 隔离提交物评分

Use Python 3.11+ for repository commands (`tomllib` is required); the shared
image provides Python 3.12. 仓库命令需要 Python 3.11+；固定镜像使用 Python 3.12。

上游默认 Harbor 自动 agent 工作流会拒绝需要阶段隔离的任务，目前不支持本题。
可由外部 agent 产出工作区，再使用题包内的 `provenance/evaluate_submission.py`
评分。该命令只评分已有提交物，不启动 LLM agent，也不运行独立的 issue judge。

提交目录至少应包含 `loader.py` 和 `report.md`；报告须按题面给出修改前判定和
解释。`--output` 必须指向尚不存在的新目录。需要本机 Docker CLI、正在运行的
Docker daemon，以及 `task.toml` 指定的固定镜像。首次获取镜像可能需要网络，
但不下载模型、检查点或数据。Prepare 和 verifier 在不同 CPU 容器中运行，均
使用 `--network none`；verifier 只读挂载提交物、准备数据和可信评分代码。

The upstream default Harbor automatic agent workflow rejects tasks requiring
isolated stages and does not currently support this task. An external agent can
produce a workspace for the task-local `provenance/evaluate_submission.py` CLI
to grade. This command evaluates an existing submission; it does not launch an
LLM agent or run the separate issue judge.

The submission directory must contain `loader.py` and `report.md`, including the
starting-state verdict and explanation required by the instruction. `--output`
must name a new directory that does not exist. You need the Docker CLI, a running
Docker daemon and the digest-pinned image specified in `task.toml`. Obtaining
the image initially may require network access; no model, checkpoint or dataset
download is needed. Prepare and verifier run in separate CPU containers, both
with `--network none`. The verifier mounts the submission, prepared data and
trusted grading code read-only.

The CLI uses `--pull never`. Pull the pinned image once before grading (network
may be required); both evaluation containers themselves remain offline.
CLI 使用 `--pull never`。首次评分前手动拉取固定镜像；这一步可能联网，评分
容器本身保持断网。

```bash
docker pull ghcr.io/merak0514/autorlbench-justrl2@sha256:8d2eb1fe292a4e381915734b7854213002ca9bae6b50efab1bd8cc75cad14a8b
```

From the repository root / 从仓库根目录运行：

```bash
python3 tasks/data/justrl2_data_relocation/provenance/evaluate_submission.py \
  --submission /absolute/path/to/workspace \
  --output /absolute/path/to/new-results
```

Replace both paths with your submitted workspace and a fresh results location.
The CLI grades the solution contract described above. Local checker commands
below demonstrate behavior only; they do not establish Docker isolation or an
automated agent success rate. `autorlbench validate` checks package structure,
not completion of an upstream automatic evaluation workflow.

请将两个路径替换为实际提交工作区和新的结果目录。CLI 评分范围是上文的解法
契约。下方本地检查仅展示行为，不证明 Docker 隔离或自动 agent 成功率。
`autorlbench validate` 检查题包结构，不代表上游自动评测工作流已运行成功。

## Commands shared by both languages / 共用命令

Run validation from the repository root / 从仓库根目录验证题包：

```bash
python3 -m autorlbench validate tasks/data/justrl2_data_relocation
```

Run this CPU demonstration from the task root with the maintainer reference
query. It checks repaired reads without preparing a model or launching training.
`workspace/run.sh` checks the current loader when invoked as `./run.sh` from
the prepared workspace with `AUTORLBENCH_PREPARED` set. The faulty starting
loader is expected to fail its data contract.

从任务目录运行以下 CPU 命令，可用维护者参考查询演示修复后的读取。
在准备好的工作区内设置 `AUTORLBENCH_PREPARED` 后运行 `./run.sh`，则检查当前
loader；故障起始版本预期不能通过。

```bash
cd tasks/data/justrl2_data_relocation
python3 -I -S environment/incident_engine/check_reads.py \
  --scene environment/scene \
  --loader evaluations/solution/oracle/reference_loader.py \
  --snapshot environment/snapshot.json
```

The submission evaluator supplies `AUTORLBENCH_ENVIRONMENT`,
`AUTORLBENCH_WORKSPACE` and `AUTORLBENCH_PREPARED` to `environment/prepare.sh`;
all three are required.
For a local prepare check, use a separate workspace copy and an empty prepared
directory. From the repository root:

提交物评分器会给 `environment/prepare.sh` 提供上述三个必需变量。手动检查 prepare
时使用独立工作区副本和空的 prepared 目录，从仓库根目录运行：

```bash
task_dir="$PWD/tasks/data/justrl2_data_relocation"
prepare_check_dir="$(mktemp -d)"
cp -R "$task_dir/workspace" "$prepare_check_dir/workspace"
mkdir "$prepare_check_dir/prepared"
AUTORLBENCH_ENVIRONMENT="$task_dir/environment" \
AUTORLBENCH_WORKSPACE="$prepare_check_dir/workspace" \
AUTORLBENCH_PREPARED="$prepare_check_dir/prepared" \
  bash "$task_dir/environment/prepare.sh"
```

These local data checks do not establish isolation. Use the standalone
submission evaluator above for separate Docker prepare and verifier stages.
这些本地数据检查不证明隔离；需要独立 Docker prepare/verifier 阶段时，使用
上文的提交物评分器。

### Healthy control / 健康对照

From the repository root, choose a destination that does not already exist:
从仓库根目录运行，目标目录必须尚不存在：

```bash
python3 -B tasks/data/justrl2_data_relocation/provenance/make_control.py /tmp/justrl2-control
python3 -m autorlbench validate /tmp/justrl2-control
```

生成器复用当前题包的中性题面、原始样本、历史日志、CPU 检查器、固定镜像和
Docker 隔离要求，只改变起始 resolver 与维护者判定/说明。健康对照可原样提交，
但必须通过相同的新读取检查。README、`provenance/`、oracle 和隐藏判定均属
维护者材料，不应放入 agent 可见输入。生成题包不等于开展了 agent 评测。

The generator reuses the neutral instructions, original samples, historical
logs, CPU checker, pinned image and Docker isolation requirements. Only the
starting resolver and maintainer verdict/description change. An unchanged
healthy submission must pass the same fresh read checks. Keep README,
`provenance/`, oracles and hidden verdicts outside agent inputs. Generating a
control is not an agent evaluation.

## English

The default package includes the scene and runtime needed for CPU data-contract
debugging. No model, dataset or checkpoint download, and no Hugging Face hosting,
is needed. Prepare only copies bundled environment and scene files. The former
approximately 66.7 GB training snapshot is no longer a prerequisite. The
standalone submission evaluator uses the existing digest-pinned shared miles
image and separate CPU Docker stages with `gpus = 0`. The image may still be
large, and its initial pull may require network access.

The data comes from the verified historical scene and a frozen
UltraData-RL-2609 Math selection. The bundle preserves the original 64 training
records (32 easy + 32 hard) and 16 evaluation records (8 easy + 8 hard), including
IDs, prompts, answers and content hashes. Selection and hash records are in
`provenance/selection.json`; source pins are in the
adjacent `provenance/sources.lock.json`, and the current scene pin is in
`environment/snapshot.json`. Scene payloads total 69 files and 141,589 bytes,
excluding runtime code, logs and the container image. Difficulty and historical
accuracy values are author-supplied historical strata, not newly measured results.

In the historical incident, difficult examples moved while old paths remained
in use. Recovery and resampling let training continue with only easy survivors.
The present task tests that read contract. Neutral instructions ask the agent
to assess the current implementation; healthy code may remain unchanged. The
hidden verifier requires a correct starting verdict, an explanation and all
behavior checks. It does not grade prose semantics; the issue judge remains a
separate evaluation.

`check_reads.py` reads all 64 original training records, then moves their
payloads into fresh unpredictable directories in a temporary copy, permutes
the catalog, plants wrong-byte decoys at stale paths and checks the same IDs
and contents again. Negative probes cover unknown IDs, wrong hashes with
existing/missing paths, matching catalog entries with missing files and
ambiguous matches. A fixed reader checks identity, byte hashes and difficulty;
samples cannot be replaced, omitted or modified. The hidden verifier performs
an independent fresh run and does not grade preplanted JSON or historical logs.
Use the shared commands above to validate, inspect repaired reads or prepare a
separate local copy without training.

## Optional historical snapshot / 可选历史完整快照

历史 GPU 收据保留在 `provenance/receipts/`，仅记录当时的运行，不证明当前版本
完成完整 PPO 端到端验收、奖励改善、新 GPU 校准或 LLM 盲测。历史续训代码在
提交 `aa18e97`；若需复查，可使用用户已持有的本地完整快照与配套文件。
默认 CPU 验收不读取该快照，也不要求上传、托管或再次申请导出。
旧快照元数据中的 `model_files`、模型 revision 与 checkpoint 字段不是当前
prepare 或评分的模型文件依赖。

Historical GPU receipts remain in `provenance/receipts/`. They do not establish
current full PPO end-to-end success, reward improvement, new GPU calibration
or blind LLM testing. Historical continuation code is available at `aa18e97`
for optional use with the user's existing local full snapshot and companion
files. Default CPU acceptance neither reads that snapshot nor requires its
upload, hosting or another export request. Historical snapshot metadata such
as `model_files`, model revisions and checkpoint fields does not create a
current prepare or grading dependency on model files.
