# PTDBench Docker 环境设计

---

## 1. 要解决的问题

这是一个**以复现特定训练失败为评分标准**的 benchmark。这一句话决定了环境设计的取舍：

- 环境如果会漂，被复现的"ground truth"就会被悄悄改写，而且不会有任何东西报错。
- 因此"可复现"优先于"统一"、"镜像少"。
- 因此宁可**显式失败**，也不要**静默降级**。

同时，任务集横跨了多套互不兼容的训练栈（verl， areal， slime， miles 等）

---

## 2. 总体形态：环境相同的任务共享同一个镜像

`docker/images/` 下每个 `<name>.Dockerfile` 负责一个环境，可能被多个任务共同使用。

### 命名（tag）规范

tag 描述的是这套栈，多个任务共用，不从内容派生 hash。它**只编码「烤进镜像、单个任务在**`prepare.sh` **里换不动」的那一层**——编译型 / 绑 torch+CUDA ABI / 基础镜像层。能用`pip install --target` 在 prepare 阶段覆盖掉的纯 Python 包（`transformers`、`tokenizers`、`datasets`、`accelerateray` …，见本节下面「可覆盖的包不进 tag」）**不进 tag**；它们的精确版本一律交给 `FROM` 的 digest（+ 自建镜像的 requirements）去钉。

#### 固定槽位，缺位即不存在

tag 按**固定顺序**拼接；每个槽缺席不是「省略」，而是**该组件在镜像里确实不存在**：

```
[<框架>-<版本>-]<引擎><版本>-torch<版本>-flashattn<版本>-cu<版本>-py<版本>
```

| 槽 | 写法 | 缺位的含义 |
|---|---|---|
| 框架（领头） | `slime-v0.3.2`（烤进镜像）、`verl-`（family 标签，见下） | 框架不烤进镜像（随任务 workspace bind-mount 进来，如 ppo-critic 的 veRL） |
| 推理引擎 | `vllm0.11.0`、`sglang0.5.15` | 镜像不烤任何 rollout / inference 引擎 |
| torch | `torch2.11.0` | 无 torch |
| flash-attn | `flashattn2.8.3` | 镜像里**确实没装** flash-attn |
| CUDA | `cu128` | 无 CUDA（CPU 镜像） |
| python | `py3.12` | —（python 基本总在场，这一槽只承载信息、不承载缺位信号） |

领头的框架槽只在**框架确实烤进镜像**时写版本（`slime-v0.3.2`、`slime-v0.5.0rc0`）。
veRL 系列的 verl 框架其实不烤进镜像（`import verl` 在 base 里会失败，框架随 workspace
来），所以 `verl-` 只是一个给「这批 veRL 任务共用」的 family 标签、不带版本，后面照常接
真实的引擎 / torch / flashattn 槽。

例：

- `vllm0.11.0-torch2.8.0-flashattn2.8.3-cu128-py3.12` —— 框架领头缺位 = veRL 随 workspace 来。
- `verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12` —— 同样 veRL 随 workspace 来，只是
挂了个 `verl-` family 标签把它和上一行区分开。
- `slime-v0.3.2-sglang0.5.15-torch2.11.0-flashattn2.8.3-cu129-py3.12` —— 框架烤进镜像，领头是
`slime-v0.3.2`，torch / flashattn 如实列出（镜像里确实装了）。
- `torch2.13-py3.12` —— 无引擎、无 CUDA、无 flash-attn，三个缺位准确表达「CPU、不跑
rollout、无 FA」（rank-distillation 那类）。

格式约定：全小写、无 `+`。引擎 / `torch` / `flash-attn` 写**完整的 major.minor.patch**
（`vllm0.11.0`、`torch2.11.0`、`flashattn2.8.3`），只有 `.post1` / `.dev2` 这类纯发布后缀
交给 digest；但真正区分栈的预发布标记要保留（`slime-v0.5.0rc0` 就是因为 `0.5.0rc0 ≠ 0.5.15`
才分家，`flashattn2.7.4` 则是把 `2.7.4.post1` 的 `.post1` 丢给 digest）。CUDA 仍写 `cuNNN`
不写 `cuda12.9`，`py` 仍用 major.minor（`py3.12`）——这两槽不承诺到 patch。

#### 「缺位即不存在」是一个承诺——槽位必须反映镜像真实内容

因为缺位有语义，`flashattn` / `py` 这些槽写的是**镜像里实际存在的东西（含传递依赖）**，
不是「我在 Dockerfile 里显式装的」——upstream 的 verl / slime 镜像往往传递性地带了
flash-attn，你没显式装也得如实填上，否则「缺位 = 没装」就成了谎。建议配一条构建后一致性
检查（`import flash_attn` / `python --version` / `pip show vllm` 断言与 tag 吻合），顺带挡住
「换了 base 镜像忘改 tag」。

#### 可覆盖的包不进 tag

判据是**覆盖成本**，不是「影响不影响数值」：纯 Python / 装 wheel 即可的包，任务能在
`prepare.sh` 里 `pip install --target` + `prepared/env` 的 `PYTHONPATH` 自己换掉（见本节
下文第 4 段），镜像就不必替它们承诺，于是不进 tag。`transformers` 跨任务从 4.46 到 5.x
都有，但它正是这类——哪个任务要别的版本自己覆盖，不该逼出一个新镜像。

#### Dockerfile 的两种形态

无论 tag 怎么拼，每个 Dockerfile 都塌缩成「一行钉死 digest 的 `FROM`」，要么指向上游框架官方镜像，要么指向我们自建后推到仓库的预构建镜像（镜像本身有详细构建过程时，先构建好推上去，详细过程另行维护，不塞进运行时 Dockerfile）。

按这套 tag 命名还让「看着不同、其实同一套栈」的环境显形、便于合并——但是否真能合并，要靠在合并后的镜像上重跑任务、确认成绩不动来判断，不能只看 tag 像就合（见第 1 节「可复现优先于统一」）。

任务在 `task.toml` 里只写环境名：

```toml
[environment]
image = "verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12"
```

环境名就是 `docker/images/<name>.Dockerfile` 去掉 `.Dockerfile` 后缀的那一段

**为什么 slime 有两个环境**：不是新旧补丁关系，而是两套端到端不同的栈。retool /
search-r1 的 recipe 要 import `sglang.srt.constants.GPU_MEMORY_TYPE_CUDA_GRAPH`，
0.5.0rc0 里没有这个常量；而 rlve 四个任务的既有成绩是在 0.5.0rc0 上录的，往前挪就等于
换 torch 重写它们的 ground truth。谈不拢的两套栈就拿两个名字。

同理，一个镜像被多个任务共享，镜像内容就是这些任务共同的 ground truth 环境。改一个 Dockerfile
等于同时改所有用它的任务；需要不同依赖的任务应该开一个新环境名。只缺一两个纯 Python 包的任务
在 `environment/prepare.sh` 里 `pip install --target` 到 workspace，再通过 `prepared/env` 的
`PYTHONPATH` 交给后两个阶段（见 `tasks/reward_design/task_evaluator`）——prepare 容器对镜像
site-packages 的改动不会带到 agent / verifier 容器里。

---

## 3. 构建

### 按需构建

评测时 docker backend 先检查任务的镜像 `ptdbench:<name>` 是否已存在：存在就直接用，
不存在才用 `docker/images/<name>.Dockerfile` 构建（构建上下文是 `docker/`）。
prepare / agent / verifier 三个阶段用同一个镜像，各自起容器。

### 显式预构建：`ptdbench build-images`

```sh
python3 -m ptdbench build-images                  # docker/images/ 下所有缺失的镜像

# 只处理指定的
python3 -m ptdbench build-images \
    verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12 \
    slime-v0.5.0rc0-sglang0.5.0rc0-torch2.8.0-flashattn2.7.4-cu126-py3.12

# 已存在也重建
python3 -m ptdbench build-images --rebuild \
    verl-vllm0.17.0-torch2.10.0-flashattn2.8.3-cu129-py3.12
```

**tag 是固定的，不从内容派生。** 所以改了 Dockerfile（或它 COPY 进去的构建上下文资产，如
`miles-ngc2506-assets/`）之后，已有的镜像不会被自动发现过期，需要 `--rebuild`。
`ptdbench_fetch.py` 等抓取工具随任务走、不进构建上下文，改它们不涉及镜像。

### FROM 必须钉 digest

上游的 tag 会移动（`vllm017.latest` 这种尤其如此），而任务的录制结果不能跟着移动。所以
`FROM` 一律写成 `image:tag@sha256:...`，测试会检查每个 Dockerfile 都满足这一点。
换 digest 是一次有意识的升级：换完之后要把用这个环境的任务重跑一遍。

---

## 4. 运行时

### 镜像里不烤任何 PTDBench 基础设施

Dockerfile 就是一行钉死 digest 的 `FROM`，镜像内容就是上游框架镜像（或我们预构建的栈镜像）
本身，**没有任何 PTDBench 基础设施烤进去**。早期镜像末尾的「工具块」已撤销，它的各项职责
改由后端和任务自己承担：

- **挂载点**：`/ptdbench/{prepared,models,datasets}` 不再靠镜像 `mkdir -p` 预建 —— docker
  后端 bind-mount 时自动创建容器内路径，cybertron 后端在远端自建这三个符号链接，local
  后端直接用宿主机路径。
- **工作目录**：docker 后端起容器时显式传 `-w`，不依赖镜像的 `WORKDIR`。
- **镜像级环境变量**（`CUDA_DEVICE_ORDER`、`TOKENIZERS_PARALLELISM` 等）：不再有镜像级
  默认。哪个任务需要，就在自己的 `environment/prepare.sh` 里写进 `prepared/env`（见下面
  activate 一节），随任务走、不随镜像走。

抓取工具（`ptdbench_fetch.py` 等）和 activate 逻辑同样不烤进镜像：前者随各任务放在
`environment/` 里、prepare.sh 直接 `python3` 调（见「抓取工具」一节），后者见下。

例外是 miles-ngc2506 系列的自建重建 Dockerfile：它们要烤的是栈内容（fork 树、wheel），但末尾
仍保留着旧工具块字样，那段在新后端下已无运行时作用。

### activate 逻辑（运行时注入，不再是镜像里的文件）

以前这段装在镜像的 `/etc/profile.d/ptdbench-activate.sh`，靠 `sh -lc` 登录 shell 自动 source。
现在改为 harness 在运行时提供，单一来源是 `ptdbench/runtimes/base.py:PTDBENCH_ACTIVATE_BODY`：

- **docker / local 后端**：`CommandSpec.to_argv()` 用非登录 `sh -c`，把这段 body 拼在每条 shell
命令前面执行；
- **Harbor 后端**：prepare 把 body 写进 `prepared/ptdbench-activate.sh`，agent / verifier 的
非登录 `bash -c` 靠 `BASH_ENV` 指到它自动 source（cybertron 远端同理，另外显式导出代理与
`PTDBENCH_RESULTS`）。

因为都是非登录 shell，`/etc/profile` 不会重置 PATH，镜像 `ENV PATH` 原样生效，所以**不再需要**
以前那节"补回 PATH"的逻辑，`/etc/ptdbench-image-path` 也随之去掉。body 只做两件事：

1. **接上 prepare 的产物** —— 读 `$PTDBENCH_PREPARED/env` 的 `KEY=VALUE` 行。

   这个文件被当作**数据**逐行解析，不是 `.` 进 shell：它在 agent 和 verifier 的每条命令里都会被读，
    任务包不该能借此在所有命令里执行任意 shell。它放在 `$PTDBENCH_PREPARED` 而不是 workspace，
    因为 agent 能写 workspace，而 verifier 也读这个文件 —— 评分不能参考 agent 改得动的东西。
2. **给乱装的包找个落点（仅 agent / rerun 阶段）** —— 设 `PYTHONUSERBASE=/workspace/.ptdbench-user`、

   `PIP_USER=1`，以及 `PIP_BREAK_SYSTEM_PACKAGES=1`（Ubuntu 24.04 的系统解释器标了
    externally managed，不设的话连 `--user` 安装都会被拒）。agent 真去 `pip install` 的话
    落在自己的 workspace 里，不改镜像的 site-packages，verifier 也看不见。

### 三个阶段与挂载

| 阶段 | 干什么 | 网络 | `/ptdbench/{models,datasets}` |
|---|---|---|---|
| `prepare` | 跑任务的 `environment/prepare.sh`：下权重、下数据集、写 `prepared/env` | 需要 | 可写 |
| `agent` | agent 在 workspace 里干活 | 按 backend 策略 | 只读 |
| `verifier` | 评分 | — | 只读 |

每个阶段的离线相关变量（`HF_HUB_OFFLINE` 等）由 `SandboxRuntime._network_policy_env` 统一给出。

### 抓取工具（随任务放在 environment/ 里）

| 脚本 | 作用 |
|---|---|
| `ptdbench_fetch.py` | 整仓取权重/数据集到 `$PTDBENCH_{MODELS,DATASETS}` 下按 repo 名（`owner__repo`）命名的目录；目录存在即命中缓存，不再校验 sha256。repo 名会泄题的用 `--cache-as` 落到中性别名下（缓存在 cybertron 上对 agent 只读可见） |
| `ptdbench_dataset.py` + `ptdbench_datasets/<recipe>.py` | 跑数据集配方，处理"需要变换而非下载"的那类数据集；只下放到用到它的任务里 |

prepare.sh 直接 `python3 "$PTDBENCH_ENVIRONMENT/ptdbench_fetch.py" …` 调它们——跑在镜像自带的
解释器里，用那个镜像的 huggingface_hub / datasets。不再有 `install-ptdbench-tooling.sh`，
也不再把这些脚本烤进镜像或在运行时装进 `/usr/local`。

### 离线评测：`warm-cache.sh`

把下载从镜像构建挪到 prepare 阶段，代价是评测机要能联网 —— 对 airgapped 的 GPU 节点是笔烂账。
`./docker/warm-cache.sh [--models-dir DIR] [--datasets-dir DIR]` 在有网的机器上把每个任务的
`prepare.sh` 各跑一遍（缺镜像时先 `build-images`），填满两个缓存根目录；把它们搬到评测机上，
之后每个 prepare 阶段都是缓存命中。镜像本身用 `docker save` / `docker load` 搬过去。
它不产出可运行的 workspace：每个任务拿到的 `/workspace` 和 `prepared` 用完即弃，只有两个缓存活下来。
