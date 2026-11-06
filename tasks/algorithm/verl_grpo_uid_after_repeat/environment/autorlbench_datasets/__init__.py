"""把上游资产转成任务要的形状：verl 格式的 parquet，或者拼好的检索索引。

为什么需要这一步：任务的 launch 脚本要 `/workspace/<name>_verl/{train,test}.parquet`，
而 verl 的 `RLHFDataset` 要求每行有

    prompt                    chat messages 列表（不是字符串）
    data_source               精确字符串，决定路由到哪个打分器
    reward_model.ground_truth 真值

（见 workspace/verl/utils/dataset/rl_dataset.py:313 和
  workspace/verl/utils/reward_score/__init__.py:127,142）

上游的原始格式直接喂进去会在 `example["prompt"]` 上崩，所以"换成标准数据集"=
标准数据集 + 这里这份确定性配方。`search_r1_nq_hotpotqa` / `wiki18_e5` 是同一类
事情的另外两个形状：前者把两个 QA 集按 Search-R1 的模板合成 parquet，后者把上游
切成两块的 FAISS 索引 `cat` 回一个能读的文件。共同点都是"下载来的字节 ≠ 任务要的字节"。

## 为什么叫 autorlbench_datasets

这里的配方要 `from datasets import load_dataset`（HF 那个库）。包一旦叫 `datasets`，
它自己就会把那个库遮掉，而报错会发生在 `load_dataset` 上，离原因很远。

这个包随它所服务的任务一起放在该任务的 `environment/` 里，和 `fetch_hf.py` /
`autorlbench_dataset.py` 做邻居；`autorlbench_dataset.py` 把本目录加进 sys.path 后
`import autorlbench_datasets.<recipe>` 才解析得到。

## 两种调用方式

容器内（prepare.sh 走这条，带缓存，见 autorlbench_dataset.py）：

    python3 "$AUTORLBENCH_ENVIRONMENT/autorlbench_dataset.py" \\
        --recipe dapo_math_17k --dest "$AUTORLBENCH_WORKSPACE/DAPO-Math-17k_verl"

宿主机上手工跑一份：

    python3 -m autorlbench_datasets.dapo_math_17k --out-dir /some/where/DAPO-Math-17k_verl

运行环境：需要 `datasets` + `pandas` + `pyarrow` + `huggingface_hub`，并且要能联网
访问 HF。verl / slime 的环境镜像里都有；宿主机自带的 `/usr/bin/python3` 没有。
`wiki18_e5` 还要 ~86GB 的临时磁盘，见那份配方的 docstring。第一次跑请人盯着，
对一下打印出来的行数和 schema。
"""
