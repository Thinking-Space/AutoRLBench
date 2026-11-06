"""BytedTsinghua-SIA/DAPO-Math-17k -> DAPO-Math-17k_verl/{train,test}.parquet

verl_grpo_uid_after_repeat 用的这一份产物（launch 脚本要
/workspace/DAPO-Math-17k_verl/{train,test}.parquet）。

## 为什么需要一条自己定的 split 规则

上游是**单个 split、不带 train/test**。Hub 上的 parquet 有 1791700 行，但是同一份
17917 条题首尾相接复制了 100 遍（作者在讨论区确认过这是意外重复，不是 179 万道题）。
历史产物是 16125 / 1792（= 17917，比例 10.0017%），但那次划分的 seed 和边界**任务里
任何地方都没有记录**，`agent_trace.json` 翻遍了也没有。所以复刻不出来，只能先按
行 id 去掉这 100 份副本，再定一条新的、写下来的划分规则。

## 规则（v1）

    key    = extra_info.original_index 或 extra_info.index
               上游实际字段名是 index，值是 UUID，行内唯一且与行序无关。
               original_index 是早先按历史产物猜的名字，真有这个字段时仍优先用它。
    digest = sha256(b"ptdbench-dapo-split-v1:" + key)
    test   当 int(digest[:16], 16) % 10 == 0，否则 train

选哈希而不选 `random.shuffle(seed=42)` 是因为：哈希划分**与行序无关、与库版本无关**。
`datasets.train_test_split` 和 `numpy.random.permutation` 的结果都会随版本漂移，
而"同样的上游 + 同样的脚本 = 同样的字节"是这份资产存在的全部意义。
salt 里的 `v1` 是留给以后改规则用的——改了就换 v2，别就地改语义。
（salt 字符串是从 PTDBench 原样继承的：换掉它，切出来的 train/test 就变了，
grader 的基线对不上。）

期望切出来 ~10% 作为 test。**这不会等于历史的 16125/1792**，所以 grader_spec.json
里的 historical_final_baseline / required_final_score 必须重跑重算。
（这是已知且已授权的后果，不是 bug。）

## 上游 schema

上游应当已经是 verl 格式（data_source='math_dapo'，style='rule-lighteval/MATH_v2'），
这一点取自历史产物里真实 dump 出来的 schema。**但本机无网，没能亲自核对上游**——
所以下面对每个必需字段都做了显式检查，对不上就报错停下，而不是猜着往下走。

用法：
    python3 -m autorlbench_datasets.dapo_math_17k --out-dir ./prepared/DAPO-Math-17k_verl
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from ._common import hf_datasets_cache, write_split

REPO = "BytedTsinghua-SIA/DAPO-Math-17k"
# 钉死 revision。这个仓的 main 曾经、而且仍然可能再被改；行数断言挡得住"行数变了"，
# 挡不住"17917 个 id 还在，题面被换掉"。
REVISION = "65877096c24ffa7abc4e4fa5edb95cf3413a5674"
DATA_SOURCE = "math_dapo"  # reward_score/__init__.py:142 精确匹配这个串

SPLIT_SALT = b"ptdbench-dapo-split-v1:"
TEST_MODULUS = 10  # 1/10 进 test

# 去重之后的行数，不是 parquet 的物理行数。物理行数是这个数乘 100。
EXPECTED_TOTAL = 17917  # 历史产物 16125 + 1792；对不上说明上游变了

# 缓存键的一部分（见 environment/autorlbench_dataset.py）。**改了配方就要改这个数**：
# 划分规则、schema 检查、写盘方式，任何一处变了，上一次缓存下来的 parquet 就不再是
# 这份配方的产物。不改的话，先跑的那个任务会一直拿到旧字节，而"同样的配方
# 产出同样的数据"正是这些文件存在的理由。
# v2：先按 extra_info.index 去掉 100 倍复制，再用这个 UUID 做划分键。v1 把
# 1791700 行原样拿去对 17917，并且找不到 original_index 时退回题面文本。
RECIPE_VERSION = "2"


def split_of(key: str) -> str:
    digest = hashlib.sha256(SPLIT_SALT + key.encode("utf-8")).hexdigest()
    return "test" if int(digest[:16], 16) % TEST_MODULUS == 0 else "train"


def row_id(row: dict) -> str | None:
    """上游用来标识一道题的 UUID。

    文件里的字段名是 extra_info.index。original_index 是按历史产物猜的名字，
    这个 revision 上并不存在；两边都认，是为了上游改名时不要静默退回题面。
    """
    extra = row.get("extra_info") or {}
    key = extra.get("original_index") or extra.get("index")
    return str(key) if key else None


def row_fingerprint(row: dict) -> bytes:
    """同一 id 的副本必须逐字段相同。只盖住划分和打分会读的那些字段。"""
    prompt = row.get("prompt") or [{}]
    content = ""
    if prompt and isinstance(prompt[0], dict):
        content = prompt[0].get("content") or ""
    reward = row.get("reward_model") or {}
    digest = hashlib.sha256()
    for part in (
        row.get("data_source") or "",
        row.get("ability") or "",
        reward.get("ground_truth") or "",
        reward.get("style") or "",
        content,
    ):
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return digest.digest()


def dedupe_copies(rows_iter) -> tuple[list[dict], int]:
    """丢掉 100 倍复制，保留每个 id 第一次出现的那一行。

    顺序就是第一份副本的顺序。同一 id 如果出现内容不同的副本，停下来：
    留第一份会把"上游改了一部分副本"吞掉。
    """
    kept: list[dict] = []
    seen: dict[str, bytes] = {}
    total = 0
    for row in rows_iter:
        total += 1
        key = row_id(row)
        if not key:
            raise SystemExit(
                "上游行没有 extra_info.index / original_index，无法去掉 100 倍复制。"
            )
        fingerprint = row_fingerprint(row)
        previous = seen.get(key)
        if previous is None:
            seen[key] = fingerprint
            kept.append(row)
        elif previous != fingerprint:
            raise SystemExit(
                f"extra_info.index={key} 的重复副本内容不一致，不能只留第一份。"
            )
    return kept, total


def split_key(row: dict, fallback_used: list[bool]) -> str:
    """取一个稳定的行标识。

    首选上游的 UUID。万一上游没有这个字段，退回到题面文本——它同样与行序无关，
    只是万一上游有重复题面，两条会被分到同一边。退化时会在最后报出来，不静默。
    """
    key = row_id(row)
    if key:
        return key
    fallback_used[0] = True
    prompt = row["prompt"]
    return prompt[0]["content"]


def check_schema(row: dict) -> None:
    if row.get("data_source") != DATA_SOURCE:
        raise SystemExit(
            f"上游 data_source 是 {row.get('data_source')!r}，期望 {DATA_SOURCE!r}。\n"
            f"打分器按这个串精确路由（reward_score/__init__.py:142），不能改。"
        )
    prompt = row.get("prompt")
    if not isinstance(prompt, list) or not prompt or "content" not in prompt[0]:
        raise SystemExit(
            f"上游 prompt 不是 chat messages 列表：{type(prompt).__name__}。\n"
            f"RLHFDataset 要求 example['prompt'] 是 list（rl_dataset.py:313）。"
        )
    rm = row.get("reward_model") or {}
    if "ground_truth" not in rm:
        raise SystemExit(
            "上游 reward_model 里没有 ground_truth"
            "（reward_manager/dapo.py:92 直接取这个键）。"
        )


def build(out_dir: Path, *, allow_count_mismatch: bool = False) -> list[Path]:
    """下载上游、按规则划分、写出两个 parquet；返回写出的路径。

    和 main() 分开是因为容器里的 `autorlbench_dataset` 要在**缓存目录**里调用它，
    然后再把产物硬链到任务要的位置——它需要的是返回值，不是一个进程退出码。
    """
    # 在 import 之前：见 _common.hf_cache_root()。cache_dir 不传的话上游会下到
    # ~/.cache/huggingface，那是容器可写层，宿主机 datasets/ 里什么都不会出现。
    cache_dir = hf_datasets_cache()
    from datasets import load_dataset

    print(f"下载 {REPO}@{REVISION[:12]} ...")
    dataset = load_dataset(REPO, cache_dir=str(cache_dir), revision=REVISION)
    if len(dataset.keys()) != 1:
        print(f"!! 上游有多个 split：{list(dataset.keys())}，全部合并后重新划分")
    raw = (dict(record) for split in dataset for record in dataset[split])
    rows, raw_count = dedupe_copies(raw)

    if not rows:
        raise SystemExit("上游一行都没有")
    check_schema(rows[0])
    if raw_count != len(rows):
        print(f"  去掉重复副本：{raw_count} 行 -> {len(rows)} 行")

    if len(rows) != EXPECTED_TOTAL:
        msg = (
            f"去重后 {len(rows)} 行（原始 {raw_count} 行），"
            f"历史产物合计 {EXPECTED_TOTAL} 行"
        )
        if not allow_count_mismatch:
            raise SystemExit(f"{msg}。上游变了，确认后加 --allow-count-mismatch。")
        print(f"!! {msg}（--allow-count-mismatch 已开，继续）")

    fallback_used = [False]
    buckets: dict[str, list[dict]] = {"train": [], "test": []}
    for row in rows:
        buckets[split_of(split_key(row, fallback_used))].append(row)

    written = []
    for split_name in ("train", "test"):
        out = []
        for idx, row in enumerate(buckets[split_name]):
            extra = dict(row.get("extra_info") or {})
            extra["index"] = idx
            extra["split"] = split_name
            out.append({**row, "extra_info": extra})
        path = write_split(out, out_dir, split_name)
        pct = 100.0 * len(out) / len(rows)
        print(f"  {split_name}: {len(out)} 行 ({pct:.2f}%) -> {path}")
        written.append(path)

    if fallback_used[0]:
        print(
            "\n!! 上游没有 extra_info.index / original_index，划分退回到了题面文本。\n"
            "   规则仍然确定，但重复题面会被分到同一边——请记一笔。"
        )
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    args = ap.parse_args()

    build(args.out_dir, allow_count_mismatch=args.allow_count_mismatch)

    print(
        "\n提醒：这个划分不等于历史的 16125/1792，grader_spec.json 里\n"
        "historical_final_baseline / required_final_score 需要重跑重算。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
