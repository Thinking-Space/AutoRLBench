"""共用的小工具：HF 缓存根 + 写 parquet。"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def hf_cache_root() -> Path:
    """HF 系库的缓存根，宿主机上就是 datasets/hf。

    布局刻意和 `HF_HOME=$AUTORLBENCH_DATASETS/hf` 会产出的一样（hub 在 hf/hub，
    datasets 在 hf/datasets），是 datasets 库自己的缓存根，和 fetch_hf 的
    per-repo 下载目录各走各的。

    **必须在 import huggingface_hub / datasets 之前调用。** 那两个库都在 import 时
    就把 HF_HOME 读进模块级常量（constants.HF_HUB_CACHE），之后再改 os.environ
    是一个静默的 no-op——fetch_hf.py 的前身（ptdbench_fetch.py）原来就是这么把
    954MB 权重下到容器可写层里的。配方里的 import 都是函数内延迟 import，就是为了让
    这个调用能排在前面；谁要把 import 提到模块顶层，这个保证就没了。
    """
    root = Path(os.environ.get("AUTORLBENCH_DATASETS", "/autorlbench/datasets")) / "hf"
    root.mkdir(parents=True, exist_ok=True)
    # 除了下面显式传的 cache_dir，还有些路径（元数据、etag、load_dataset 内部
    # 调的 hf_hub_download）只认环境变量，所以两个都设。
    os.environ.setdefault("HF_HOME", str(root))
    return root


def hf_datasets_cache() -> Path:
    """`load_dataset(cache_dir=...)` 该传的那个目录。"""
    return hf_cache_root() / "datasets"


def write_split(rows: list[dict], out_dir: Path, name: str) -> Path:
    """写一个 split，返回路径。

    显式走 pandas -> parquet，不用 datasets 自己的 to_parquet：后者的分片和压缩
    参数在不同版本间会变，而我们要的是"同样的输入产出同样的字节"。
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
