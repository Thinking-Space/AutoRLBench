# AutoRLBench 交付仓库

已集成1个issue级诊断任务：[training_case_v8k2](tasks/infrastructure/training_case_v8k2/readme.md)，署名Gaoyuan。

- 固定框架：`5e8933cf6e881b0b63a51db0db9abfb4b9fdcd1e`
- 交付分支：[Zaogezi/AutoRLBench · delivery/training-case-v8k2](https://github.com/Zaogezi/AutoRLBench/tree/delivery/training-case-v8k2)
- origin：`https://github.com/Zaogezi/AutoRLBench.git`；upstream：`https://github.com/Thinking-Space/AutoRLBench.git`
- 任务：1,322份文件、1,171份框架源码，90,339,492字节
- 环境：新增`vl-grpo-cu129-recorded`镜像定义、依赖锁及docker构建上下文允许项
- 版本策略：保留约定框架基线，通过独立交付分支发布

|检查|结果|
|---|---|
|固定版本官方loader/validator|通过，0错误、0警告|
|最终包恢复与导入|归档SHA/size及1,322份任务文件逐项核验通过|
|Git完整性|全部任务文件已纳入索引，Git blob与本地原始字节一致|
|Linux权限|15份可执行文件已在Git索引中设置100755|
|隐藏参考补丁|git apply --check通过|
|共享verifier|与固定版本Git对象逐字节一致；CLI帮助与shell语法通过|
|移植入口|4份Python入口语法检查通过|
|镜像build、干净环境prepare、模型judge、新训练|本次未执行|

历史配对实测：before最后5次验证均分变化−2.578125pp，after为+2.7734375pp，配对变化改善+5.3515625pp。历史盲测结果见[BLINDTEST.md](delivery/training_case_v8k2/BLINDTEST.md)。任务内validation保留原打包时记录，本次重建检查见[delivery-receipt.json](delivery-receipt.json)。

在仓库根目录核验：

```bash
python -B -m autorlbench validate tasks/infrastructure/training_case_v8k2
git show --stat HEAD
git show HEAD -- docker/.dockerignore docker/images/vl-grpo-cu129-recorded.Dockerfile
```

完整文件校验清单：[delivery-manifest.json](delivery-manifest.json)。公开agent输入由框架从本任务的issue instruction与workspace组合；metadata、oracle和reference保留在维护者侧。

本地只保留交付仓库及现有candidates，不另建构建目录、资源缓存、归档展开副本或过程日志。
