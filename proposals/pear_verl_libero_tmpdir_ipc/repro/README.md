# Supplemental CPU checks
这两个程序只使用Python标准库；不联网、不导入生产模块、不启动Ray、vLLM或训练。

## 机制探针
```bash
CUDA_VISIBLE_DEVICES="" NO_GPU=1 PYTHONDONTWRITEBYTECODE=1 python3 repro/probe_unix_socket.py
```
在明确的个人临时子目录内，先以超过107字节的绝对路径实际绑定AF_UNIX socket并检查失败，再在同一目录使用短相对路径绑定同一个endpoint并检查成功，最后清理。没有通过/proc或符号链接访问其他目录。

这是Unix socket机制层面的复现，不是完整Ray/vLLM初始化复现；真实八节点ZMQ绑定的历史证据另见evidence/eight_node_cpu_zmq.json。相对路径只是探针的局部对照，生产修复使用节点已有的短IPC目录，两者不能说成同一个端到端实验。

默认临时目录为本包repro/.runtime，测试后无socket留存；目录本身可能保留。可用--work-dir显式设置其他自己有权限的目录，--output保存无内部路径的JSON结果。

## 只读证据与完整性校验
```bash
CUDA_VISIBLE_DEVICES="" NO_GPU=1 PYTHONDONTWRITEBYTECODE=1 python3 repro/verify_bundle.py
```
检查：故障路径长度、AST提取的真实转发白名单差异、两份补丁完整重建after源码、历史CPU和八节点记录、恢复/安装的rank和副本数、16/17步更新保存发布及原始metric行SHA、来源变换哈希、最终MANIFEST逐文件SHA。

它验证“包内材料彼此一致”，不是独立执行历史GPU实验，不替代官方validator、镜像启动检查或维护者验收。
