# verl #6068 配对40步实验

本观察窗口未满足约定的质量下降与修复缓解完整判据。

两组分别从Qwen3-VL-8B-Instruct原始权重开始，固定ScienceQA带图train2560/val1024、seed42。两步冒烟与正式轨迹独立。FSDP4 FULL_SHARD、DP2×SP2、fused/remove-padding=True、batch128、mini32、n8、lr1e-6、长度16384+4096；调度器200、正常stop40，每5验证、每20保存。

before固定4b9c14f2c306d43f84eccf47315323e3f1ce1270；after仅增加PR6268六生产文件目标修复，相同公共兼容补丁。模型/数据revision、输入SHA、配置与实际布局均保存在证据中。

|组别|前20reward|后20reward|初始验证|step40验证|最后5次均分相对初始|reward OLS|验证OLS|
|---|---:|---:|---:|---:|---:|---:|---:|
|before|0.951416|0.917871|93.847656%|90.039062%|-2.5781pp|-0.00126816|-0.00134115|
|after|0.967578|0.982861|94.628906%|97.265625%|+2.7734pp|0.00085298|0.00063802|

after相对before的验证变化量改善+5.3516个百分点。逐项冻结判据见comparison.json。before未达到最后5次验证均分相对初始下降至少5个百分点的判据；即使after改善，也不能据此宣称完整复现。

两个正式attempt均为连续0–40标量、40次外层更新、每rank160次AdamW调用、9次各1024验证输出、step20/40四rank非空model/optim/extra_state及data.pt。固定文本输入/目标/顺序跨组相同；标签/mask/advantage/实际损失梯度与W&B原件覆盖/退出均经核验。观测事件受每进程2048上限限制，累计次数跨重复前后向，不是唯一token/样本或所有token覆盖。

before实际观测response标签错误16311325次、非零advantage错误2896350次、错误位置非零梯度381610次；after对应计数0、0、0。

累计实际59.250619 GPU卡时（after阶段20.938509），包括全部历史、失败、诊断与分配间隙；无卡时硬上限，账本闭合。before旧deadline未触发；after实际进程deadline_env=inf、父进程等待无timeout，此控制差异不改变优化配方或正常stop40。

单seed、公开替代数据，仅覆盖当前40步窗口，不推断原内部数据或跨seed稳定性。checkpoint分片未计算SHA且未测试恢复，大checkpoint与原始tensor留远程，tensor清单SHA保留。BENCH_NORMAL_EXIT_STEP=41为最终更新后的内部递增，实际最终优化步40。无额外judge、官方评测、benchmark题包、commit、PR或公开上传。
