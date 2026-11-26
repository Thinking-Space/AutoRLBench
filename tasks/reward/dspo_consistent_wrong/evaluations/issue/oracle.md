# Oracle：根因与证据

## 根因
整组 G 条 rollout 都错，正确答案不在组里；negative filtering 只能重排错答案，不能 flip wrong→correct；天花板=组内 coverage

## 诊断陷阱
以为 reward 需要调参，实际是结构问题：组内 coverage≈0

## 证据
group coverage(含正确答案的组占比)≈0；evidence support AUC 0.927 能区分对错共识

## 判分要点
- 根因必须落到上述根因，不能只归因表面症状。
- 证据必须来自上述证据。
- 只结论表面症状 → Fail。
