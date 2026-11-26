# Oracle

## 根因
KL 系数 0.001 太小，policy 漂移过远，KL 爆炸→reward 崩

## 陷阱
以为学习率太大，实际是 KL 约束失效

## 证据
KL vs step 指数增长；policy 与 ref 的 log-ratio 方差爆炸

## 判分
根因须落到上述；证据须来自上述；只结论表面→Fail。
