# Oracle

## 根因
reference model 停留在初始 checkpoint→KL 惩罚基准错误→policy 漂移

## 陷阱
以为 KL 系数问题，实际是 reference 没更新

## 证据
ref 与 current 的版本差>100 步；KL 计算基准错误

## 判分
根因须落到上述；证据须来自上述；只结论表面→Fail。
