# Oracle

## 根因
无 length normalization→长序列 log-prob 更负→被系统性惩罚/偏好→长度偏置

## 陷阱
以为模型在学，实际是长度偏置

## 证据
输出长度与 reward 强相关但与 correct 无关

## 判分
根因须落到上述；证据须来自上述；只结论表面→Fail。
