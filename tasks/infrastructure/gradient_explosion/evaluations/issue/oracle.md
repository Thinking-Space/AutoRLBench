# Oracle

## 根因
grad clip 10.0 太大+batch 8 太小→梯度尖峰→loss NaN

## 陷阱
以为模型不兼容，实际是梯度爆炸

## 证据
grad norm 尖峰>1000；loss 几步内 NaN

## 判分
根因须落到上述；证据须来自上述；只结论表面→Fail。
