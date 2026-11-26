# data_contamination

- **Category**: `data`
- **One-liner**: 评测虚高，根因是 train/test 数据污染(重叠)

## 陷阱
以为模型很强，实际是数据泄漏

## 根因
train/test 重叠 30%→评测虚高；去重后掉到 60%

## 构造
`evaluations/solution/oracle/simulate_training.py`
