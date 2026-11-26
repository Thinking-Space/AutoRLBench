# tokenizer_mismatch

- **Category**: `algorithm`
- **One-liner**: 蒸馏不收敛，根因是 teacher/student tokenizer 不一致

## 陷阱
以为学习率问题，实际是 tokenizer 不匹配

## 根因
teacher(Llama tokenizer) vs student(Qwen tokenizer) 不一致→token 对不齐→KL 无意义

## 构造
`evaluations/solution/oracle/simulate_training.py`
