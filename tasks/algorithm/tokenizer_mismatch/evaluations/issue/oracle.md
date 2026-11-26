# Oracle

## 根因
teacher(Llama tokenizer) vs student(Qwen tokenizer) 不一致→token 对不齐→KL 无意义

## 陷阱
以为学习率问题，实际是 tokenizer 不匹配

## 证据
token 对齐率<50%；KL 不收敛

## 判分
根因须落到上述；证据须来自上述；只结论表面→Fail。
