import json, os, re
from datasets import load_dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

BASE_MODEL = os.environ.get("INIT_MODEL", "model")
ROLLOUT_DIR = os.environ.get("ROLLOUT_DIR")
OUT_DIR = os.environ.get("OUT_DIR", "outputs/checkpoint")
TEMPLATE_PATH = os.environ.get("QWEN3_TEMPLATE", "templates/qwen3.jinja")
IM_END, ENDOFTEXT = 151645, 151643

MATH_PROMPT = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()

ANS_RE = re.compile(r"ANSWER:\s*\**\s*\$?(-?\d[\d,]*(?:\.\d+)?)")


def extract_answer(text):
    m = ANS_RE.search(text)
    return m.group(1).replace(",", "") if m else None


def correctness(completions, answer, **kwargs):


    rewards = []
    for comp, gold in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp[0], dict) else comp
        matches = list(ANS_RE.finditer(text))
        pred = matches[-1].group(1).replace(",","") if matches else None
        rewards.append(1.0 if pred == gold.replace(",", "") else 0.0)
    return rewards

_tok = None


def has_im_end(completions, completion_ids, answer=None, trainer_state=None, **kwargs):
    if ROLLOUT_DIR:
        global _tok
        if _tok is None:
            _tok = AutoTokenizer.from_pretrained(BASE_MODEL)
        os.makedirs(ROLLOUT_DIR, exist_ok=True)
        rank = os.environ.get("RANK", "0")
        step = trainer_state.global_step if trainer_state is not None else -1
        with open(os.path.join(ROLLOUT_DIR, f"rank{rank}.jsonl"), "a") as f:
            for ids, gold in zip(completion_ids, answer or [None] * len(completion_ids)):
                f.write(json.dumps({"step": step, "gold": gold, "n_tokens": len(ids),
                                    "ended_eos": bool(len(ids) and ids[-1] == ENDOFTEXT),
                                    "text": _tok.decode(ids, skip_special_tokens=False)},
                                   ensure_ascii=False) + "\n")
    return [1.0 if IM_END in ids else 0.0 for ids in completion_ids]


def stopped_on_eos(completions, completion_ids, **kwargs):
    return [1.0 if len(ids) and ids[-1] == ENDOFTEXT else 0.0 for ids in completion_ids]


def make_prompt(question):
    return [{"role": "user", "content": MATH_PROMPT.format(prompt=question)}]


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    with open(TEMPLATE_PATH) as f:
        tokenizer.chat_template = f.read()

    assert tokenizer.eos_token_id == ENDOFTEXT

    ds = load_dataset("openai/gsm8k", "main", split="train",
                      cache_dir=os.environ.get("HF_DATASETS_CACHE", "data/cache"))
    ds = ds.shuffle(seed=0).map(
        lambda ex: {"prompt": make_prompt(ex["question"]),
                    "answer": ex["answer"].split("####")[-1].strip()},
        remove_columns=ds.column_names)

    config = GRPOConfig(
        output_dir=OUT_DIR,
        max_steps=int(os.environ.get("MAX_STEPS", 150)),
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_generations=8,
        learning_rate=float(os.environ.get("LR","1e-6")),
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        max_completion_length=1024,
        temperature=1.0,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        vllm_max_model_length=2048,

        reward_weights=[1.0, 0.0, 0.0],
        save_steps=int(os.environ.get("SAVE_STEPS",50)),
        save_only_model=True,
        logging_steps=5,
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",
        model_init_kwargs={"dtype": "bfloat16"},
    )

    trainer = GRPOTrainer(
        model=BASE_MODEL,
        args=config,
        train_dataset=ds,
        reward_funcs=[correctness, has_im_end, stopped_on_eos],
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(OUT_DIR)


if __name__ == "__main__":
    main()

