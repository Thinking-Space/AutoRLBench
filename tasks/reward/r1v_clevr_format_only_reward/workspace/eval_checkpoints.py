#!/usr/bin/env python
"""Held-out evaluation of the base model and every saved checkpoint.

Two sets, both held out from training:
  val      in-distribution (CLEVR-CoGenT-A counting questions not used for training)
  ood_val  out-of-distribution (SuperCLEVR questions with integer answers)
Protocol: greedy decoding, 256 new tokens, the first integer inside <answer>...</answer>
must equal the gold integer. Writes one metrics line per (step, split) to
<out>/eval_metrics.jsonl and every model output to <out>/eval_outputs/step<N>_<split>.jsonl.
"""
import argparse, glob, json, os, re

import torch
from datasets import load_from_disk
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

TEMPLATE = ("{q} First output the thinking process in <think> </think> and final answer (number) "
            "in <answer> </answer> tags.")
ANS = re.compile(r"<answer>\s*(\d+)\s*</answer>", re.I)


def gold_int(s):
    m = ANS.search(str(s))
    if m:
        return int(m.group(1))
    m = re.search(r"-?\d+", str(s))
    return int(m.group()) if m else None


def run(model_dir, proc_dir, ds, bs):
    proc = AutoProcessor.from_pretrained(proc_dir, max_pixels=401408, min_pixels=3136)
    proc.tokenizer.padding_side = "left"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.eval()
    model.generation_config.use_cache = True
    rows = []
    for s in range(0, len(ds), bs):
        ex = [ds[i] for i in range(s, min(s + bs, len(ds)))]
        msgs = [[{"role": "user", "content": [{"type": "image", "image": e["image"]},
                                              {"type": "text", "text": TEMPLATE.format(q=e["problem"])}]}]
                for e in ex]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        imgs, vids = process_vision_info(msgs)
        inp = proc(text=texts, images=imgs, videos=vids, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=256, do_sample=False, use_cache=True)
        outs = proc.batch_decode([o[len(i):] for i, o in zip(inp.input_ids, gen)],
                                 skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for j, (e, o) in enumerate(zip(ex, outs)):
            m = ANS.search(o)
            pred = int(m.group(1)) if m else None
            g = gold_int(e["solution"])
            rows.append({"index": s + j, "question": e["problem"], "gold": g, "pred": pred,
                         "correct": pred is not None and pred == g, "output": o})
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-n", type=int, default=500)
    ap.add_argument("--ood-n", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps", default="", help="comma list; default: base + all checkpoints")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "eval_outputs"), exist_ok=True)
    sets = {"val": load_from_disk(os.path.join(a.data, "val")).select(range(a.val_n)),
            "ood_val": load_from_disk(os.path.join(a.data, "ood_val")).select(range(a.ood_n))}
    ck = {int(p.rsplit("-", 1)[1]): p for p in glob.glob(os.path.join(a.ckpt_root, "checkpoint-*"))}
    models = {0: a.base, **ck}
    want = [int(x) for x in a.steps.split(",") if x] or sorted(models)
    mpath = os.path.join(a.out, "eval_metrics.jsonl")
    for step in want:
        for name, ds in sets.items():
            rows = run(models[step], a.base, ds, a.batch_size)
            n = len(rows)
            acc = sum(r["correct"] for r in rows) / n
            fmt = sum(r["pred"] is not None for r in rows) / n
            with open(os.path.join(a.out, "eval_outputs", f"step{step}_{name}.jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            rec = {"step": step, "split": name, "n": n, "accuracy": round(acc, 4),
                   "answer_parsed": round(fmt, 4)}
            print(json.dumps(rec), flush=True)
            with open(mpath, "a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
