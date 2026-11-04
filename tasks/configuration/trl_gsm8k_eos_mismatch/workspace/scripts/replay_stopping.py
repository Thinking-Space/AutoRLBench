import argparse
import json
import os
import re
import sys

# Replay of the recorded rollouts under the workspace's generation
# configuration.
#
# The historical run's diagnosis (outputs/evaluation_metrics.json) came from
# "aligned end-of-turn stopping and greedy decoding", and the retained
# rollout records (outputs/rollout_samples.jsonl, decoded with
# skip_special_tokens=False) carry the raw completions in which generation
# ran past a completed turn. This script re-executes the stopping rule
# itself -- no model, no GPU: the stopping tokens are read from
# model/generation_config.json (the file a repair edits), each retained
# completion is cut at the first stopping token it contains, and the cut
# text is scored with the training's own last-match answer extraction
# (scripts/train_grpo.py::correctness).
#
# Special tokens are atomic in the tokenizer, so locating the token's
# string form in the decoded text finds exactly the token's position; the
# replay is therefore a string-level analogue of decoding with
# stop_token_ids and ignore_eos=True (the config under test is the sole
# stopping authority). Token counts of the cut prefixes are not
# reconstructible without a tokenizer, so no token-count metric is emitted.

# Same extraction as training's correctness reward: last match wins.
ANS_RE = re.compile(r"ANSWER:\s*\**\s*\$?(-?\d[\d,]*(?:\.\d+)?)")


def extract_last_answer(text):
    matches = list(ANS_RE.finditer(text))
    return matches[-1].group(1).replace(",", "") if matches else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay recorded rollouts under the workspace generation config")
    parser.add_argument("--rollouts", default="outputs/rollout_samples.jsonl",
                        help="retained rollout records (frozen evidence)")
    parser.add_argument("--generation-config", default="model/generation_config.json",
                        help="generation config under test (the repair target)")
    parser.add_argument("--added-tokens", default="model/added_tokens.json",
                        help="special-token id -> string map used to locate stop tokens")
    parser.add_argument("--out-dir", default=os.environ.get("EVAL_OUT", "outputs/eval"))
    args = parser.parse_args()

    with open(args.generation_config) as fh:
        gen_cfg = json.load(fh)
    eos_ids = gen_cfg.get("eos_token_id")
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    if not eos_ids:
        print("generation config has no eos_token_id", file=sys.stderr)
        return 1

    # id -> marker string; special tokens are atomic, so the first string
    # occurrence is the first token occurrence. Ids absent from the map
    # cannot be located in text and are reported rather than silently
    # dropped.
    added = json.load(open(args.added_tokens))
    by_id = {v: k for k, v in added.items()}
    stop_strings = [(i, by_id[i]) for i in eos_ids if i in by_id]
    unmapped = [i for i in eos_ids if i not in by_id]

    records = [json.loads(line) for line in open(args.rollouts, encoding="utf-8")]

    os.makedirs(args.out_dir, exist_ok=True)
    correct = 0
    out = []
    for rec in records:
        text = rec["text"]
        # First stopping token in the completion, under the config under test.
        cut, stop_id = len(text), None
        for token_id, marker in stop_strings:
            pos = text.find(marker)
            if pos != -1 and pos < cut:
                cut, stop_id = pos, token_id
        replayed = text[:cut]
        pred = extract_last_answer(replayed)
        gold = rec["gold"].replace(",", "")
        ok = pred == gold
        correct += ok
        out.append({"step": rec["step"], "gold": gold, "pred": pred,
                    "stopped": stop_id is not None, "stop_token_id": stop_id,
                    "stop_at_char": cut, "recorded_n_tokens": rec["n_tokens"],
                    "correct": ok})

    n = len(records)
    summary = {
        "protocol": "deterministic replay: retained rollouts cut at the first "
                    "stopping token from model/generation_config.json, scored "
                    "with the training's last-match answer extraction",
        "n_examples": n,
        "eos_token_id": eos_ids,
        "unmapped_eos_ids": unmapped,
        "accuracy": (correct / n) if n else 0.0,
        "frac_finished_eos": (sum(r["stopped"] for r in out) / n) if n else 0.0,
        "frac_answer_extracted": (sum(r["pred"] is not None for r in out) / n) if n else 0.0,
    }
    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    with open(os.path.join(args.out_dir, "eval_samples.jsonl"), "w") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
