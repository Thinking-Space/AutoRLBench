#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Construct an explicit math replay fixture; never label it a model generation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


def source_sample_class(source):
    spec = importlib.util.spec_from_file_location("justrl2_sample_types", source / "miles/utils/types.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Sample


def cp_indices(total, response, rank, cp_size=2):
    """Independent geometry reference, including the next-token logit offset."""
    chunk = (total + 2 * cp_size - 1) // (2 * cp_size)
    prompt = total - response
    indices = []
    for index in [rank, 2 * cp_size - rank - 1]:
        start, end = max(index * chunk, prompt - 1), min((index + 1) * chunk, total - 1)
        indices.extend(i + 1 - prompt for i in range(start, end))
    return indices


def build(source, model, output):
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    Sample = source_sample_class(source)
    context = "The reference card lists the first ten positive integers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. "
    long_answer = (
        "Counting one object and one more gives two objects. On the number line, start at one "
        "and move one step to reach two. Addition combines disjoint groups, so two singletons "
        "have cardinality two. "
    )
    # Repeated text is deliberate fixture construction, documented in metadata.
    # Wrong short answer ->0, correct longer answer ->1. Rewards are authored labels.
    rows = [
        (context * 10 + "What is 1 + 1? Answer with only the integer.", "3", 0.0),
        ("Compute 1 + 1. Explain using several elementary methods, and end with the integer answer.",
         long_answer * 8 + "Therefore 1 + 1 = 2. Answer: 2.", 1.0),
    ]
    samples, ledger = [], []
    for index, (question, answer, reward) in enumerate(rows):
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                              tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(answer, add_special_tokens=False) + [130073]
        sample = Sample(index=index, group_index=index, prompt=prompt,
                        tokens=prompt_ids + response_ids,
                        response=tokenizer.decode(response_ids, skip_special_tokens=False),
                        response_length=len(response_ids), label="2", reward=reward,
                        loss_mask=[1] * len(response_ids), status=Sample.Status.COMPLETED,
                        rollout_log_probs=None,
                        metadata={"fixture_kind": "constructed_scripted_math_trajectory",
                                  "generation_performed": False, "reward_source": "authored_exact_integer_label",
                                  "question": question, "answer_before_eos": answer})
        sample.validate()
        if not all(0 <= token < len(tokenizer) for token in sample.tokens):
            raise ValueError("Out-of-vocabulary fixture token")
        counts = [len(cp_indices(len(sample.tokens), sample.response_length, rank)) for rank in range(2)]
        if sum(counts) != sample.response_length:
            raise ValueError("CP slices do not cover response")
        samples.append(sample.to_dict())
        ledger.append({"sample_index": index, "dp_replica": index, "prompt_tokens": len(prompt_ids),
                       "response_tokens": len(response_ids), "total_tokens": len(sample.tokens),
                       "cp_response_counts": counts, "reward": reward})
    if not (ledger[0]["cp_response_counts"][1] == 0 < ledger[1]["cp_response_counts"][1]):
        raise ValueError("Pinned tokenizer fixture no longer produces empty/nonempty DP peers")
    if any(sum(row["cp_response_counts"][cp] for row in ledger) <= 1 for cp in range(2)):
        raise ValueError("Insufficient global whitening mask count")
    payload = {"rollout_id": 0, "samples": samples}
    (output / "replay.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    torch.save(payload, output / "replay.pt")
    loaded = torch.load(output / "replay.pt", weights_only=False)
    if loaded != payload:
        raise ValueError("Replay serialization mismatch")
    manifest = {"constructed": True, "online_generation": False,
                "model": "openbmb/JustRL-II-base-model",
                "model_revision": "b558bbc71d9433e7c5869effcbac07d4c7f2c5b4",
                "global_batch_size": 2, "micro_batch_size": 1, "balance_data": False,
                "topology_per_role": {"cp": 2, "dp": 2, "tp": 1, "pp": 1},
                "samples": ledger, "files": {}}
    for name in ["replay.json", "replay.pt"]:
        data = (output / name).read_bytes()
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source.resolve(), args.model.resolve(), args.output.resolve())
