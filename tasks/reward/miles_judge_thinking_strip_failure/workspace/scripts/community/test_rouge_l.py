#!/usr/bin/env python3
"""
Debug Miles ROUGE-L (rm_type=rouge_l): compare two strings and print F1 / recall / precision.

Run from the Miles repo root (directory that contains train.py):

  python scripts/community/test_rouge_l.py "prediction text" "reference / label text"

Strip Qwen think / ###Response from the first string (same as training reward):

  python scripts/community/test_rouge_l.py --as-response '...full model output...' 'label from jsonl'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import miles.rollout.rm_hub.rouge_l as rouge_l_mod  # noqa: E402
from miles.rollout.rm_hub.f1 import normalize_answer  # noqa: E402
from miles.rollout.rm_hub.rouge_l import (  # noqa: E402
    compute_rouge_l_reward,
    extract_solution_text_for_rouge,
    rouge_l_f1,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Miles ROUGE-L scoring on two strings.")
    parser.add_argument("prediction", help="Model answer (or full response if --as-response).")
    parser.add_argument("reference", help="Ground-truth label string.")
    parser.add_argument(
        "--as-response",
        action="store_true",
        help="Run extract_solution_text_for_rouge(prediction) before scoring (matches rm_hub reward path).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print normalized strings, tokenization mode, lengths, and LCS length.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print one JSON object with metrics (for scripting).",
    )
    args = parser.parse_args()

    pred_in = args.prediction
    ref = args.reference

    if args.as_response:
        pred_used = extract_solution_text_for_rouge(pred_in) or ""
        extract_note = "extract_solution_text_for_rouge(prediction)"
    else:
        pred_used = pred_in
        extract_note = "prediction (raw)"

    f1, recall, precision = rouge_l_f1(pred_used if pred_used else None, ref)
    reward_training = compute_rouge_l_reward(pred_in, ref)

    norm_p = normalize_answer(str(pred_used))
    norm_r = normalize_answer(str(ref))
    pred_toks = rouge_l_mod._rouge_l_tokens(norm_p)
    ref_toks = rouge_l_mod._rouge_l_tokens(norm_r)
    lcs_len = rouge_l_mod._lcs_length(pred_toks, ref_toks) if pred_toks and ref_toks else 0
    cjk = bool(rouge_l_mod._CJK_RE.search(norm_p) or rouge_l_mod._CJK_RE.search(norm_r))

    payload = {
        "f1": f1,
        "recall_l": recall,
        "precision_l": precision,
        "compute_rouge_l_reward": reward_training,
        "lcs_length": lcs_len,
        "pred_token_len": len(pred_toks),
        "ref_token_len": len(ref_toks),
        "tokenization": "character (CJK)" if cjk else "whitespace words",
        "scored_as": extract_note,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"Input path: {extract_note}")
    if args.as_response and pred_used != pred_in.strip():
        print("(think / marker stripped; shown below is text used for ROUGE-L)\n")
    print("--- prediction (used) ---")
    print(pred_used)
    print("--- reference ---")
    print(ref)
    print()
    print(f"ROUGE-L F1:     {f1:.6f}")
    print(f"ROUGE-L recall: {recall:.6f}")
    print(f"ROUGE-L prec:   {precision:.6f}")
    print(f"compute_rouge_l_reward(prediction, ref): {reward_training:.6f}")
    if not args.as_response and reward_training != f1:
        print(
            "(note: reward strips </think> / ###Response from prediction; "
            "use --as-response or pass only the final answer to align scored text.)",
        )

    if args.verbose:
        print()
        print(f"normalize_answer(pred): {norm_p!r}")
        print(f"normalize_answer(ref):  {norm_r!r}")
        print(f"Tokenization: {payload['tokenization']}")
        print(f"pred len={len(pred_toks)} ref len={len(ref_toks)} LCS={lcs_len}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
