"""Does the training-time fake quantizer reproduce vLLM's real one end to end?

The unit parity test (``tests/test_turboquant_parity.py``) compares centroid
indices and value levels on random tensors, which pins the arithmetic but not
the *plumbing*: where in the network the quantizer sits, whether RoPE has been
applied, whether the GQA head layout matches, which layers are skipped.  Any of
those can be wrong while every unit test passes, and the result is a student
trained for a quantizer it is never deployed with.

This script closes that gap by comparing log-probabilities of the *same tokens*
between the two implementations:

  vLLM  : generate greedily with ``--kv-cache-dtype <preset>``.  Every generated
          token's log-prob comes from a real decode step against the real
          quantized paged cache.
  HF    : teacher-force that exact sequence through the training-side model with
          the fake quantizer installed, and read the same tokens' log-probs.

The headline number is meaningless on its own, because HF and vLLM never agree
exactly (different kernels, different reduction orders).  So the run also
measures a **full-precision control** -- the same comparison with the quantizer
off on both sides -- which gives the engine-level noise floor.  The verdict is
relative:

    delta_quantized ~ delta_fp        -> the fake quantizer matches
    delta_quantized >> delta_fp       -> it does not, and the gap is the bug
    delta_quantized ~ delta_no_quant  -> the fake quantizer is doing nothing
                                         useful (compare against ``fp`` run
                                         through the *quantized* vLLM tokens)

Prefill asymmetry
-----------------
vLLM's TurboQuant backend only quantizes what enters the paged cache.  A
first-chunk prefill (``max_query_len == max_seq_len``) is served by flash-attn
on the raw post-RoPE K/V and never reads the cache, so at inference a *prompt*
token attends to unquantized K/V while a *generated* token attends to the
quantized cache.  A teacher-forced training forward quantizes both.  The script
therefore evaluates two HF variants:

  fq_uniform       quantize everywhere (what training does today)
  fq_prefill_exact prompt queries see raw K/V, response queries see quantized
                   K/V (what inference actually does)

Comparing the two says how much that asymmetry is worth, and whether training
should adopt the split.

Phases run as separate processes so the 7B model is never resident twice::

    python parity_forward_logprobs.py --phase vllm --kv-scheme turboquant_k3v4_nc --out tq_vllm.json
    python parity_forward_logprobs.py --phase hf   --tokens tq_vllm.json          --out tq_hf.json
    python parity_forward_logprobs.py --phase vllm --kv-scheme none               --out fp_vllm.json
    python parity_forward_logprobs.py --phase hf   --tokens fp_vllm.json          --out fp_hf.json
    python parity_forward_logprobs.py --phase report --quantized tq_hf.json --control fp_hf.json --out report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_MODEL = "/historical/shared/qat/models/base/deepseek-r1-distill-qwen-7b"
DEFAULT_PARQUET = (
    "/historical/shared/qat/data/processed/skywork_or1_7b_math_difficulty1-15_16k.parquet"
)


# --------------------------------------------------------------------------- #
#  prompt                                                                      #
# --------------------------------------------------------------------------- #
def _prompt_token_ids(args) -> tuple[list[int], str]:
    """Return the prompt's token ids, taken from the real training dataset.

    A synthetic prompt would not exercise the length regime the quantizer
    actually runs in, so the default is a row of the parquet veRL trains on.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.prompt_text:
        text = args.prompt_text
    else:
        import pyarrow.parquet as pq

        table = pq.read_table(args.prompt_parquet, columns=[args.prompt_key])
        raw = table.column(args.prompt_key)[args.prompt_row].as_py()
        if isinstance(raw, list):
            # veRL chat format: [{"role": ..., "content": ...}, ...]
            text = tokenizer.apply_chat_template(
                raw, tokenize=False, add_generation_prompt=True
            )
        else:
            text = str(raw)

    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > args.max_prompt_tokens:
        # Keep the tail: the generation-prompt suffix must survive.
        ids = ids[-args.max_prompt_tokens :]
    return list(ids), text


# --------------------------------------------------------------------------- #
#  phase: vllm                                                                 #
# --------------------------------------------------------------------------- #
def phase_vllm(args) -> dict:
    # sm_103a's bundled ptxas cannot assemble the Triton sampler; the rest of
    # the pipeline sets this too, so match it here for comparability.
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    prompt_ids, prompt_text = _prompt_token_ids(args)

    engine_kwargs = {}
    quantized = args.kv_scheme not in (None, "", "none")
    if quantized:
        engine_kwargs["kv_cache_dtype"] = args.kv_scheme

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=len(prompt_ids) + args.max_new_tokens + 64,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        trust_remote_code=True,
        seed=0,
        **engine_kwargs,
    )

    verification = None
    if quantized:
        # Doubles as a real-engine exercise of the verifier the training
        # rollout depends on.  Recorded rather than fatal here: this script's
        # job is the log-prob comparison, and losing that to a verifier
        # limitation would be the wrong trade.  The training path keeps it
        # fail-closed.
        from qat_kv.native_tq_verify import verify_native_turboquant_rollout

        try:
            verification = verify_native_turboquant_rollout(llm, args.kv_scheme)
        except Exception as exc:
            verification = {"rollout_kv_verified": False, "error": f"{type(exc).__name__}: {exc}"}
            print(f"[parity] WARNING: engine verification failed: {exc}", file=sys.stderr)

    sampling = SamplingParams(
        # Greedy by default (reproducible).  --temperature 1.0 reproduces the
        # training rollout's regime: sampled tokens come from the tail of the
        # distribution, where two quantizers disagree far more than they do on
        # the argmax token.
        temperature=args.temperature,
        seed=args.seed,
        top_p=1.0,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        # vLLM's default logprobs_mode is "raw_logprobs": log_softmax of the
        # logits before temperature scaling, i.e. directly comparable to an HF
        # log_softmax.  Requesting 1 also returns the sampled token's entry.
        logprobs=1,
        detokenize=False,
    )
    outputs = llm.generate([TokensPrompt(prompt_token_ids=prompt_ids)], sampling)
    completion = outputs[0].outputs[0]

    token_ids = list(completion.token_ids)
    logprobs = []
    for token_id, entry in zip(token_ids, completion.logprobs):
        if token_id not in entry:
            raise RuntimeError(
                f"vLLM did not return a logprob for its own sampled token {token_id}"
            )
        logprobs.append(float(entry[token_id].logprob))

    if len(logprobs) != len(token_ids):
        raise RuntimeError(
            f"got {len(logprobs)} logprobs for {len(token_ids)} tokens"
        )
    if not token_ids:
        raise RuntimeError("vLLM generated no tokens; nothing to compare")

    return {
        "phase": "vllm",
        "model": args.model,
        "kv_scheme": args.kv_scheme,
        "quantized": quantized,
        "temperature": args.temperature,
        "prompt_len": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "prompt_text_sha": _sha(prompt_text),
        "generated_token_ids": token_ids,
        "vllm_logprobs": logprobs,
        "finish_reason": completion.finish_reason,
        "verification": verification,
    }


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
#  phase: hf                                                                   #
# --------------------------------------------------------------------------- #
def phase_hf(args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM

    from qat_kv.turboquant_attn_hook import install_turboquant_fake_quant

    from transformers import AutoTokenizer

    record = json.loads(Path(args.tokens).read_text())
    tokenizer = AutoTokenizer.from_pretrained(record["model"], trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    prompt_ids = record["prompt_token_ids"]
    gen_ids = record["generated_token_ids"]
    prompt_len = len(prompt_ids)
    ids = prompt_ids + gen_ids

    dtype = getattr(torch, args.hf_dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            record["model"], dtype=dtype, attn_implementation="sdpa", trust_remote_code=True
        )
    except TypeError:  # transformers < 5 spelling
        model = AutoModelForCausalLM.from_pretrained(
            record["model"],
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    model = model.to("cuda").eval()

    def token_logprobs(pad_left: int = 0) -> list[float]:
        """Log-probs of the response tokens.

        ``pad_left`` reproduces veRL's batch layout, which left-pads every
        prompt to ``data.max_prompt_length`` and derives ``position_ids`` from
        the attention mask.  The rollout engine never sees those pads, so if
        the padded forward does not reproduce the unpadded one, actor and
        rollout are scoring the same tokens under different conditions --
        which would show up exactly as a systematic log-prob offset.
        """
        if pad_left:
            padded = [pad_token_id] * pad_left + ids
            attn = [0] * pad_left + [1] * len(ids)
            input_ids = torch.tensor([padded], device="cuda")
            attention_mask = torch.tensor([attn], device="cuda")
            # veRL's convention: positions restart at 0 on the first real token.
            position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
            kwargs = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            offset = pad_left
        else:
            input_ids = torch.tensor([ids], device="cuda")
            kwargs = {}
            offset = 0

        with torch.inference_mode():
            logits = model(input_ids=input_ids, use_cache=False, **kwargs).logits[0].float()
        lp = torch.log_softmax(logits, dim=-1)
        # The logit predicting position t sits at index t-1.
        idx = torch.arange(offset + prompt_len, offset + len(ids), device="cuda")
        targets = torch.tensor(ids, device="cuda")[idx - offset]
        return lp[idx - 1, targets].tolist()

    variants: dict[str, list[float]] = {}
    quantized = record.get("quantized", False)

    # Full precision first, while no quantizer is installed at all -- a
    # stronger control than toggling a flag on an installed hook.
    variants["fp"] = token_logprobs()

    if quantized:
        state = install_turboquant_fake_quant(model, record["kv_scheme"])
        variants["fq_uniform"] = token_logprobs()
        state.assert_layers_quantized()

        state.reset_counters()
        with state.prefill_exact(prompt_len):
            variants["fq_prefill_exact"] = token_logprobs()
        state.assert_layers_quantized()

        # Sanity: with the quantizer disabled the hook must reproduce the
        # uninstalled forward exactly, or the hook itself perturbs the model.
        state.reset_counters()
        with state.disabled():
            variants["fp_via_disabled_hook"] = token_logprobs()

        # Does veRL's left-padded batch layout move the log-probs at all?
        # If this matches fq_uniform, padding is exonerated.
        if args.pad_left:
            state.reset_counters()
            variants["fq_leftpad"] = token_logprobs(pad_left=args.pad_left)
            state.reset_counters()
            with state.disabled():
                variants["fp_leftpad"] = token_logprobs(pad_left=args.pad_left)

            # Padding must be a no-op: with a correct mask, a left-padded
            # forward has to reproduce the unpadded one exactly (up to
            # floating point).  If it does not, the padded numbers describe a
            # broken harness rather than anything about training, and quoting
            # them would be worse than not measuring at all.
            #
            # This guard exists because the first version of this variant did
            # fail it -- the error was concentrated in the earliest response
            # tokens and decayed, exactly as it would if the model were
            # attending to the pad tokens.
            drift = max(
                abs(a - b) for a, b in zip(variants["fp_leftpad"], variants["fp"])
            )
            variants_meta_pad_drift = drift
            if drift > 0.05:
                raise RuntimeError(
                    f"left-padded full-precision log-probs differ from unpadded "
                    f"by up to {drift:.3f}; padding is supposed to be a no-op, so "
                    "the padded variant is measuring a harness bug, not training. "
                    "Fix the mask/position_ids before trusting fq_leftpad."
                )

        quant_metadata = state.metadata()
    else:
        quant_metadata = None

    return {
        "phase": "hf",
        "model": record["model"],
        "kv_scheme": record["kv_scheme"],
        "quantized": quantized,
        "hf_dtype": args.hf_dtype,
        "prompt_len": prompt_len,
        "num_scored_tokens": len(gen_ids),
        "vllm_logprobs": record["vllm_logprobs"],
        "variants": variants,
        "quantizer": quant_metadata,
        "vllm_verification": record.get("verification"),
    }


# --------------------------------------------------------------------------- #
#  phase: report                                                               #
# --------------------------------------------------------------------------- #
def _stats(a: list[float], b: list[float]) -> dict:
    n = len(a)
    if n == 0 or len(b) != n:
        raise ValueError(f"cannot compare {len(a)} and {len(b)} values")
    diffs = [x - y for x, y in zip(a, b)]
    abs_diffs = [abs(d) for d in diffs]
    mean_a = sum(a) / n
    mean_b = sum(b) / n
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    denom = math.sqrt(var_a * var_b)
    return {
        "n": n,
        "mean_abs_diff": sum(abs_diffs) / n,
        "median_abs_diff": sorted(abs_diffs)[n // 2],
        "max_abs_diff": max(abs_diffs),
        "mean_signed_diff": sum(diffs) / n,
        "pearson_r": (cov / denom) if denom > 0 else float("nan"),
        "frac_abs_diff_over_0p1": sum(1 for d in abs_diffs if d > 0.1) / n,
        "frac_abs_diff_over_0p5": sum(1 for d in abs_diffs if d > 0.5) / n,
    }


def _delta_agreement(vllm_lp, hf_fq, hf_fp) -> dict:
    """Compare the *change* each implementation's quantizer caused.

    Absolute log-prob differences conflate two things: the engine-vs-framework
    gap (present even in full precision) and any quantizer mismatch.  Taking
    each side's difference from the same full-precision baseline cancels the
    first.  What is left is the perturbation the quantizer introduced, and the
    two implementations should agree on it.

    ``effect_size``/``std`` say whether the fake quantizer damages the model by
    the right *amount*; ``corr`` says whether it damages the *same tokens*.
    A faithful reimplementation matches on both.  Matching magnitude with
    partial correlation is the signature of an operator that is right on
    average but resolves individual near-boundary cases differently -- which is
    what a discontinuous 3-bit grid does under any input perturbation.
    """
    dv = [a - b for a, b in zip(vllm_lp, hf_fp)]
    dh = [a - b for a, b in zip(hf_fq, hf_fp)]
    n = len(dv)
    mv, mh = sum(dv) / n, sum(dh) / n
    vv = sum((x - mv) ** 2 for x in dv)
    vh = sum((x - mh) ** 2 for x in dh)
    cov = sum((x - mv) * (y - mh) for x, y in zip(dv, dh))
    denom = math.sqrt(vv * vh)
    return {
        "n": n,
        # E_{x~rollout}[log p_actor(x) - log p_rollout(x)] = -KL(rollout||actor),
        # so this is <= 0 by construction and is 0 only if the two policies
        # match.  It is the same number the training metric reports as
        # qat/rollout_logprob_bias.
        "bias_actor_minus_vllm": sum(hf_fq[i] - vllm_lp[i] for i in range(len(vllm_lp)))
        / len(vllm_lp),
        "corr": (cov / denom) if denom > 0 else float("nan"),
        "regression_slope_fake_to_vllm": (cov / vh) if vh > 0 else float("nan"),
        "vllm_effect_size": sum(abs(x) for x in dv) / n,
        "fake_effect_size": sum(abs(x) for x in dh) / n,
        "vllm_effect_std": math.sqrt(vv / n),
        "fake_effect_std": math.sqrt(vh / n),
    }


def phase_report(args) -> dict:
    quantized = json.loads(Path(args.quantized).read_text())
    control = json.loads(Path(args.control).read_text())

    ref = quantized["vllm_logprobs"]
    comparisons = {
        name: _stats(ref, values) for name, values in quantized["variants"].items()
    }
    control_stats = _stats(control["vllm_logprobs"], control["variants"]["fp"])

    noise_floor = control_stats["mean_abs_diff"]
    best_name = min(
        (n for n in comparisons if n.startswith("fq_")),
        key=lambda n: comparisons[n]["mean_abs_diff"],
        default=None,
    )

    verdict = {}
    if best_name is not None:
        best = comparisons[best_name]["mean_abs_diff"]
        # How much of the vLLM-vs-HF gap does the fake quantizer actually
        # remove?  Comparing the quantized engine against an unquantized HF
        # forward is the "do nothing" baseline.
        do_nothing = comparisons["fp"]["mean_abs_diff"]
        verdict = {
            "best_variant": best_name,
            "fp_control_noise_floor": noise_floor,
            "best_variant_mean_abs_diff": best,
            "no_quantizer_mean_abs_diff": do_nothing,
            "ratio_to_noise_floor": (best / noise_floor) if noise_floor > 0 else None,
            "fraction_of_gap_explained": (
                (do_nothing - best) / do_nothing if do_nothing > 0 else None
            ),
            "prefill_asymmetry_gain": (
                comparisons["fq_uniform"]["mean_abs_diff"]
                - comparisons["fq_prefill_exact"]["mean_abs_diff"]
            )
            if "fq_prefill_exact" in comparisons and "fq_uniform" in comparisons
            else None,
        }

    fp_baseline = quantized["variants"]["fp"]
    delta = {
        name: _delta_agreement(ref, values, fp_baseline)
        for name, values in quantized["variants"].items()
        if name.startswith("fq_")
    }

    return {
        "phase": "report",
        "quantization_delta_agreement": delta,
        "kv_scheme": quantized["kv_scheme"],
        "hf_dtype": quantized["hf_dtype"],
        "num_scored_tokens": quantized["num_scored_tokens"],
        "prompt_len": quantized["prompt_len"],
        "quantized_run_comparisons": comparisons,
        "fp_control_comparison": control_stats,
        "verdict": verdict,
        "vllm_verification": quantized.get("vllm_verification"),
    }


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("vllm", "hf", "report"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--kv-scheme", default="turboquant_k3v4_nc")
    parser.add_argument("--prompt-parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--prompt-row", type=int, default=0)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--hf-dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--pad-left",
        type=int,
        default=0,
        help="left-pad the sequence by this many tokens in the HF phase, "
        "reproducing veRL's prompt padding (data.max_prompt_length)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokens", default=None, help="vllm-phase json (hf phase)")
    parser.add_argument("--quantized", default=None, help="hf-phase json (report)")
    parser.add_argument("--control", default=None, help="fp hf-phase json (report)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.phase == "hf" and not args.tokens:
        parser.error("--phase hf requires --tokens")
    if args.phase == "report" and not (args.quantized and args.control):
        parser.error("--phase report requires --quantized and --control")

    result = {"vllm": phase_vllm, "hf": phase_hf, "report": phase_report}[args.phase](args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[parity] wrote {out}")
    if args.phase == "report":
        print(json.dumps(result["verdict"], indent=2))
        print("delta agreement:", json.dumps(result["quantization_delta_agreement"], indent=2))
        print(json.dumps(result["quantized_run_comparisons"], indent=2))
        print("fp control:", json.dumps(result["fp_control_comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
