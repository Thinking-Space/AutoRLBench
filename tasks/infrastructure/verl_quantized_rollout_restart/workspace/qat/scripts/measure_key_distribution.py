"""Do real post-Hadamard keys deviate from the Gaussian Lloyd-Max assumes?

TurboQuant rotates unit-normalised keys with a Hadamard matrix specifically so
the coordinates become approximately N(0, 1/d), which is the distribution its
fixed Lloyd-Max table is optimal for.  On synthetic keys the rotation does this
almost perfectly (kurtosis 2.93 vs 3.0) and a learnable table recovers nothing.
But synthetic keys are not the question.  This script measures the real ones and
then asks how much a learnable table could recover from them.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from qat_kv.turboquant_fake_quant import (
    TurboQuantFakeQuantConfig, turboquant_key_rotate, lloyd_max_centroids)
from qat_kv.learnable_centroids import LearnableKeyCentroids


def fit_centroids(sample, bits, steps=400, lr=1e-2):
    fixed = lloyd_max_centroids(sample.new_zeros(1).numel() or 128, bits)
    return fixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/historical/shared/qat/models/base/deepseek-r1-distill-qwen-7b")
    ap.add_argument("--data", default="/historical/shared/qat/data/processed/skywork_or1_7b_math_difficulty1-15_16k.parquet")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = pq.read_table(args.data).to_pylist()[: args.n_prompts]
    texts = []
    for r in rows:
        p = r["prompt"]
        if isinstance(p, list):
            p = tok.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        texts.append(str(p))
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=args.max_tokens)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa").cuda().eval()
    enc = {k: v.cuda() for k, v in enc.items()}

    # Capture post-RoPE keys the same way the quantizer sees them: hook the
    # attention interface, which receives K after rotary embedding.
    captured = []
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    orig = qwen2.eager_attention_forward if hasattr(qwen2, "eager_attention_forward") else None

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    base = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def capture(module, query, key, value, attention_mask, **kw):
        captured.append(key.detach().float().reshape(-1, key.shape[-1]).cpu())
        return base(module, query, key, value, attention_mask, **kw)

    ALL_ATTENTION_FUNCTIONS.register("capture_probe", capture)
    model.config._attn_implementation = "capture_probe"
    with torch.no_grad():
        model(**enc, use_cache=False)

    K = torch.cat(captured, dim=0)
    d = K.shape[-1]
    cfg = TurboQuantFakeQuantConfig(head_dim=d, key_quant_bits=args.bits,
                                    value_quant_bits=4, norm_correction=True)
    y, _ = turboquant_key_rotate(K, cfg)
    f = y.reshape(-1)
    m, s = f.mean().item(), f.std().item()
    kurt = float(((f - m) ** 4).mean() / s ** 4)
    skew = float(((f - m) ** 3).mean() / s ** 3)

    # How much could a learnable table recover on this actual distribution?
    fixed = lloyd_max_centroids(d, args.bits)
    sample = f[torch.randperm(f.numel())[:400000]]

    def mse(c):
        mids = (c[1:] + c[:-1]) / 2
        idx = torch.searchsorted(mids.contiguous(), sample.contiguous(), right=True)
        return (c[idx.clamp(max=c.numel() - 1)] - sample).pow(2).mean()

    baseline = float(mse(fixed))
    mod = LearnableKeyCentroids(fixed)
    opt = torch.optim.Adam(mod.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad(); loss = mse(mod()); loss.backward(); opt.step()
    learned = float(loss)

    out = {
        "model": args.model, "head_dim": d, "bits": args.bits,
        "n_key_vectors": int(K.shape[0]), "n_coords": int(f.numel()),
        "post_hadamard": {"std": s, "expected_std": d ** -0.5,
                          "kurtosis": kurt, "gaussian_kurtosis": 3.0, "skew": skew},
        "centroid_fit": {"lloyd_max_mse": baseline, "learned_mse": learned,
                         "relative_change_pct": 100 * (learned / baseline - 1)},
        "learned_table": [float(x) for x in mod().detach()],
        "lloyd_max_table": [float(x) for x in fixed.sort().values],
        "drift": mod.drift(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
