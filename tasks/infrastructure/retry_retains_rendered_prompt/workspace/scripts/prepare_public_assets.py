"""Initialize a small PUBLIC Qwen2 architecture; no download or private data."""
import argparse
import json
from pathlib import Path


def prepare(destination):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261115)
    words = ["<unk>", "<pad>", "<eos>", "user", "assistant", "Write", "a", "short", "sequence", "of",
             "red", "blue", "green", "tokens", ".", ":"]
    words += [f"word{i}" for i in range(256 - len(words))]
    raw = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tokenizer.chat_template = "{% for m in messages %}{{ m['role'] + ' : ' + m['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant : {% endif %}"
    tokenizer.save_pretrained(destination)
    config = Qwen2Config(vocab_size=256, hidden_size=128, intermediate_size=256,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=256, rms_norm_eps=1e-6, rope_theta=10000.0,
                         tie_word_embeddings=False, use_sliding_window=False,
                         bos_token_id=None, eos_token_id=2, pad_token_id=1)
    model = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16)
    model.save_pretrained(destination, safe_serialization=True)
    (destination / "public_asset.json").write_text(json.dumps({
        "architecture": "transformers.Qwen2ForCausalLM", "initialization": "random", "seed": 20261115, "dtype": "bfloat16",
        "parameters": sum(p.numel() for p in model.parameters()), "private_assets": False,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=Path(__file__).resolve().parents[1] / "assets/public-qwen2")
    prepare(parser.parse_args().dest)
