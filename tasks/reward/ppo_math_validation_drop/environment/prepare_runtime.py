"""Stage the pinned pure-Python training and reward implementation."""
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ASSETS = (
    (
        "https://files.pythonhosted.org/packages/02/ca/2eaa5359f2ccb8c2e1656bc26305ad0cf438aa392ce4b29ae67a315c186e/transformers-5.9.0-py3-none-any.whl",
        10787648,
        "1d19509bcff7028ebc6b277d71caa712e8353778463d38764237d14b42b52788",
        "wheel",
    ),
    (
        "https://files.pythonhosted.org/packages/a8/db/253133d7e7cb40d3af384bb2f5c0b4a2b7fdcffbc95c688cc67a20a3c103/accelerate-1.14.0-py3-none-any.whl",
        389246,
        "e94390c2863b873be18f623f9df48a0d8fe5eff13ea7f1a00092b0a7904888c6",
        "wheel",
    ),
    (
        "https://codeload.github.com/verl-project/verl/tar.gz/1876b06d0a3e4e71e06230be10af14492ca8a75b",
        2693568,
        "073302fd349ef8479571f88d6bd4022e60a6bf189df6396809d80f70d40c4b37",
        "verl",
    ),
    (
        "https://files.pythonhosted.org/packages/62/76/6b4969bccc842b6567f7e6ee015684b9428a9b7fcbdf479e73716f43597f/math_verify-0.9.0-py3-none-any.whl",
        29967,
        "3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194",
        "wheel",
    ),
    (
        "https://files.pythonhosted.org/packages/e9/61/f75cd1fa54d8434276126034aed54dd120747de9a8fa013cdd79545ccbeb/latex2sympy2_extended-1.11.0-py3-none-any.whl",
        209050,
        "aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234",
        "wheel",
    ),
    (
        "https://files.pythonhosted.org/packages/54/a5/91d16d65b66251c2423bfedb9219c2e803f11d91af9bfba75f05ddc33bfb/mathruler-0.1.0-py3-none-any.whl",
        15190,
        "e8ce0d9e07e892bf971fbff5e6eee330f6a66fd022666c99828a0000c788079f",
        "wheel",
    ),
    (
        "https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/antlr4-python3-runtime-4.9.3.tar.gz",
        117034,
        "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b",
        "src/antlr4",
    ),
)


def stage_asset(url, size, digest, kind, target):
    with tempfile.TemporaryFile() as archive:
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read(size + 1)
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"Unexpected upstream bytes: {url}")
        archive.write(payload)
        archive.seek(0)
        if kind == "wheel":
            with zipfile.ZipFile(archive) as wheel:
                wheel.extractall(target)
            return
        with tempfile.TemporaryDirectory() as extracted:
            with tarfile.open(fileobj=archive) as tar:
                tar.extractall(extracted, filter="data")
            (tree,) = Path(extracted).iterdir()
            source = tree / kind
            if not source.is_dir() and kind == "src/antlr4":
                source = tree / "antlr4"
            shutil.copytree(source, target / source.name, dirs_exist_ok=True)


def main():
    target = Path(os.environ["AUTORLBENCH_PREPARED"]) / "pysite"
    target.mkdir(parents=True, exist_ok=True)
    for asset in ASSETS:
        stage_asset(*asset, target)
    apply_deepseek_tokenizer_compatibility(target)
    print(f"Staged {len(ASSETS)} verified runtime assets", file=sys.stderr)


def apply_deepseek_tokenizer_compatibility(target):
    path = target / "verl/utils/tokenizer/continuous_token_wiring.py"
    source = path.read_text(encoding="utf-8")
    original = "    has_mm_processor = _is_multimodal_processor(processor)\n    resolved_family = resolve_continuous_token_model_family("
    historical = """    has_mm_processor = _is_multimodal_processor(processor)
    # R1-Distill-Qwen keeps Qwen2 layers but uses DeepSeek's tokenizer/template.
    # Architecture alone therefore cannot select its token boundary handling.
    if (
        _normalize_model_family(model_family) == ContinuousTokenModelFamily.AUTO
        and _normalize_hf_model_type(hf_model_type) == "qwen2"
        and not has_mm_processor
        and getattr(tokenizer, "eos_token", None) == DeepSeekContinuousTokenBuilder._EOS_TOKEN
    ):
        model_family = ContinuousTokenModelFamily.DEEPSEEK
    resolved_family = resolve_continuous_token_model_family("""
    if source.count(original) != 1:
        raise ValueError("Unexpected VeRL tokenizer dispatch source")
    path.write_text(source.replace(original, historical), encoding="utf-8")


if __name__ == "__main__":
    main()
