"""Fetch the immutable public inputs once. The paired training runs can then be offline."""
import argparse
from hashlib import sha256
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

from cache_backend import prepare_sources

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
REVISION = "d039ab0d4b9311b617e8ffec1d06c8464364aeb3"
MODEL_FILES = {
    "added_tokens.json": "74135b8664b56088c0006f1c8e848d79a8eba003411f72ebf1dc2ee96227be3a",
    "chat_template.json": "12462caec000542ca1bb73cd0ba25aaf320d90335fc46731fa6a8355ea34bd26",
    "config.json": "b70fb4bfde88df9eeebc9d8ff523733b8bf70d6c9b06c610325960e06ae9db52",
    "generation_config.json": "067a2a54e5f87162ecac6e0e911cc4665fc8f7f3324794ecbac0f76badb56636",
    "merges.txt": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510",
    "model.safetensors": "74dea5904032e5ae99a2e0eef5179e6ac0f1dedc3ab0c7c2a5d4d387c843203e",
    "preprocessor_config.json": "6cb6e36d6fcb88ca1502c4a26750715dc3e7dedddc9a8f17b27d8d167d1457e7",
    "processor_config.json": "e7bff42da73ae9eec9042ef20e066e11f1ee20f025358ff79131e3c0fb549b46",
    "special_tokens_map.json": "aa0ff906077086dfa9734a7f97f68c825877a48f9468807be65504495cdeef09",
    "tokenizer.json": "5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a",
    "tokenizer_config.json": "8a68fb58021bc3df79d70f3d5fa79d2e709083872be1f6965cc9bcd12f3ba66a",
    "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
}


def fetch(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with path.open("rb") as src:
            h = sha256()
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch in existing {path.name}")
        return
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urlopen(url, timeout=120) as source:
                shutil.copyfileobj(source, tmp)
            tmp.flush()
            with tmp_path.open("rb") as src:
                h = sha256()
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError(f"SHA-256 mismatch downloading {path.name}")
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    args = cli.parse_args()
    prepare_sources(args.directory / "upstream")
    for name, digest in MODEL_FILES.items():
        fetch(f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}",
              args.directory / "smolvlm" / name, digest)
    print(f"Verified public SmolVLM and SGLang inputs in {args.directory}")
