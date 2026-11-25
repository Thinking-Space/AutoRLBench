#!/usr/bin/env python3
"""Apply the image-compatibility guards to the packaged miles tree.

The vendored tree is pinned to a commit slightly newer than the shared
image's sglang (0.5.20.dev59): four references expect ServerArgs fields or
symbols that postdate the image. Each guard is a behavior-neutral fallback
for this task's model and topology (single-engine Qwen3 runs), documented in
environment/source_manifest.json, and identical in every arm.

Run from the task root: python3 environment/apply_compat_guards.py
"""

from __future__ import annotations

import sys
from pathlib import Path

TREE = Path(__file__).resolve().parent.parent / "workspace" / "miles"

GUARDS = [
    # 1. sglang.srt.entrypoints.openai lacks encoding_dsv41 on this image; the
    #    dsv41 path is DeepSeek-V4.1-template-only and unused on this model.
    (
        "miles/utils/chat_template_utils/deepseek.py",
        "from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv4, encoding_dsv41",
        """try:
    from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv4, encoding_dsv41
except ImportError:
    from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv4

    encoding_dsv41 = encoding_dsv4  # image sglang predates dsv41; unused on this model""",
    ),
    # 2./3. ServerArgs.attn_dp_size postdates the image; single-engine runs
    #    never enable attention DP, so the absent field means "disabled".
    (
        "miles/backends/sglang_utils/arguments.py",
        "attn_dp_size=args.sglang_attn_dp_size,",
        'attn_dp_size=getattr(args, "sglang_attn_dp_size", 1),  # image sglang predates this ServerArgs field',
    ),
    (
        "miles/utils/lora/arguments.py",
        "attn_dp_size=args.sglang_attn_dp_size,",
        'attn_dp_size=getattr(args, "sglang_attn_dp_size", 1),  # image sglang predates this ServerArgs field',
    ),
    # 4. enable_prefill_weight_versions postdates the image; the assertion
    #    path cannot be enabled on this stack.
    (
        "miles/utils/weight_version.py",
        '        if args.sglang.common_value("enable_prefill_weight_versions"):',
        """        try:
            prefill_versions_on = args.sglang.common_value("enable_prefill_weight_versions")
        except AttributeError:
            prefill_versions_on = False  # image sglang predates this ServerArgs field
        if prefill_versions_on:""",
    ),
    # 5. The ray-job entrypoint hard-sets no_proxy=127.0.0.1 for the driver.
    #    Pod egress proxies refuse pod-IP destinations and the stack addresses
    #    engines and gates by pod IP, so the driver's exemptions must cover
    #    this host's own IPs, resolved here (shell substitution does not
    #    survive the job-entrypoint quoting layers).
    (
        "miles/utils/external_utils/ray_job.py",
        '"export no_proxy=127.0.0.1 && export PYTHONUNBUFFERED=1 && "',
        '''"export no_proxy=127.0.0.1,localhost,"
            + _pod_ips_csv()
            + " && export NO_PROXY=$no_proxy && export PYTHONUNBUFFERED=1 && "''',
    ),
]

POD_IP_HELPER = '''
def _pod_ips_csv() -> str:
    """This host's own IPs, for the job driver's proxy exemptions."""
    import socket

    ips = {"127.0.0.1"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            ips.add(info[4][0])
    except socket.gaierror:
        pass
    return ",".join(sorted(ips))


def run_ray_job('''


def main() -> int:
    for rel, old, new in GUARDS:
        path = TREE / rel
        text = path.read_text(encoding="utf-8")
        marker = next((ln for ln in new.splitlines() if "#" in ln), new.splitlines()[0]).strip()
        if marker in text:
            print(f"already guarded: {rel}")
            continue
        if old not in text:
            print(f"PATTERN MISSING: {rel}: {old[:60]!r}", file=sys.stderr)
            return 1
        text = text.replace(old, new, 1)
        if "run_ray_job" in rel and "_pod_ips_csv" not in text:
            text = text.replace("\ndef run_ray_job(", POD_IP_HELPER, 1)
        path.write_text(text, encoding="utf-8")
        print(f"guarded: {rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
