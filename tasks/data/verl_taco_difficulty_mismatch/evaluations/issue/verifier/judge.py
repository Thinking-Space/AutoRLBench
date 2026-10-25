#!/usr/bin/env python3
"""Judge an issue diagnosis against the hidden oracle with an OpenAI API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


WEIGHTS = {
    "verdict_score": 0.2,
    "root_cause_score": 0.6,
    "evidence_score": 0.2,
}
PASS_SCORE = 0.8


class JudgeError(RuntimeError):
    pass


_TOO_HARD = re.compile(
    r"訓練集過難|訓練集太難|訓練資料過難|訓練資料太難|訓練數據過難|訓練數據太難|"
    r"too hard|overly hard|too difficult|overly difficult|"
    r"excessively hard|excessively difficult|"
    r"harder than the (?:validation|held-?out|test)",
    re.IGNORECASE,
)
_MISMATCH = re.compile(
    r"難度不匹配|難度不一致|難度不對齊|難度不match|"
    r"difficulty mismatch|difficulty gap|difficulty discrepancy|"
    r"difficult(?:y|ies).{0,48}(?:do not|don't|doesn't|not) match|"
    r"(?:do not|don't|doesn't|not) match.{0,48}difficult|"
    r"train(?:ing)?(?:\s+set|\s+data)?.{0,40}(?:val(?:idation)?|held-?out|test).{0,40}difficult|"
    r"(?:val(?:idation)?|held-?out|test).{0,40}train(?:ing)?.{0,40}difficult",
    re.IGNORECASE,
)
_LABELS = re.compile(r"very[\s_-]?hard", re.IGNORECASE)


def root_cause_hit(answer: str) -> bool:
    """Full root-cause credit when either required formulation is present."""
    if _TOO_HARD.search(answer) or _MISMATCH.search(answer):
        return True
    lowered = answer.lower()
    return _LABELS.search(answer) is not None and "easy" in lowered


def read_required(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise JudgeError(f"missing or invalid {label}: {path}")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError as exc:
        raise JudgeError(f"{label} must be UTF-8") from exc
    if not text:
        raise JudgeError(f"{label} is empty")
    return text


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise JudgeError("judge response did not contain a JSON object")


def validate_judgment(value: Any) -> dict[str, Any]:
    expected = {*WEIGHTS, "reason"}
    if not isinstance(value, dict) or set(value) != expected:
        raise JudgeError(
            "judge response keys must be verdict_score, root_cause_score, "
            "evidence_score, and reason"
        )
    for key in WEIGHTS:
        score = value[key]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise JudgeError(f"{key} must be a number")
        if not 0.0 <= float(score) <= 1.0:
            raise JudgeError(f"{key} must be between 0 and 1")
        value[key] = float(score)
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise JudgeError("reason must be a non-empty string")
    value["reason"] = value["reason"].strip()
    return value


def call_judge(
    *,
    prompt: str,
    oracle: str,
    answer: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "<canonical_oracle>\n"
                    + oracle
                    + "\n</canonical_oracle>\n\n<agent_diagnosis>\n"
                    + answer
                    + "\n</agent_diagnosis>"
                ),
            },
        ],
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        chat_completions_url(base_url),
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        content = response_payload["choices"][0]["message"]["content"]
    except (
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        raise JudgeError(f"OpenAI judge request failed: {exc}") from exc
    if not isinstance(content, str):
        raise JudgeError("OpenAI judge response content must be text")
    return validate_judgment(extract_json_object(content))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    verifier = Path(os.environ.get("AUTORLBENCH_VERIFIER", Path(__file__).parent))
    results = Path(os.environ.get("AUTORLBENCH_RESULTS", "/results"))
    oracle = Path(os.environ.get("AUTORLBENCH_ORACLE", "/task/oracle.md"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answer", type=Path, default=results / "agent_answer.md")
    parser.add_argument("--oracle", type=Path, default=oracle)
    parser.add_argument("--prompt", type=Path, default=verifier / "judge_prompt.md")
    parser.add_argument("--result", type=Path, default=results / "result.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        answer = read_required(args.answer.resolve(), "agent_answer.md")
        oracle = read_required(args.oracle.resolve(), "oracle.md")
        prompt = read_required(args.prompt.resolve(), "judge_prompt.md")
        model = os.environ.get("AUTORLBENCH_JUDGE_MODEL", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not model or not base_url or not api_key:
            raise JudgeError(
                "AUTORLBENCH_JUDGE_MODEL, OPENAI_BASE_URL, and OPENAI_API_KEY "
                "are required"
            )
        timeout_s = float(os.environ.get("AUTORLBENCH_JUDGE_TIMEOUT_S", "60"))
        judgment = call_judge(
            prompt=prompt,
            oracle=oracle,
            answer=answer,
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout_s=timeout_s,
        )
        matched = root_cause_hit(answer)
        if matched and judgment["root_cause_score"] != 1.0:
            judgment["root_cause_score"] = 1.0
            judgment["reason"] += (
                " Root-cause credit is full because the diagnosis identifies a "
                "training set that is too hard or a train/validation difficulty "
                "mismatch."
            )
        elif matched:
            judgment["root_cause_score"] = 1.0
        # New scoring logic: verdict must be 1.0, root_cause and evidence must be > 0.8
        passed = (
            judgment["verdict_score"] == 1.0
            and judgment["root_cause_score"] > 0.8
            and judgment["evidence_score"] > 0.8
        )
        score = sum(judgment[key] * weight for key, weight in WEIGHTS.items())
        score = round(score, 6)
        result = {
            "passed": passed,
            "score": score,
            "metrics": {
                "status": "judged",
                "judge_model": model,
                "root_cause_rule": "matched" if matched else "unmatched",
                **judgment,
            },
        }
        write_json(args.result.resolve(), result)
        return 0 if result["passed"] else 1
    except (JudgeError, OSError, ValueError) as exc:
        result = {
            "passed": False,
            "score": None,
            "metrics": {"status": "unavailable", "error": str(exc)},
        }
        try:
            write_json(args.result.resolve(), result)
        except OSError:
            pass
        print(f"issue judge unavailable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
