"""Build the public repository-debugging workload used by the rollout.

Every task is a small synthetic Python project with one planted defect and a
failing unit test.  The agent has to inspect the project through tools, so a
trajectory is a genuine multi-turn conversation whose context grows with every
file it reads.  The generator is deterministic: the same seed always yields the
same JSONL bytes, and `--check` compares them against the recorded digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

DOMAINS = {
    "inventory": ("item", "stock", "warehouse", "reorder", "sku"),
    "billing": ("invoice", "charge", "customer", "refund", "ledger"),
    "scheduler": ("job", "slot", "worker", "deadline", "queue"),
    "telemetry": ("sample", "metric", "window", "bucket", "series"),
    "catalog": ("product", "price", "category", "discount", "listing"),
    "routing": ("route", "hop", "segment", "carrier", "zone"),
}
# Each defect: (correct expression, broken expression, failing behaviour).
DEFECTS = [
    ("total += value", "total -= value", "accumulates the wrong sign"),
    ("if count >= limit:", "if count > limit:", "misses the boundary value"),
    ("return values[: size]", "return values[: size - 1]", "drops the last element"),
    ("ratio = part / whole if whole else 0.0", "ratio = part / (whole + 1) if whole else 0.0", "skews the ratio"),
    ("key = name.strip().lower()", "key = name.strip()", "treats differently-cased names as distinct"),
    ("start = max(0, offset)", "start = offset", "accepts negative offsets"),
    ("best = min(candidates, key=cost)", "best = max(candidates, key=cost)", "selects the most expensive candidate"),
    ("window = items[index - span + 1 : index + 1]", "window = items[index - span : index]", "shifts the window by one"),
]
SYSTEM = (
    "You are a software engineer working in a sandboxed repository. Use the provided tools to "
    "inspect files and run the test-suite. When you have identified the defect, call submit_fix "
    "with the file path, the function name and a short explanation. Do not guess without reading code."
)


def _helper(rng: random.Random, noun: str, idx: int) -> str:
    verb = rng.choice(["normalize", "collect", "merge", "validate", "render", "index", "partition", "score"])
    lines = [
        f"def {verb}_{noun}_{idx}(records, *, strict=False):",
        f'    """{verb.capitalize()} {noun} records (helper #{idx})."""',
        "    out = []",
        "    for position, record in enumerate(records):",
        f"        if strict and not isinstance(record, dict):",
        f"            raise TypeError(f'{noun} record {{position}} must be a mapping')",
        f"        entry = dict(record)",
        f"        entry.setdefault('{noun}_id', position)",
        f"        entry['{verb}_done'] = True",
        "        out.append(entry)",
        "    return out",
        "",
    ]
    return "\n".join(lines)


def _target(defect: tuple[str, str, str], noun: str, fname: str, broken: bool) -> str:
    good, bad, _ = defect
    line = bad if broken else good
    body = {
        0: [f"def {fname}(values):", "    total = 0", "    for value in values:", f"        {line}", "    return total"],
        1: [f"def {fname}(count, limit):", f"    {line}", "        return True", "    return False"],
        2: [f"def {fname}(values, size):", "    values = list(values)", f"    {line}"],
        3: [f"def {fname}(part, whole):", f"    {line}", "    return round(ratio, 6)"],
        4: [f"def {fname}(names):", "    seen = {}", "    for name in names:", f"        {line}", "        seen[key] = seen.get(key, 0) + 1", "    return seen"],
        5: [f"def {fname}(items, offset, size):", f"    {line}", "    return items[start : start + size]"],
        6: [f"def {fname}(candidates, cost):", "    if not candidates:", "        return None", f"    {line}", "    return best"],
        7: [f"def {fname}(items, index, span):", f"    {line}", "    return sum(window) / len(window)"],
    }[DEFECTS.index(defect)]
    doc = f'    """Core {noun} computation used by the public API."""'
    return "\n".join([body[0], doc, *body[1:], ""])


TESTS = {
    0: ("assert {f}([1, 2, 3]) == 6", "AssertionError: assert -6 == 6"),
    1: ("assert {f}(5, 5) is True", "AssertionError: assert False is True"),
    2: ("assert {f}([1, 2, 3], 3) == [1, 2, 3]", "AssertionError: assert [1, 2] == [1, 2, 3]"),
    3: ("assert {f}(1, 4) == 0.25", "AssertionError: assert 0.2 == 0.25"),
    4: ("assert {f}(['A', 'a']) == {{'a': 2}}", "AssertionError: assert {'A': 1, 'a': 1} == {'a': 2}"),
    5: ("assert {f}([1, 2, 3], -1, 2) == [1, 2]", "AssertionError: assert [] == [1, 2]"),
    6: ("assert {f}([3, 1, 2], cost=lambda x: x) == 1", "AssertionError: assert 3 == 1"),
    7: ("assert {f}([1, 2, 3, 4], 3, 2) == 3.5", "AssertionError: assert 2.5 == 3.5"),
}


def build_task(task_id: int, seed: int) -> dict:
    rng = random.Random(seed * 100003 + task_id)
    domain = rng.choice(sorted(DOMAINS))
    nouns = DOMAINS[domain]
    defect_index = rng.randrange(len(DEFECTS))
    defect = DEFECTS[defect_index]
    modules = rng.sample(["core", "models", "service", "storage", "utils", "reports", "api", "cache"], 6)
    target_module = rng.choice(modules)
    target_noun = rng.choice(nouns)
    target_fn = f"compute_{target_noun}_{rng.choice(['total', 'window', 'ratio', 'selection', 'keys', 'page', 'limit'])}"
    files: dict[str, str] = {}
    for module in modules:
        chunks = [f'"""{domain}.{module}: {module} helpers for {domain} records."""', "", "from __future__ import annotations", ""]
        for idx in range(rng.randint(5, 9)):
            chunks.append(_helper(rng, rng.choice(nouns), idx))
        if module == target_module:
            chunks.insert(rng.randint(4, len(chunks)), _target(defect, target_noun, target_fn, broken=True))
        else:
            decoy = f"compute_{rng.choice(nouns)}_{rng.choice(['summary', 'batch', 'view'])}"
            chunks.append(_target(rng.choice(DEFECTS), rng.choice(nouns), decoy, broken=False))
        files[f"{domain}/{module}.py"] = "\n".join(chunks)
    test_src, failure = TESTS[defect_index]
    files[f"tests/test_{target_module}.py"] = "\n".join([
        f"from {domain}.{target_module} import {target_fn}", "", "",
        f"def test_{target_fn}():", "    " + test_src.format(f=target_fn), "",
    ])
    files["README.md"] = f"# {domain}\n\nInternal {domain} toolkit. Run `pytest -q` to execute the test-suite.\n"
    path = f"{domain}/{target_module}.py"
    user = (f"The test-suite of the `{domain}` repository is failing. Find the defect that makes it fail. "
            f"Inspect the repository with the tools before answering, then call submit_fix.")
    return {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "label": json.dumps({"path": path, "function": target_fn}, sort_keys=True),
        "metadata": {
            "task_id": f"repo-{task_id:04d}",
            "files": files,
            "test_failure": f"FAILED tests/test_{target_module}.py::test_{target_fn} - {failure}",
            "defect_class": defect[2],
        },
    }


def build(count: int, seed: int) -> bytes:
    rows = [json.dumps(build_task(i, seed), sort_keys=True, ensure_ascii=False) for i in range(count)]
    return ("\n".join(rows) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--sha256", default=None, help="expected digest of the generated JSONL")
    options = parser.parse_args()
    data = build(options.count, options.seed)
    digest = hashlib.sha256(data).hexdigest()
    if options.sha256 and digest != options.sha256:
        raise SystemExit(f"workload digest mismatch: {digest} != {options.sha256}")
    options.out.parent.mkdir(parents=True, exist_ok=True)
    options.out.write_bytes(data)
    print(json.dumps({"tasks": options.count, "bytes": len(data), "sha256": digest, "path": str(options.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
