"""Build/check the text-complete proxy training set, without model inference.

Run with --write to regenerate the active parquet and its manifest. With no
arguments, verify the frozen bytes, row contents, multiplicities and answers.
Requires pyarrow; all mathematical checks use only the standard library.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import combinations, permutations, product
import json
from math import isclose, sqrt
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[3]
ASSETS = TASK / "workspace/autorlbench_case/assets"
ACTIVE = ASSETS / "train_proxy_fixed16_x64_v3.parquet"
MANIFEST = HERE / "v3_manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_answers(cases):
    answers = [c["ground_truth"] for c in cases]
    # The reviewed formulas are attached to the exact question bytes in cases.
    assert answers == ["90", "6", "$6^7$", "8", "36", "16", "0.73r", "56", "100",
                       r"$3\sqrt{3}$", r"2\sqrt{2}+2", r"\frac{31}{300}", "90", "36",
                       r"\frac{\pi}{7}", r"\frac{1}{3}"]
    assert (2 + 4) * (360 // 12) // 2 == 90  # Inscribed angles.
    vertices = list(product([0, 1], repeat=3))
    faces = [[i for i, v in enumerate(vertices) if v[axis] == side]
             for axis in range(3) for side in (0, 1)]
    labelings = sum(len({sum(p[i] for i in face) for face in faces}) == 1
                    for p in permutations(range(1, 9)))
    assert labelings == 144 and labelings // 24 == 6
    assert sum([6**6] * 6) == 6**7

    def trailing_zeroes(n):
        total = 0
        while n:
            n //= 5
            total += n
        return total

    first = [n for n in range(5, 15) if trailing_zeroes(2*n) == 3*trailing_zeroes(n)]
    assert first == [8, 9, 13, 14] and sum(map(int, str(sum(first)))) == 8
    assert (6 + 1)**2 + 1 == 50 and 6**2 == 36
    points = list(product(range(3), repeat=2))
    lines = [lambda p: p[0] == 0, lambda p: p[0] == 2,
             lambda p: p[1] == 0, lambda p: p[1] == 2,
             lambda p: p[0] == 1, lambda p: p[1] == 1,
             lambda p: p[0] == p[1], lambda p: p[0]+p[1] == 2]
    triangles = 0
    for a, b, c in combinations(points, 3):
        area2 = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        if area2 and all(any(line(u) and line(v) for line in lines)
                         for u, v in combinations((a, b, c), 2)):
            triangles += 1
    assert triangles == 16
    assert round(sqrt(3)-1, 2) == 0.73

    def polygon_area(vertices):
        return abs(sum(x*v-y*u for (x, y), (u, v)
                       in zip(vertices, vertices[1:]+vertices[:1]))) / 2

    # The embedded 3x3 Asymptote tile has four congruent pentagons of area 1.25.
    assert polygon_area([(1, 2), (1, 3), (2, 3), (2, 2), (1.5, 1.5)]) == 1.25
    assert round(100 * 5 / 9) == 56
    assert (180 + 180/9) / 2 == 100
    assert isclose(sqrt(3)/4 * (2*sqrt(3))**2, 3*sqrt(3))
    ratio = 2 + 2*sqrt(2)
    assert isclose((1-sqrt(2)/ratio)**2, 0.5)
    assert Fraction(sum(j % i == 0 for i, j in permutations(range(1, 26), 2)), 600) == Fraction(31, 300)
    assert sum((b-d)*(a-c) < 0 for a, b, c, d in permutations(range(1, 7), 4)) // 2 == 90
    assert 36 * 10 == 360  # Star interior-angle sum.
    # Work in units of pi: shaded area=3+3*(theta/pi), total area=9.
    shaded = 3 + 3*Fraction(1, 7)
    assert shaded/(9-shaded) == Fraction(8, 13)
    assert polygon_area([(-2, 1), (2, 4), (0, 4), (0, 3), (-2, 3)]) / 12 == 1/3


def build(write=False):
    definition = json.loads((HERE / "v3_cases.json").read_text())
    cases = definition["cases"]
    verify_answers(cases)
    source = HERE / "v2/train_proxy_fixed16_x64.parquet"
    assert sha(source) == definition["source_sha256"]
    original = pq.read_table(source)
    by_index = {case["source_index"]: case for case in cases}
    expected = original.to_pylist()
    for row in expected:
        case = by_index[row["extra_info"]["index"]]
        row["prompt"][0]["content"] = case["question"] + r" Please output the final answer within \boxed{}."
        row["extra_info"]["original_question"] = case["question"]
        assert row["reward_model"]["ground_truth"] == case["ground_truth"]
    if write:
        pq.write_table(pa.Table.from_pylist(expected, schema=original.schema), ACTIVE)
    actual = pq.read_table(ACTIVE).to_pylist()
    assert actual == expected
    counts = Counter(row["prompt"][0]["content"] for row in actual)
    assert len(counts) == 16 and set(counts.values()) == {64}
    train_questions = {row["extra_info"]["original_question"].strip() for row in actual}
    for path in [ASSETS / "aime32.parquet", ASSETS / "math500_32.parquet"]:
        val = pq.read_table(path).to_pylist()
        assert not train_questions.intersection(
            row["extra_info"]["original_question"].strip() for row in val)
    assets = {str(p.relative_to(TASK)): {"sha256": sha(p), "bytes": p.stat().st_size}
              for p in [ACTIVE, ASSETS / "aime32.parquet", ASSETS / "math500_32.parquet"]}
    metadata = {"version": definition["version"], "rows": 1024, "unique_train_questions": 16,
                "repetitions": 64, "cases_sha256": sha(HERE / "v3_cases.json"),
                "archived_v2_sha256": sha(source), "assets": assets,
                "changes": {str(c["slot"]): c["action"] for c in cases if c["action"] != "unchanged"},
                "historical_metrics_apply_to": "proxy-data-v2",
                "paired_gpu_qualification": "pending for v3; historical v2 solutions were run"}
    if write:
        MANIFEST.write_text(json.dumps(metadata, indent=2) + "\n")
    else:
        assert json.loads(MANIFEST.read_text()) == metadata
    print("PASS: 16 answers, 1024 rows, 64 repetitions each, held-out separation, asset hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    build(parser.parse_args().write)
