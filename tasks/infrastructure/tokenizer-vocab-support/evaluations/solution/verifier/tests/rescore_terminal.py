"""Recompute #71's terminal metric from the native held-out responses."""

from decimal import Decimal
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

WHEELS = {
    "math_verify-0.9.0-py3-none-any.whl": "3703e7c4885354027fa84409d762a596a2906d1fd4deb78361876bd905a76194",
    "latex2sympy2_extended-1.11.0-py3-none-any.whl": "aebb77d52ce269e25028e4bea89ddb14d242ba36bcf7b636496fb5fd9728d234",
}


def main():
    answers, data, step, prepared = sys.argv[1:]
    with TemporaryDirectory() as dependencies:
        for name, expected_digest in WHEELS.items():
            payload = (Path(prepared) / "wheels" / name).read_bytes()
            if sha256(payload).hexdigest() != expected_digest:
                raise ValueError(f"Unexpected verifier dependency digest: {name}")
            with ZipFile(BytesIO(payload)) as wheel:
                wheel.extractall(dependencies)
        sys.path[:0] = [str(Path(__file__).resolve().parent), dependencies]
        from math_verify.grader import verify  # Check dependency imports before scoring.
        from native_math_score import _get_pool, compute_score
        from summarize import DRAWS, heldout, inputs

        if not callable(verify):
            raise ValueError("Math-Verify grader did not load")
        expected = inputs(Path(data))
        with _get_pool():
            counts, _ = heldout(
                Path(answers), expected, int(step),
                native_score=lambda source, response, gold: compute_score(response, gold),
            )
    print(Decimal(sum(counts.values())) / Decimal(len(expected) * DRAWS))


if __name__ == "__main__":
    main()
