"""The issue-level judge: scores from result.json, raw response preserved for audit."""

import json
import runpy
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
# All per-task judges are byte-identical; this one is the copy the package-task skill
# points contributors at, so it is the copy the behavior is defined on.
CANONICAL_JUDGE = (
    REPO_ROOT
    / "tasks"
    / "reward"
    / "slime_retool_length_reward_bonus"
    / "evaluations"
    / "issue"
    / "verifier"
    / "judge.py"
)

JUDGMENT = {
    "verdict_score": 1.0,
    "root_cause_score": 0.9,
    "evidence_score": 0.9,
    "reason": "the agent found the missing uid assignment",
}


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args) -> bool:
        return False


class IssueJudgeTests(unittest.TestCase):
    def _stage(self, root: Path) -> dict[str, str]:
        verifier = root / "verifier"
        verifier.mkdir()
        (verifier / "judge_prompt.md").write_text("grade the diagnosis\n", encoding="utf-8")
        (verifier / "oracle.md").write_text("uid must be assigned before repeat\n", encoding="utf-8")
        results = root / "results"
        results.mkdir()
        (results / "agent_answer.md").write_text("the uid assignment is too late\n", encoding="utf-8")
        return {
            "AUTORLBENCH_VERIFIER": str(verifier),
            "AUTORLBENCH_RESULTS": str(results),
            "AUTORLBENCH_ORACLE": str(verifier / "oracle.md"),
            "AUTORLBENCH_JUDGE_MODEL": "gpt-5.5",
            "OPENAI_BASE_URL": "https://judge.example/v1",
            "OPENAI_API_KEY": "secret-key",
        }

    def _run_judge(self, urlopen) -> None:
        with patch.dict("os.environ", self.env):
            with patch("urllib.request.urlopen", urlopen):
                # parse_args reads sys.argv; under pytest that is the test runner's argv.
                with patch("sys.argv", ["judge.py"]):
                    runpy.run_path(str(CANONICAL_JUDGE), run_name="__main__")

    def setUp(self) -> None:
        self._fixture = tempfile.TemporaryDirectory()
        self.addCleanup(self._fixture.cleanup)
        self.root = Path(self._fixture.name)
        self.env = self._stage(self.root)
        self.results = Path(self.env["AUTORLBENCH_RESULTS"])

    def test_scores_and_raw_response_are_both_written(self):
        content = json.dumps(JUDGMENT)
        body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
        captured = {}

        def urlopen(request, timeout=None):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(body)

        with self.assertRaises(SystemExit) as caught:
            self._run_judge(urlopen)
        self.assertEqual(caught.exception.code, 0)

        result = json.loads((self.results / "result.json").read_text(encoding="utf-8"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["metrics"]["root_cause_score"], 0.9)

        # The raw response and the exact request that produced it sit next to the scores,
        # so a surprising verdict can be audited without re-running the judge.
        raw = json.loads((self.results / "judge-raw.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["response"], content)
        self.assertEqual(raw["judgment"]["reason"], JUDGMENT["reason"])
        self.assertEqual(captured["url"], "https://judge.example/v1/chat/completions")
        self.assertIn("uid must be assigned before repeat", captured["payload"]["messages"][1]["content"])
        self.assertIn("the uid assignment is too late", captured["payload"]["messages"][1]["content"])

    def test_an_unreachable_judge_scores_zero_and_writes_no_raw_file(self):
        def urlopen(request, timeout=None):
            raise urllib.error.URLError("connection refused")

        with self.assertRaises(SystemExit) as caught:
            self._run_judge(urlopen)
        self.assertEqual(caught.exception.code, 2)

        result = json.loads((self.results / "result.json").read_text(encoding="utf-8"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["metrics"]["status"], "unavailable")
        self.assertFalse((self.results / "judge-raw.json").exists())

    def test_every_task_ships_the_same_judge(self):
        copies = sorted(REPO_ROOT.glob("tasks/*/*/evaluations/issue/verifier/judge.py"))
        self.assertGreater(len(copies), 1)
        canonical = CANONICAL_JUDGE.read_bytes()
        for copy in copies:
            self.assertEqual(
                copy.read_bytes(),
                canonical,
                f"{copy} drifted from the canonical judge; re-copy it from {CANONICAL_JUDGE}",
            )


if __name__ == "__main__":
    unittest.main()
