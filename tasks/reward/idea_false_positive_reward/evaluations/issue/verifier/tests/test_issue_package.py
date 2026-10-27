import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "task.toml").is_file())
SPEC = importlib.util.spec_from_file_location("issue_judge", ROOT / "evaluations/issue/verifier/judge.py")
JUDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)


class IssuePackageTests(unittest.TestCase):
    def test_standard_issue_only_layout_and_resource_boundaries(self):
        self.assertEqual({p.name for p in (ROOT / "environment").iterdir()}, {
            "prepare.sh", "check_assets.py", "data_manifest.json", "source_manifest.json", "runtime",
        })
        self.assertEqual({p.name for p in (ROOT / "evaluations/issue").iterdir()}, {
            "instruction.md", "oracle.md", "verifier",
        })
        self.assertFalse((ROOT / "evaluations/solution").exists())
        self.assertFalse((ROOT / "validation").exists())
        self.assertFalse(list(ROOT.rglob("*.wandb")))
        resources = ROOT / "evaluations/issue/verifier/resources"
        self.assertTrue((resources / "README.md").is_file())
        self.assertTrue((resources / "reviewer_notes/curated_reward_examples/README.md").is_file())
        self.assertTrue((resources / "tools/curate_reward_trajectories.py").is_file())
        self.assertTrue((ROOT / "evaluations/issue/verifier/tests").is_dir())


    def test_standard_files_match_reference(self):
        resources = ROOT / "evaluations/issue/verifier/resources"
        expected = json.loads((resources / "scaffolding_manifest.json").read_text())
        for name in ["instruction.md", "verifier/test.sh", "verifier/judge.py", "verifier/judge_prompt.md"]:
            content = (ROOT / "evaluations/issue" / name).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), expected["sha256"][name])

    def test_grading_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Path(tmp) / "result.json"
            args = SimpleNamespace(answer=Path(tmp) / "answer.md", oracle=ROOT / "evaluations/issue/oracle.md",
                                   prompt=ROOT / "evaluations/issue/verifier/judge_prompt.md", result=result)
            args.answer.write_text("Offline grading fixture; no external inference.")
            for score, exit_code in [(1.0, 0), (0.0, 1)]:
                judgment = {key: score for key in JUDGE.WEIGHTS}
                judgment["reason"] = "Offline fixture"
                with patch.object(JUDGE, "parse_args", return_value=args), patch.object(JUDGE, "call_judge", return_value=judgment), patch.dict("os.environ", {
                    "AUTORLBENCH_JUDGE_MODEL": "offline-fixture", "OPENAI_BASE_URL": "https://judge.example.invalid/v1", "OPENAI_API_KEY": "EMPTY",
                }):
                    self.assertEqual(JUDGE.main(), exit_code)
                self.assertEqual(json.loads(result.read_text())["passed"], exit_code == 0)

    def test_missing_credentials_do_not_request_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            answer = path / "answer.md"
            answer.write_text("unhealthy")
            args = SimpleNamespace(answer=answer, oracle=ROOT / "evaluations/issue/oracle.md",
                                   prompt=ROOT / "evaluations/issue/verifier/judge_prompt.md", result=path / "result.json")
            with patch.object(JUDGE, "parse_args", return_value=args), patch.object(JUDGE, "call_judge") as call, patch.dict("os.environ", {
                "AUTORLBENCH_JUDGE_MODEL": "", "OPENAI_BASE_URL": "", "OPENAI_API_KEY": "",
            }):
                self.assertEqual(JUDGE.main(), 2)
                call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
