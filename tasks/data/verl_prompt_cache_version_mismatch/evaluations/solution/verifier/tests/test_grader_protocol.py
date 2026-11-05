"""CPU evidence replay and protocol mutations, never a GPU-training claim.

Set CACHE_EVIDENCE_DIR to the reviewer evidence root with candidate/reference
subdirectories containing metrics.jsonl and validation-60.jsonl. Fresh-contract
unit fixtures explicitly mock only checkpoint verification; the actual fresh
entry point does not provide that bypass.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stderr
import io

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
loader = importlib.util.spec_from_file_location("cache60_grader", HERE / "grade_submission.py")
grader = importlib.util.module_from_spec(loader)
loader.loader.exec_module(grader)


def find_evidence():
    if os.environ.get("CACHE_EVIDENCE_DIR"):
        return Path(os.environ["CACHE_EVIDENCE_DIR"])
    for ancestor in HERE.parents:
        candidate = ancestor / "outputs/cache-version-60step-pilot"
        if (candidate / "reference/validation-60.jsonl").is_file():
            return candidate
    return None


def dump_rows(path, rows):
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def assign(config, name, value):
    parts = name.split(".")
    for part in parts[:-1]:
        config = config.setdefault(part, {})
    config[parts[-1]] = value


class GraderProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = find_evidence()
        if cls.evidence is None:
            raise unittest.SkipTest("Set CACHE_EVIDENCE_DIR to actual candidate/reference evidence")
        cls.spec = json.loads((HERE.parent / "resources/grader_spec.json").read_text())
        samples = grader.load_rows(cls.evidence / "reference/validation-60.jsonl")
        identities = {}
        for sample in samples:
            identities.setdefault(sample["problem_id"], {
                "problem_id": sample["problem_id"], "input": sample["input"],
                "ground_truth": str(sample["gts"]), "source": sample["data_source"],
            })
        cls.items = list(identities.values())

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.results = self.root / "results"
        self.logs = self.results / "experiment_logs"
        (self.logs / "fresh_eval").mkdir(parents=True)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.resources = self.root / "resources"
        self.resources.mkdir()
        shutil.copyfile(self.evidence / "reference/metrics.jsonl", self.logs / "metrics.jsonl")
        shutil.copyfile(self.evidence / "reference/validation-60.jsonl", self.logs / "fresh_eval/60.jsonl")

    def tearDown(self):
        self.temp.cleanup()

    def replay(self, spec=None, items=None):
        return grader.evaluate(self.results, spec or self.spec, self.items if items is None else items, replay=True)

    def fresh_setup(self):
        config = {}
        for name, value in grader.MANDATORY_CONTRACT.items():
            assign(config, name, copy.deepcopy(value))
        data = self.workspace / "data"
        model = self.workspace / "model"
        data.mkdir()
        model.mkdir()
        (self.workspace / "case").mkdir()
        (self.workspace / "case/dataset_cache.py").write_text("# fixture\n")
        protected_code = {}
        for name in ("case/dapo_reward.py", "verl/utils/reward_score/math_dapo.py", "verl/trainer/ppo/core_algos.py"):
            path = self.workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = ("# protected fixture: " + name + "\n").encode()
            path.write_bytes(payload)
            protected_code[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        (self.resources / "protected_code.json").write_text(json.dumps(protected_code))
        (self.workspace / "report.md").write_text("Repair report for this test fixture.\n")
        registry = {"derived": {}, "model": {"files": []}}
        for name in ("train.parquet", "val.parquet", "prompt-cache-v1.parquet", "prompt-cache-v1.json"):
            payload = name.encode()
            (data / name).write_bytes(payload)
            registry["derived"][name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        (data / "manifest.json").write_text("{}\n")
        payload = (data / "manifest.json").read_bytes()
        registry["prepared_manifest"] = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
        payload = b'{"model_type":"qwen3"}\n'
        (model / "config.json").write_bytes(payload)
        registry["model"]["files"].append({"path": "config.json", "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)})
        paths = {"data.train_files": data / "train.parquet", "data.val_files": data / "val.parquet",
                 "data.cache_seed_path": data / "prompt-cache-v1.parquet", "data.cache_manifest_path": data / "manifest.json",
                 "data.cache_dir": self.logs / "processed-prompt-cache",
                 "data.custom_cls.path": self.workspace / "case/dataset_cache.py", "actor_rollout_ref.model.path": model}
        for name, value in paths.items():
            assign(config, name, str(value))
        (self.logs / "effective_config.json").write_text(json.dumps(config))
        (self.resources / "contract.json").write_text(json.dumps(grader.MANDATORY_CONTRACT))
        (self.resources / "asset_registry.json").write_text(json.dumps(registry))
        return config

    def fresh(self, mock_checkpoint=True):
        def call():
            return grader.evaluate(self.results, self.spec, self.items, workspace=self.workspace, resources=self.resources)
        if mock_checkpoint:
            with mock.patch.object(grader, "verify_checkpoint", return_value={"unit_test_checkpoint_mock": True}):
                return call()
        return call()

    def test_actual_recorded_candidate_fails_reference_passes(self):
        good = self.replay()
        self.assertTrue(good["passed"], good)
        self.assertAlmostEqual(good["value"], .13020833333333333)
        shutil.copyfile(self.evidence / "candidate/metrics.jsonl", self.logs / "metrics.jsonl")
        shutil.copyfile(self.evidence / "candidate/validation-60.jsonl", self.logs / "fresh_eval/60.jsonl")
        bad = self.replay()
        self.assertFalse(bad["passed"], bad)
        self.assertEqual(bad["errors"], ["Metric outside accepted interval"])
        self.assertAlmostEqual(bad["value"], .04114583333333333)

    def test_missing_final_step_rejected(self):
        rows = grader.load_rows(self.logs / "metrics.jsonl")
        dump_rows(self.logs / "metrics.jsonl", rows[:-1])
        self.assertFalse(self.replay()["passed"])

    def test_duplicate_metric_step_rejected(self):
        rows = grader.load_rows(self.logs / "metrics.jsonl")
        dump_rows(self.logs / "metrics.jsonl", rows + [rows[-1]])
        self.assertFalse(self.replay()["passed"])

    def test_missing_scheduled_validation_rejected(self):
        rows = grader.load_rows(self.logs / "metrics.jsonl")
        rows[50]["data"] = {key: value for key, value in rows[50]["data"].items() if not key.startswith(("val-core/", "val-aux/"))}
        dump_rows(self.logs / "metrics.jsonl", rows)
        self.assertFalse(self.replay()["passed"])

    def test_nonfinite_gradient_rejected(self):
        rows = grader.load_rows(self.logs / "metrics.jsonl")
        rows[30]["data"]["actor/grad_norm"] = float("nan")
        dump_rows(self.logs / "metrics.jsonl", rows)
        self.assertFalse(self.replay()["passed"])

    def test_aborted_answers_rejected(self):
        rows = grader.load_rows(self.logs / "metrics.jsonl")
        rows[30]["data"]["response/aborted_ratio"] = .1
        dump_rows(self.logs / "metrics.jsonl", rows)
        self.assertFalse(self.replay()["passed"])

    def test_missing_final_answers_rejected(self):
        (self.logs / "fresh_eval/60.jsonl").unlink()
        self.assertFalse(self.replay()["passed"])

    def test_duplicate_question_replacing_another_rejected(self):
        rows = grader.load_rows(self.logs / "fresh_eval/60.jsonl")
        different = next(row for row in rows if row["problem_id"] != rows[0]["problem_id"])
        rows[0] = copy.deepcopy(different)
        dump_rows(self.logs / "fresh_eval/60.jsonl", rows)
        self.assertFalse(self.replay()["passed"])

    def test_tampered_per_answer_score_rejected(self):
        rows = grader.load_rows(self.logs / "fresh_eval/60.jsonl")
        rows[0]["score"] = 1 - rows[0]["score"]
        dump_rows(self.logs / "fresh_eval/60.jsonl", rows)
        result = self.replay()
        self.assertFalse(result["passed"])
        self.assertIn("Per-response", result["errors"][0])

    def test_tampered_logged_accuracy_rejected(self):
        rows = grader.load_rows(self.logs / "metrics.jsonl")
        for key in tuple(rows[-1]["data"]):
            if "aime24" in key and key.endswith("/mean@32"):
                rows[-1]["data"][key] = .99
        dump_rows(self.logs / "metrics.jsonl", rows)
        result = self.replay()
        self.assertFalse(result["passed"])
        self.assertIn("independent text rescoring", result["errors"][0])

    def test_duplicate_trusted_eval_ids_rejected(self):
        items = copy.deepcopy(self.items)
        items[-1] = items[0]
        self.assertFalse(self.replay(items=items)["passed"])

    def test_trusted_items_data_source_alias_is_supported(self):
        items = copy.deepcopy(self.items)
        for item in items:
            item["data_source"] = item.pop("source")
        result = self.replay(items=items)
        self.assertTrue(result["passed"], result)

    def test_frozen_evidence_path_is_supported_only_by_reviewer_replay(self):
        (self.logs / "frozen_eval").mkdir()
        (self.logs / "fresh_eval/60.jsonl").rename(self.logs / "frozen_eval/60.jsonl")
        result = self.replay()
        self.assertTrue(result["passed"], result)

    def test_fresh_contract_passes_with_explicit_unit_checkpoint_mock(self):
        self.fresh_setup()
        result = self.fresh()
        self.assertTrue(result["passed"], result)
        self.assertTrue(result["verification"]["checkpoint"]["unit_test_checkpoint_mock"])

    def test_missing_report_rejected_in_fresh_protocol(self):
        self.fresh_setup()
        (self.workspace / "report.md").unlink()
        result = self.fresh()
        self.assertFalse(result["passed"])
        self.assertIn("report.md", result["errors"][0])

    def test_wrong_sampling_rejected_in_fresh_protocol(self):
        config = self.fresh_setup()
        config["actor_rollout_ref"]["rollout"]["val_kwargs"]["n"] = 16
        (self.logs / "effective_config.json").write_text(json.dumps(config))
        result = self.fresh()
        self.assertFalse(result["passed"])
        self.assertIn("val_kwargs.n", result["errors"][0])

    def test_wrong_budget_rejected_in_fresh_protocol(self):
        config = self.fresh_setup()
        config["trainer"]["total_training_steps"] = 100
        (self.logs / "effective_config.json").write_text(json.dumps(config))
        self.assertFalse(self.fresh()["passed"])

    def test_modified_cache_seed_rejected(self):
        self.fresh_setup()
        (self.workspace / "data/prompt-cache-v1.parquet").write_bytes(b"modified")
        result = self.fresh()
        self.assertFalse(result["passed"])
        self.assertIn("Immutable asset", result["errors"][0])

    def test_modified_prepared_manifest_rejected(self):
        self.fresh_setup()
        (self.workspace / "data/manifest.json").write_text('{"version":2}\n')
        result = self.fresh()
        self.assertFalse(result["passed"])
        self.assertIn("Immutable asset modified or missing: manifest.json", result["errors"][0])

    def test_modified_reward_source_rejected(self):
        self.fresh_setup()
        (self.workspace / "case/dapo_reward.py").write_text("# changed reward shaping\n")
        result = self.fresh()
        self.assertFalse(result["passed"])
        self.assertIn("Protected training code modified or missing: case/dapo_reward.py", result["errors"][0])

    def test_missing_required_protected_algorithm_source_rejected(self):
        self.fresh_setup()
        path = self.resources / "protected_code.json"
        protected = json.loads(path.read_text())
        protected.pop("verl/trainer/ppo/core_algos.py")
        path.write_text(json.dumps(protected))
        result = self.fresh()
        self.assertFalse(result["passed"])
        self.assertIn("cover fixed reward and algorithm sources", result["errors"][0])

    def test_changed_base_model_rejected(self):
        self.fresh_setup()
        (self.workspace / "model/config.json").write_text("{}")
        self.assertFalse(self.fresh()["passed"])

    def test_missing_real_step60_checkpoint_is_rejected(self):
        self.fresh_setup()
        result = self.fresh(mock_checkpoint=False)
        self.assertFalse(result["passed"])
        self.assertIn("global_step_60", result["errors"][0])

    def test_historical_replay_has_no_fresh_gpu_claim(self):
        result = self.replay()
        self.assertTrue(result["passed"])
        self.assertFalse(result["fresh_gpu_run_verified"])

    def test_reviewer_cli_is_blocked_inside_harness(self):
        with mock.patch.dict(os.environ, {"AUTORLBENCH_REVIEWER_REPLAY": "1", "AUTORLBENCH_RESULTS": str(self.results)}), \
                mock.patch.object(sys, "argv", ["grader", "--historical-replay", str(self.evidence / "reference")]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            grader.main()
        self.assertEqual(caught.exception.code, 2)

    def test_reviewer_cli_requires_explicit_reviewer_opt_in(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(sys, "argv", ["grader", "--historical-replay", str(self.evidence / "reference")]), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            grader.main()
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
