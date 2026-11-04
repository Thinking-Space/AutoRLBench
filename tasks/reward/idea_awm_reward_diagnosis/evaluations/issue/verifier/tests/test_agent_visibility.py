import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "task.toml").is_file())
RESOURCES = ROOT / "evaluations/issue/verifier/resources"
SPEC = importlib.util.spec_from_file_location("agent_visibility", RESOURCES / "tools/check_agent_visibility.py")
VISIBILITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VISIBILITY)


def hidden_references():
    manifest = json.loads((RESOURCES / "curated_trajectory_manifest.json").read_text())
    ids = [case["traj_uid"] for case in manifest["cases"]]
    oracle = (ROOT / "evaluations/issue/oracle.md").read_text()
    fragments = [paragraph for paragraph in oracle.split("\n\n")
                 if len(paragraph.split()) >= 20 and not paragraph.startswith("#")]
    return ids, fragments


class AgentVisibilityTests(unittest.TestCase):
    def test_visible_manifest_contains_only_neutral_records(self):
        manifest = json.loads((ROOT / "environment/source_manifest.json").read_text())
        self.assertEqual(set(manifest), {"schema_version", "assets", "path_changes", "sanitization"})
        self.assertEqual(set(manifest["assets"]), {"sources", "model", "dataset"})
        for entry in manifest["assets"]["sources"]:
            self.assertLessEqual(set(entry), {"path", "project", "repository", "captured_revision"})
        self.assertEqual(set(manifest["assets"]["model"]), {"repo", "revision", "bundled"})
        self.assertEqual(set(manifest["assets"]["dataset"]),
                         {"repo", "revision", "manifest", "download_repo", "download_revision", "bundled"})
        self.assertEqual(set(manifest["sanitization"]), {"scope", "records"})
        for entry in manifest["sanitization"]["records"]:
            self.assertLessEqual(set(entry), {"path_at_capture", "path", "sha256_before", "sha256_after"})
        for entry in manifest["path_changes"]:
            self.assertEqual(set(entry), {"from", "to"})


    def test_all_agent_visible_mounts_are_free_of_reviewer_annotations(self):
        ids, fragments = hidden_references()
        self.assertEqual(len(ids), 8)
        findings = VISIBILITY.scan_agent_visible(
            ROOT / "workspace", ROOT / "environment", ROOT / "evaluations/issue/instruction.md",
            case_ids=ids, answer_fragments=fragments,
        )
        self.assertEqual(findings, [], "\n".join(findings))

    def test_guard_rejects_annotations_but_accepts_unlabelled_trajectory_ids(self):
        uid = hidden_references()[0][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, environment = root / "workspace", root / "environment"
            workspace.mkdir()
            environment.mkdir()
            instruction = root / "instruction.md"
            instruction.write_text("Inspect the historical experiment.")
            raw = workspace / "logs/rollouts/200.jsonl"
            raw.parent.mkdir(parents=True)
            raw.write_text(json.dumps({"traj_uid": uid, "score": 1.0}) + "\n")
            raw_digest = hashlib.sha256(raw.read_bytes()).hexdigest()
            def scan():
                return VISIBILITY.scan_agent_visible(workspace, environment, instruction,
                                                     case_ids=[uid], answer_fragments=["A canonical diagnosis copied here."])
            self.assertEqual(scan(), [])
            leak = environment / "source_manifest.json"
            for text in (json.dumps({"counterexample_ids": [uid]}),
                         json.dumps({"note": uid}),
                         "There are 28 counterexamples in the historical run.",
                         "A canonical diagnosis copied here.",
                         "reviewer_notes/curated_reward_examples/example.md"):
                with self.subTest(payload=text):
                    leak.write_text(text)
                    self.assertTrue(scan())
            leak.write_text("x" * (1024 * 1024 - 10) + "A canonical diagnosis copied here.")
            self.assertTrue(scan(), "Cross-chunk answer excerpts must not evade the guard")
            leak.unlink()
            labelled = raw.parent / "counterexample_ids.json"
            labelled.write_text(json.dumps([uid]))
            self.assertTrue(scan())
            labelled.unlink()
            (environment / "external").symlink_to(RESOURCES, target_is_directory=True)
            self.assertTrue(scan())
            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), raw_digest)

    def test_unpacked_package_rejects_archive_inputs(self):
        uid = hidden_references()[0][0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace, environment = root / 'workspace', root / 'environment'
            workspace.mkdir()
            environment.mkdir()
            instruction = root / 'instruction.md'
            instruction.write_text('Inspect the experiment.')
            raw = workspace / 'logs/rollouts/200.jsonl'
            raw.parent.mkdir(parents=True)
            raw.write_text(json.dumps({'traj_uid': uid, 'score': 1.0}) + '\n')
            def scan():
                return VISIBILITY.scan_agent_visible(workspace, environment, instruction, case_ids=[uid])
            self.assertEqual(scan(), [])
            for suffix in ('.xz', '.gz', '.zip'):
                archive = raw.with_name(raw.name + suffix)
                archive.write_bytes(b'unexpected archive placeholder')
                self.assertTrue(scan())
                archive.unlink()
            self.assertEqual(scan(), [])


if __name__ == "__main__":
    unittest.main()
