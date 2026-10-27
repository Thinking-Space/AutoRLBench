"""Prepared inputs retain original bytes; prepare pins the external snapshot."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'task.toml').is_file())
SPEC = importlib.util.spec_from_file_location('asset_storage', ROOT / 'environment/check_assets.py')
STORAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STORAGE)


class AssetStorageTests(unittest.TestCase):
    def fixture(self, root):
        original = b'{"metric": 0.3740234375, "task": "unchanged"}\n' * 4
        (root / 'sample.jsonl').write_bytes(original)
        return original, dict(path='sample.jsonl', sha256=hashlib.sha256(original).hexdigest(),
                              bytes=len(original))

    def test_verification_preserves_existing_bytes_and_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, entry = self.fixture(root)
            STORAGE.verify_asset(root, entry)
            STORAGE.verify_asset(root, entry)
            self.assertEqual((root / entry['path']).read_bytes(), original)
            self.assertEqual([p.name for p in root.iterdir()], ['sample.jsonl'])

    def test_corrupt_file_is_not_repaired_or_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, entry = self.fixture(root)
            target = root / entry['path']
            target.write_bytes(b'corrupt')
            with self.assertRaises(RuntimeError):
                STORAGE.verify_asset(root, entry)
            self.assertEqual(target.read_bytes(), b'corrupt')

    def test_size_hash_missing_file_and_archive_storage_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, entry = self.fixture(root)
            for patch in ({'bytes': len(original)-1}, {'sha256': '0'*64}, {'storage': {}}):
                with self.subTest(patch=patch), self.assertRaises(RuntimeError):
                    STORAGE.verify_asset(root, {**entry, **patch})
            with self.assertRaises(FileNotFoundError):
                STORAGE.verify_asset(root, {**entry, 'path': 'missing.jsonl'})
            self.assertEqual([p.name for p in root.iterdir()], ['sample.jsonl'])

    def test_asset_paths_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, entry = self.fixture(root)
            (root / 'link').symlink_to(root.parent, target_is_directory=True)
            for unsafe in ('../outside', '/tmp/outside', 'link/outside'):
                with self.subTest(path=unsafe), self.assertRaises(RuntimeError):
                    STORAGE.verify_asset(root, {**entry, 'path': unsafe})

    def test_all_manifest_inputs_exist_unpacked_and_hash_match(self):
        manifest = json.loads((ROOT / 'environment/data_manifest.json').read_text())
        self.assertNotIn('workspace_archives', manifest)
        entries = manifest['files'] + manifest['workspace_files']
        self.assertEqual(len(entries), 25)
        for entry in entries:
            self.assertNotIn('storage', entry)
            self.assertTrue((ROOT / 'workspace' / entry['path']).is_file())
            STORAGE.verify_asset(ROOT / 'workspace', entry)
        self.assertFalse([p for p in ROOT.rglob('*') if p.is_file()
                          and p.suffix in {'.xz', '.gz', '.bz2', '.zip', '.tar', '.tgz', '.zst'}])

    def test_prepare_uses_pinned_download_and_then_read_only_validation(self):
        # The fixture records the framework-helper invocation; no remote access.
        with tempfile.TemporaryDirectory() as temporary:
            capture = Path(temporary) / 'fetch-args.txt'
            command = ('autorlbench_fetch_url() { printf "%s\\n" "$@" >> "$FETCH_ARGS"; }; '
                       'export -f autorlbench_fetch_url; bash "$PREPARE_SCRIPT"')
            result = subprocess.run(['bash', '-c', command],
                                    env={**os.environ, 'AUTORLBENCH_ENVIRONMENT': str(ROOT/'environment'),
                                         'AUTORLBENCH_WORKSPACE': str(ROOT/'workspace'),
                                         'FETCH_ARGS': str(capture),
                                         'PREPARE_SCRIPT': str(ROOT/'environment/prepare.sh')},
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = capture.read_text().splitlines()
        manifest = json.loads((ROOT / 'environment/data_manifest.json').read_text())
        download = manifest['download']
        expected = []
        self.assertEqual(len(download['revision']), 40)
        for entry in manifest['files']:
            relative = str(Path(entry['path']).relative_to(download['destination']))
            url = f"https://huggingface.co/datasets/{download['repo']}/resolve/{download['revision']}/{relative}"
            expected += ['--type', 'dataset', '--url', url, '--sha256', entry['sha256'],
                         '--size', str(entry['bytes']), '--dest', str(ROOT/'workspace'/entry['path'])]
        self.assertEqual(args, expected)
        summary = json.loads(result.stdout)
        self.assertTrue(summary['read_only'])
        self.assertFalse(summary['decompression_required'])
        self.assertFalse(summary['network_required'])
        self.assertEqual(summary['workspace_files_verified'], 17)
        self.assertEqual(summary['dataset_assets_verified'], 8)

    def test_only_inactive_environment_implementations_are_omitted(self):
        package = ROOT / 'workspace/idea_source/agent_system/environments/env_package'
        self.assertEqual({p.name for p in package.iterdir() if p.is_dir()}, {'awm'})
        source = ROOT / 'workspace/idea_source/agent_system/environments/env_manager.py'
        tree = ast.parse(source.read_text())
        make_envs = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_envs')
        first_branch = next(n for n in make_envs.body if isinstance(n, ast.If)
                            and 'agent_world_model' in ast.unparse(n.test))
        self.assertIsInstance(first_branch.body[-1], ast.Return)
        self.assertIn('env_package.awm', ast.unparse(first_branch))


if __name__ == '__main__':
    unittest.main()
