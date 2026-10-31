# SPDX-License-Identifier: Apache-2.0
"""Standard-library tests using tiny synthetic bytes; never load real models."""
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def record(path):
    raw = path.read_bytes()
    return {'size': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


class PublicRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='synthetic-public-runtime-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def module(self, name):
        self.assertTrue((HERE / (name + '.py')).is_file(), name + ' entry point is missing')
        return importlib.import_module(name)

    def test_asset_bytes_must_match_size_and_hash_even_when_cached(self):
        prepare = self.module('prepare_assets')
        path = self.root / 'synthetic.bin'
        path.write_bytes(b'explicitly synthetic weights')
        expected = record(path)
        prepare.verify_file(path, expected)
        path.write_bytes(b'X' * expected['size'])
        with self.assertRaisesRegex(ValueError, 'hash|SHA'):
            prepare.verify_file(path, expected)
        path.write_bytes(b'X')
        with self.assertRaisesRegex(ValueError, 'size'):
            prepare.verify_file(path, expected)

    def test_inventory_rejects_unexpected_files_and_symlinks(self):
        prepare = self.module('prepare_assets')
        folder = self.root / 'model'
        folder.mkdir()
        path = folder / 'synthetic.bin'
        path.write_bytes(b'synthetic')
        expected = {path.name: record(path)}
        prepare.verify_inventory(folder, expected)
        (folder / 'extra').write_text('not pinned')
        with self.assertRaises(ValueError):
            prepare.verify_inventory(folder, expected)
        (folder / 'extra').unlink()
        path.unlink()
        path.symlink_to(self.root / 'outside')
        with self.assertRaises(ValueError):
            prepare.verify_inventory(folder, expected)

    def test_archive_is_checked_against_complete_source_inventory(self):
        prepare = self.module('prepare_assets')
        source = self.root / 'file.py'
        source.write_bytes(b'# synthetic source\n')
        archive = self.root / 'source.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(source, arcname='JustRL-II-synthetic/file.py')
        assets = {'justrl2_revision': 'synthetic', 'public_source_archive': record(archive)}
        pins = {'file.py': record(source)['sha256']}
        prepare.verify_source_archive(archive, assets, pins)
        with self.assertRaises(ValueError):
            prepare.verify_source_archive(archive, assets, {**pins, 'missing.py': '0' * 64})

    def test_public_cli_defaults_and_midpoint_validation(self):
        run = self.module('run_case')
        args = run.parse_args(['--prepared', str(self.root / 'prepared'), '--output', str(self.root / 'home'), '--arm', 'faulty'])
        self.assertEqual((args.steps, args.relocate_at, args.batch, args.samples, args.max_response), (8, 4, 8, 2, 32768))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.parse_args(['--prepared', '.', '--output', 'x', '--arm', 'healthy', '--steps', '6'])

    def test_conversion_has_one_rank_and_training_keeps_three_visible_gpus(self):
        run = self.module('run_case')
        original = {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b,GPU-c', 'WORLD_SIZE': '99', 'RANK': '7',
                    'RELOCATION_CODE': '/stale', 'MODEL_ARGS_ROTARY_BASE': '12', 'PYTHONPATH': '/stale'}
        env = run.conversion_env(self.root / 'code', self.root / 'hf', self.root / 'native', original, 29400)
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-a')
        self.assertEqual([env[k] for k in ('WORLD_SIZE', 'RANK', 'LOCAL_RANK')], ['1', '0', '0'])
        self.assertEqual(env['MASTER_ADDR'], '127.0.0.1')
        self.assertNotIn('MODEL_ARGS_ROTARY_BASE', env)
        self.assertNotIn('RELOCATION_CODE', env)
        self.assertEqual(original['CUDA_VISIBLE_DEVICES'], 'GPU-a,GPU-b,GPU-c')
        self.assertIn('source justrl2/model_args/minicpm5-2b.sh', run.CONVERSION_SCRIPT)
        self.assertIn('tools/convert_hf_to_torch_dist.py "${MODEL_ARGS[@]}"', run.CONVERSION_SCRIPT)
        self.assertNotIn('--rollout', run.CONVERSION_SCRIPT)

    def test_release_inventory_is_rehashed_and_tracker_is_required(self):
        run = self.module('run_case')
        native = self.root / 'native'
        (native / 'release').mkdir(parents=True)
        (native / 'latest_checkpointed_iteration.txt').write_text('release')
        files = {}
        for name in ('.metadata', '__0_0.distcp'):
            path = native / 'release' / name
            path.write_bytes(b'explicitly synthetic checkpoint')
            files[name] = record(path)
        receipt = {'iteration': 'release', 'selected': 'release', 'referenced_files': files}
        self.assertEqual(run.verify_native_inputs(native, receipt), files)
        (native / 'release/__0_0.distcp').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            run.verify_native_inputs(native, receipt)
        (native / 'latest_checkpointed_iteration.txt').write_text('7')
        with self.assertRaises(ValueError):
            run.verify_native_inputs(native, receipt)

    def test_subprocess_log_preserves_actual_bytes_and_exit_code(self):
        run = self.module('run_case')
        log = self.root / 'stdout.log'
        with redirect_stdout(io.StringIO()):
            code = run.run_logged([sys.executable, '-c', 'import os; os.write(1,b"synthetic\\xff\\n"); os.write(2,b"stderr\\n"); raise SystemExit(7)'],
                                  cwd=self.root, env=os.environ.copy(), log=log)
        self.assertEqual(code, 7)
        self.assertEqual(log.read_bytes(), b'synthetic\xff\nstderr\n')

    def test_existing_home_rejected_before_staging_or_cuda(self):
        run = self.module('run_case')
        home = self.root / 'home'
        home.mkdir()
        args = run.parse_args(['--prepared', str(self.root / 'missing'), '--output', str(home), '--arm', 'healthy'])
        with self.assertRaises(FileExistsError):
            run.run_case(args)

    def public_fixture(self):
        # Reuse Hume's four tiny synthetic shards, never the real corpus/model.
        fixtures = importlib.import_module('public_data.test_prepare_public')
        fixture = fixtures.PublicPreparationTests()
        fixture.setUp()
        self.addCleanup(fixture.temporary.cleanup)
        for item in fixture.lock['files']:
            item['git_blob_sha1'] = 'b' * 40
        fixture.save_inputs()
        public = self.root / 'public_data'
        public.mkdir()
        for source, name in ((fixture.lock_path, 'sources.lock.json'), (fixture.selection_path, 'selection.json'),
                             (HERE / 'public_data/prepare_public.py', 'prepare_public.py')):
            shutil.copyfile(source, public / name)
        return fixture

    def test_public_preparer_interface_and_canonical_source_hash(self):
        prepare = self.module('prepare_assets')
        fixture = self.public_fixture()
        fixture.prepare()
        with patch.object(prepare, 'HERE', self.root):
            expected = prepare.verify_dataset(fixture.out)
            self.assertEqual(len(expected), 6)
            # Even a self-consistent edited receipt must not invent a composite SHA.
            case = json.loads((fixture.out / 'CASE.json').read_text())
            case['source']['sha256'] = '0' * 64
            prepare.write_json(fixture.out / 'CASE.json', case)
            receipt = json.loads((fixture.out / 'PUBLIC_PROVENANCE.json').read_text())
            receipt['source'] = case['source']
            receipt['case_sha256'] = record(fixture.out / 'CASE.json')['sha256']
            prepare.write_json(fixture.out / 'PUBLIC_PROVENANCE.json', receipt)
            with self.assertRaisesRegex(ValueError, 'composite'):
                prepare.verify_dataset(fixture.out)

    def test_cpu_prepare_pipeline_with_explicitly_synthetic_tensor_converter(self):
        prepare = self.module('prepare_assets')
        fixture = self.public_fixture()
        cache = self.root / 'cache'
        (cache / 'model').mkdir(parents=True)
        pins = []
        for name, raw in (('pytorch_model.bin', b'NOT REAL WEIGHTS: synthetic'),
                          ('config.json', b'{"synthetic": true}')):
            path = cache / 'model' / name
            path.write_bytes(raw)
            pins.append({'path': 'model/' + name, 'url': path.as_uri(), **record(path)})
        source = self.root / 'synthetic.py'
        source.write_bytes(b'# synthetic source\n')
        archive = cache / 'source.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(source, arcname='JustRL-II-synthetic/synthetic.py')
        assets = {'justrl2_revision': 'synthetic', 'model_revision': 'synthetic-model', 'assets': pins,
                  'public_source_archive': {'url': archive.as_uri(), **record(archive)}}
        prepare.write_json(self.root / 'assets.json', assets)
        (self.root / 'runtime_support').mkdir()
        prepare.write_json(self.root / 'runtime_support/source_files.json', {'files': {'synthetic.py': record(source)['sha256']}})
        from types import SimpleNamespace
        args = SimpleNamespace(out=self.root / 'prepared', asset_dir=cache, source_dir=fixture.sources, download=False)

        def synthetic_convert(source, target):
            self.assertEqual(source.read_bytes(), b'NOT REAL WEIGHTS: synthetic')
            target.write_bytes(b'NOT SAFETENSORS: synthetic test output')
            return {'source_sha256': record(source)['sha256'], 'target_sha256': record(target)['sha256'],
                    'target_size': target.stat().st_size, 'tensor_count': 1, 'all_tensors_exactly_equal': True}

        with patch.object(prepare, 'HERE', self.root), \
                patch('runtime_support.checkpoint_evidence.convert_bin_to_safetensors', side_effect=synthetic_convert) as convert, \
                redirect_stdout(io.StringIO()):
            prepare.prepare(args)
            convert.assert_called_once()
            prepare.verify_prepared(args.out)
            (args.out / 'hf-safetensors/config.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                prepare.verify_prepared(args.out)

    def test_fetch_validates_before_publishing_and_rechecks_cache(self):
        prepare = self.module('prepare_assets')
        source = self.root / 'synthetic-source'
        source.write_bytes(b'explicitly synthetic asset')
        pin = {**record(source), 'url': source.as_uri()}
        target = self.root / 'downloaded'
        prepare.fetch_asset(target, pin)
        self.assertEqual(target.read_bytes(), source.read_bytes())
        target.write_bytes(b'edited')
        with self.assertRaises(ValueError):
            prepare.fetch_asset(target, pin)
        target.unlink()
        with self.assertRaises(ValueError):
            prepare.fetch_asset(target, {**pin, 'sha256': '0' * 64})
        self.assertFalse(target.exists())
        self.assertFalse(target.with_name('downloaded.partial').exists())

    def lifecycle(self, training_code=9, mutate_native=False):
        run = self.module('run_case')
        prepared = self.root / 'prepared'
        data = prepared / 'dataset'
        data.mkdir(parents=True)
        for name in ('CASE.json', 'manifest.json', 'eval.jsonl', 'PUBLIC_PROVENANCE.json'):
            (data / name).write_text('{"synthetic": true}\n')
        args = run.parse_args(['--prepared', str(prepared), '--output', str(self.root / 'home'), '--arm', 'faulty'])
        assets = {'model_revision': 'synthetic-model', 'justrl2_revision': 'synthetic-source'}
        converter = {'files': {'model.safetensors': {'size': 1, 'sha256': 'a' * 64}}}

        def stage(home, prepared, assets):
            code = home / 'source/synthetic'
            code.mkdir(parents=True)
            (home / 'source/source_receipt.json').write_text('{"synthetic": true}\n')
            return code, {'synthetic': True}

        def convert(home, *unused):
            native = home / 'native'
            (native / 'release').mkdir(parents=True)
            (native / 'latest_checkpointed_iteration.txt').write_text('release')
            files = {}
            for name in ('.metadata', '__0_0.distcp'):
                path = native / 'release' / name
                path.write_bytes(b'explicitly synthetic checkpoint')
                files[name] = record(path)
            receipt = {'native_checkpoint': {'iteration': 'release', 'selected': 'release', 'referenced_files': files}}
            run.write_json(native / 'conversion_receipt.json', receipt)
            return receipt

        def training(command, *, cwd, env, log):
            self.assertEqual(Path(command[1]), args.output.resolve() / 'plugin/train.sh')
            self.assertEqual(env['RELOCATION_MAX_RESPONSE'], '32768')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '0,1,2')
            (args.output / 'dataset/manifest.json').write_text('synthetic final manifest\n')
            if mutate_native:
                (args.output / 'native/release/__0_0.distcp').write_text('unexpected mutation')
            if training_code == 0:
                (args.output / 'run').mkdir()
                run.write_json(args.output / 'run/TRAINING_RETURNED.json', {'training_returned': True, 'arm': args.arm})
            return run.run_logged_original([sys.executable, '-c', 'print("synthetic trainer"); raise SystemExit(%d)' % training_code],
                                           cwd=cwd, env=env, log=log)

        def inspect_output(root, *, expected_iteration, require_optimizer):
            self.assertEqual(expected_iteration, 7)
            self.assertTrue(require_optimizer)
            return {'iteration': '7', 'selected': 'iter_0000007', 'synthetic_role': root.name}

        with patch.object(run, 'verify_prepared', return_value=(assets, converter)), \
                patch.object(run, 'check_training_gpus'), patch.object(run, 'stage_source', side_effect=stage), \
                patch.object(run, 'convert_native', side_effect=convert), \
                patch.object(run, 'verify_converter_input'), \
                patch.object(run, 'run_logged_original', run.run_logged, create=True), \
                patch.object(run, 'run_logged', side_effect=training), \
                patch('runtime_support.checkpoint_evidence.inspect_checkpoint', side_effect=inspect_output), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0,1,2'}), redirect_stdout(io.StringIO()):
            code = run.run_case(args)
        return code, args.output

    def test_lifecycle_keeps_failure_logs_final_manifest_and_input_hashes(self):
        run = self.module('run_case')
        code, home = self.lifecycle()
        self.assertEqual(code, 9)
        self.assertEqual((home / 'stdout.log').read_text(), 'synthetic trainer\n')
        self.assertEqual((home / 'provenance/final_manifest.json').read_text(), 'synthetic final manifest\n')
        self.assertNotEqual((home / 'provenance/initial_manifest.json').read_bytes(),
                            (home / 'provenance/final_manifest.json').read_bytes())
        self.assertEqual(json.loads((home / 'RESULT.json').read_text())['returncode'], 9)
        inputs = json.loads((home / 'INPUTS.json').read_text())
        unchanged = json.loads((home / 'provenance/original_native_unchanged.json').read_text())
        self.assertEqual(inputs['native_checkpoint_files'], unchanged)
        self.assertEqual(set(inputs['plugin_files']), set(run.PLUGINS))
        for name in run.PLUGINS:
            self.assertEqual((home / 'plugin' / name).read_bytes(), (HERE / 'runtime' / name).read_bytes())
        self.assertFalse((home / 'provenance/output_checkpoints.json').exists())

    def test_lifecycle_success_inspects_both_optimizer_checkpoints(self):
        code, home = self.lifecycle(training_code=0)
        self.assertEqual(code, 0)
        checkpoints = json.loads((home / 'provenance/output_checkpoints.json').read_text())
        self.assertEqual(set(checkpoints), {'actor', 'critic'})
        self.assertEqual(json.loads((home / 'RESULT.json').read_text())['stage'], 'complete')

    def test_lifecycle_detects_native_mutation_even_if_training_fails(self):
        with self.assertRaisesRegex(ValueError, 'changed during training'):
            self.lifecycle(mutate_native=True)
        home = self.root / 'home'
        self.assertFalse((home / 'provenance/original_native_unchanged.json').exists())
        self.assertEqual(json.loads((home / 'RESULT.json').read_text())['returncode'], 1)
        self.assertTrue((home / 'provenance/native_all_files_after.json').is_file())

    def test_native_conversion_receipt_and_reuse_bind_actual_bytes(self):
        run = self.module('run_case')
        model = self.root / 'hf-safetensors'
        model.mkdir()
        (model / 'model.safetensors').write_bytes(b'NOT SAFETENSORS: synthetic fixture')
        (model / 'config.json').write_bytes(b'{"synthetic":true}')
        source_bin = self.root / 'synthetic.bin'
        source_bin.write_bytes(b'NOT WEIGHTS: synthetic fixture')
        assets = {'justrl2_revision': 'synthetic', 'model_revision': 'synthetic-model', 'assets': [
            {'path': 'model/pytorch_model.bin', **record(source_bin)},
            {'path': 'model/config.json', **record(model / 'config.json')}]}
        converter = {'format': 'losslessly_reencoded_safetensors', 'files': run.inventory(model), 'tensor_check': {
            'source_sha256': record(source_bin)['sha256'], 'target_sha256': record(model / 'model.safetensors')['sha256'],
            'target_size': (model / 'model.safetensors').stat().st_size, 'tensor_count': 1, 'all_tensors_exactly_equal': True}}
        source_receipt = {'original_files': {'synthetic.py': 'a' * 64}}
        home = self.root / 'home'
        code = home / 'source'
        code.mkdir(parents=True)

        def synthetic_conversion(command, *, cwd, env, log):
            self.assertEqual(env['WORLD_SIZE'], '1')
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-a')
            native = Path(env['NATIVE_OUTPUT'])
            (native / 'release').mkdir()
            for name in ('.metadata', '__0_0.distcp'):
                (native / 'release' / name).write_bytes(b'explicitly synthetic native output')
            (native / 'latest_checkpointed_iteration.txt').write_text('release')
            log.write_bytes(b'explicitly synthetic conversion log\n')
            return 0

        def synthetic_inspect(native, *, expected_iteration):
            self.assertEqual(expected_iteration, 'release')
            return {'iteration': 'release', 'selected': 'release', 'state_keys': ['model.synthetic'],
                    'referenced_files': {name: record(native / 'release' / name) for name in ('.metadata', '__0_0.distcp')},
                    'restore_tested': False}

        with patch.object(run, 'run_logged', side_effect=synthetic_conversion), \
                patch('runtime_support.checkpoint_evidence.inspect_checkpoint', side_effect=synthetic_inspect), \
                patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b,GPU-c'}):
            receipt = run.convert_native(home, code, model, converter, assets, source_receipt, 29400)
            self.assertEqual(receipt['converter_input'], converter)
            self.assertEqual(receipt['conversion_log_sha256'], record(home / 'native/conversion.log')['sha256'])
            reused = self.root / 'reused'
            shutil.copytree(home / 'native', reused)
            self.assertEqual(run.verify_conversion(reused, model, converter, assets, source_receipt), receipt)
            (reused / 'release/__0_0.distcp').write_bytes(b'edited')
            with self.assertRaises(ValueError):
                run.verify_conversion(reused, model, converter, assets, source_receipt)


if __name__ == '__main__':
    unittest.main()
