"""Exercise the real verifier in Docker with a loopback HTTP test fixture."""
from datetime import datetime, timezone
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading


def main():
    verifier = Path(os.environ['AUTORLBENCH_VERIFIER'])
    oracle = Path(os.environ['AUTORLBENCH_ORACLE'])
    results = Path(os.environ['AUTORLBENCH_RESULTS'])
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE'])
    assert verifier.is_dir() and oracle.is_file()
    assert os.environ['AUTORLBENCH_STAGE'] == 'verifier'
    assert not list(Path('/dev').glob('nvidia[0-9]*'))
    probe = workspace / '.container_read_only_probe'
    try:
        probe.write_text('permission probe')
    except OSError as error:
        assert error.errno == errno.EROFS, error
    else:
        probe.unlink()
        raise AssertionError('Verifier workspace is writable')

    # Replay utilities expect the full task layout, unlike the production judge.
    # Reconstruct it privately from the existing verifier mount and workspace.
    with tempfile.TemporaryDirectory(prefix='idea-issue-replay-') as temporary:
        root = Path(temporary)
        evaluation = root / 'evaluations/issue'
        shutil.copytree(verifier.parent, evaluation)
        (root / 'task.toml').write_text('# temporary offline replay layout\n')
        (root / 'workspace').symlink_to(workspace, target_is_directory=True)
        tools = evaluation / 'verifier/resources/tools'
        sys.path.insert(0, str(tools))
        spec = importlib.util.spec_from_file_location('container_curation', tools / 'curate_reward_trajectories.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        replay = module.replay(root)
        assert replay['episodes'] == 8
        assert all(case['semantic_contradiction_confirmed'] for case in replay['cases'])

    state = {'score': 1.0, 'malformed': False, 'requests': 0}

    class FixtureHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            assert self.path == '/v1/chat/completions'
            assert payload['model'] == 'container-transport-fixture'
            assert payload['response_format'] == {'type': 'json_object'}
            assert '<canonical_oracle>' in payload['messages'][1]['content']
            assert '<agent_diagnosis>' in payload['messages'][1]['content']
            state['requests'] += 1
            value = {'invalid_fixture_response': True} if state['malformed'] else {
                'verdict_score': state['score'], 'root_cause_score': state['score'],
                'evidence_score': state['score'], 'reason': 'Deterministic HTTP transport fixture; not a semantic judgment.',
            }
            response = json.dumps({'choices': [{'message': {'content': json.dumps(value)}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(('127.0.0.1', 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    checks = []
    try:
        for name, score, malformed, credentials, expected_exit in (
            ('pass', 1.0, False, True, 0),
            ('fail', 0.0, False, True, 1),
            ('malformed', 0.0, True, True, 2),
            ('missing_credentials', 0.0, False, False, 2),
        ):
            state.update(score=score, malformed=malformed)
            env = dict(os.environ)
            for key in tuple(env):
                if key.lower().endswith('_proxy'):
                    env.pop(key)
            env.update({
                'AUTORLBENCH_JUDGE_MODEL': 'container-transport-fixture',
                'OPENAI_BASE_URL': f'http://127.0.0.1:{server.server_port}/v1',
                'OPENAI_API_KEY': 'container-fixture-only' if credentials else '',
                'AUTORLBENCH_JUDGE_TIMEOUT_S': '10',
                'PYTHONDONTWRITEBYTECODE': '1',
            })
            output = results / ('transport-' + name + '.json')
            before = state['requests']
            process = subprocess.run(['bash', str(verifier / 'test.sh'), '--result', str(output)],
                                     env=env, capture_output=True, text=True, timeout=30)
            assert process.returncode == expected_exit, (name, process.returncode, process.stderr)
            result = json.loads(output.read_text())
            assert result['passed'] is (name == 'pass')
            if credentials:
                assert state['requests'] == before + 1
            else:
                assert state['requests'] == before
            checks.append({'case': name, 'exit_code': process.returncode, 'passed': True})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    report = {
        'checked_at': datetime.now(timezone.utc).isoformat(), 'python_version': platform.python_version(),
        'verifier_workspace_read_only': True, 'hidden_oracle_available_to_verifier': True,
        'gpu_required': False, 'replayed_counterexamples': replay['episodes'],
        'verifier_http_transport_checks': checks, 'fixture_http_requests': state['requests'],
        'hosted_semantic_judge_tested': False, 'training_rerun': False,
        'scope': 'Container mechanics and standard verifier HTTP transport; fixture scores are not benchmark performance.',
    }
    (results / 'container-verifier-checks.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
