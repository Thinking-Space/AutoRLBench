"""Validate the configured image using AutoRLBench's real Docker stage wiring."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'task.toml').is_file())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.benchmark_root.resolve()))
    from autorlbench.core.types import EvaluationConfig
    from autorlbench.evaluation.evaluator import TaskEvaluator
    from autorlbench.runtimes import CommandSpec, SandboxRequest
    evaluator = TaskEvaluator('docker')
    package = evaluator.load(PROJECT)
    assert package.report.ok
    evaluation = package.paths.evaluations['issue']
    image = package.environment.environment_image
    artifact = evaluator.runtime.ensure_environment(image)
    config = EvaluationConfig(backend='docker', evaluation_level='issue', timeout_s=180,
                              keep_artifacts=True, work_dir=args.work_dir,
                              models_dir=args.work_dir / 'models', datasets_dir=args.work_dir / 'datasets',
                              verifier_network='none')
    run = evaluator._prepare_run(package, evaluation, config)
    before = {path.name: digest(path) for path in (PROJECT / 'workspace/logs/rollouts').glob('*.jsonl')}
    assert before, 'No source rollout storage files found'
    prepared = evaluator._run_prepare(package, evaluation, run, config, artifact)
    assert prepared.returncode == 0, prepared.stderr

    request = SandboxRequest(
        stage='agent', task_id=package.task_id, host_task_root=run.visible_task,
        host_visible_task=run.visible_task, host_workspace=run.workspace, host_results=run.results,
        evaluation_level='issue', host_environment=package.paths.environment,
        host_prepared=run.prepared, host_models=run.models, host_datasets=run.datasets,
        network='none', task_env=dict(package.environment.env),
    )
    tools = PROJECT / 'evaluations/issue/verifier/resources/tools'
    visibility_source = (tools / 'check_agent_visibility.py').read_text()
    manifest = json.loads((tools.parent / 'curated_trajectory_manifest.json').read_text())
    case_ids = [case['traj_uid'] for case in manifest['cases']]
    oracle_text = evaluation.oracle.read_text()
    answer_fragments = [paragraph for paragraph in oracle_text.split('\n\n')
                        if len(paragraph.split()) >= 20 and not paragraph.startswith('#')]
    # Supply hidden references to this diagnostic process only. Do not stage the
    # helper, audited IDs or oracle excerpts in any agent-visible file.
    visibility_code = (
        "from pathlib import Path\n"
        "visibility_scope = {'__name__': 'agent_visibility_check'}\n"
        f"exec({visibility_source!r}, visibility_scope)\n"
        "findings = visibility_scope['scan_agent_visible'](\n"
        "    Path('/workspace'), Path('/autorlbench/environment'), Path('/task/instruction.md'),\n"
        f"    case_ids={case_ids!r}, answer_fragments={answer_fragments!r})\n"
        "assert not findings, findings\n"
    )
    agent_code = visibility_code + '''import json, os
from pathlib import Path
assert os.environ['AUTORLBENCH_STAGE'] == 'agent'
assert not os.environ['AUTORLBENCH_ORACLE'] and not os.environ['AUTORLBENCH_VERIFIER']
assert {p.name for p in Path('/task').iterdir()} == {'instruction.md'}
assert not Path('/task/oracle.md').exists() and not Path('/task/verifier').exists()
assert not list(Path('/dev').glob('nvidia[0-9]*'))
episodes = set()
for path in Path('/workspace/logs/rollouts').glob('*.jsonl'):
    for line in path.open():
        episodes.add(json.loads(line)['traj_uid'])
assert len(episodes) == 20
Path('/results/agent_answer.md').write_text('Container transport fixture only; not an agent diagnosis.\\n')
print(json.dumps({'agent_visible_episodes': len(episodes), 'oracle_hidden': True,
                  'verifier_hidden': True, 'visible_annotation_scan': 'passed',
                  'network': 'none', 'gpu_required': False}))
'''
    agent = evaluator.runtime.run(request, CommandSpec(argv=['python3', '-B', '-c', agent_code], timeout_s=120), artifact=artifact)
    assert agent.returncode == 0, agent.stderr

    verifier_config = EvaluationConfig(
        backend='docker', evaluation_level='issue', timeout_s=180, verifier_network='none',
        verifier_cmd='python3 -B "$AUTORLBENCH_VERIFIER/resources/tools/container_verifier_checks.py"',
        models_dir=run.models, datasets_dir=run.datasets,
    )
    verifier = evaluator._run_verifier(package, evaluation, run, verifier_config, artifact, None)
    assert verifier.returncode == 0, verifier.stderr + verifier.stdout
    details = json.loads((run.results / 'container-verifier-checks.json').read_text())
    image_details = json.loads(subprocess.check_output(['docker', 'image', 'inspect', image.tag], text=True))[0]
    assert before == {path.name: digest(path) for path in (PROJECT / 'workspace/logs/rollouts').glob('*.jsonl')}
    report = {
        'checked_at': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
        'backend': 'docker', 'image_name': image.name, 'image_tag': image.tag,
        'image_id': image_details['Id'], 'image_size_bytes': image_details['Size'],
        'dockerfile_sha256': digest(image.dockerfile),
        'runtime_wiring': 'AutoRLBench TaskEvaluator and DockerRuntime',
        'prepare': {'exit_code': prepared.returncode, 'stdout': prepared.stdout.strip()},
        'agent_isolation': json.loads(agent.stdout), 'verifier': details,
        'historical_training_environment': 'conda seed',
        'source_rollouts_unchanged': True, 'training_rerun': False,
        'hosted_semantic_judge_tested': False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
