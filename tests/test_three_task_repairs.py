"""Regression tests for false-positive Solution gates in the 2026-11-23 audit."""
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / 'tasks/reward/slime_search_r1_nomask'
ICE = ROOT / 'tasks/configuration/areal_icepop_mask_window_too_narrow'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def search_workspace(tmp_path):
    for rel in ['examples/search-r1/generate_with_search.py', 'examples/search-r1/qa_em_format.py',
                'slime/__init__.py', 'slime/utils/__init__.py', 'slime/utils/types.py', 'slime/utils/misc.py']:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SEARCH / 'workspace' / rel, target)
    return tmp_path


@pytest.mark.parametrize('variant,passed,failed_check', [
    ('broken', False, 'observations_not_trainable'),
    ('body_only', False, 'observations_not_trainable'),
    ('mask_only', False, 'passage_body_reaches_model'),
    ('fixed', True, None),
    ('all_masked', False, 'model_tokens_trainable'),
    ('reward_shaping', False, 'eval_reward_is_binary_em'),
    ('equivalent', True, None),
])
def test_search_contract(search_workspace, variant, passed, failed_check):
    path = search_workspace / 'examples/search-r1/generate_with_search.py'
    text = path.read_text()
    if variant not in ('broken', 'mask_only'):
        text = text.replace('f"Doc {idx+1}(Title: {title})\\n"', 'f"Doc {idx+1}(Title: {title}) {text}\\n"')
    if variant not in ('broken', 'body_only'):
        text = text.replace('loss_mask += [1] * len(obs_tokens_ids)', 'loss_mask += [0] * len(obs_tokens_ids)')
    if variant == 'all_masked':
        text = text.replace('loss_mask += [1] * len(cur_response_token_ids)', 'loss_mask += [0] * len(cur_response_token_ids)')
    if variant == 'reward_shaping':
        text = text.replace('format_score=SEARCH_R1_CONFIGS["format_score"]', 'structure_format_score=SEARCH_R1_CONFIGS["format_score"]')
    if variant == 'equivalent':
        text = text.replace('loss_mask += [0] * len(obs_tokens_ids)', 'loss_mask.extend(0 for token in obs_tokens_ids)')
    path.write_text(text)
    completed = subprocess.run([sys.executable, '-B', str(SEARCH / 'evaluations/solution/verifier/tests/check_behavior.py'),
                                str(search_workspace)], capture_output=True, text=True, timeout=45)
    result = json.loads(completed.stdout)
    assert (result['status'] == 'passed') is passed, result
    if failed_check:
        assert result['checks'][failed_check] is False, result


def test_logs_alone_cannot_pass_search(tmp_path):
    grader = module(SEARCH / 'evaluations/solution/verifier/tests/grade_submission.py', 'search_grader')
    spec = grader.load_spec(SEARCH / 'evaluations/solution/verifier/resources/grader_spec.json')
    root = tmp_path / 'experiment_logs'; root.mkdir()
    (root / 'train.log').write_text("eval 99: {'eval/nq_test': 0.4}\n")
    _, _, old_result = grader.grade({**spec, 'behavior_contract': None}, tmp_path, tmp_path / 'absent')
    assert old_result['passed']
    _, metrics, result = grader.grade(spec, tmp_path, tmp_path / 'absent')
    assert metrics['final_score'] == 0.4
    assert not result['passed']
    assert metrics['behavior_verification']['status'] == 'unavailable'


def test_conflicting_metric_sources_fail_closed(tmp_path):
    grader = module(SEARCH / 'evaluations/solution/verifier/tests/grade_submission.py', 'search_conflicts')
    spec = grader.load_spec(SEARCH / 'evaluations/solution/verifier/resources/grader_spec.json')
    root = tmp_path / 'experiment_logs'; root.mkdir()
    (root / 'train.log').write_text("(RolloutManager) eval 99: {'eval/nq_test': 0.40}\n")
    (root / 'wandb.log').write_text("(RolloutManager) eval 99: {'eval/nq_test': 0.30}\n")
    _, metrics, result = grader.grade({**spec, 'behavior_contract': None}, tmp_path, tmp_path / 'absent')
    assert metrics['metric_conflicts'][0]['step'] == 99
    assert not result['passed']
    assert any('conflicting values' in error for error in metrics['errors'])


def test_search_contract_is_mandatory():
    grader = module(SEARCH / 'evaluations/solution/verifier/tests/grade_submission.py', 'search_spec')
    path = SEARCH / 'evaluations/solution/verifier/resources/grader_spec.json'
    value = json.loads(path.read_text())
    value.pop('behavior_contract')
    broken = path.parent / 'grader_spec.missing-contract.json'
    broken.write_text(json.dumps(value))
    try:
        with pytest.raises(ValueError, match='missing required fields'):
            grader.load_spec(broken)
    finally:
        broken.unlink()


@pytest.mark.parametrize('action,lower,upper,fraction,missing,expected', [
    ('mask', 0.67, 1.5, 0.7, False, 'failed'),
    ('mask', 0.67, 1.5, 0.04, False, 'passed'),
    ('mask', 0.67, 1.5, 0.04, True, 'unavailable'),
    ('mask', 0.5, 5, 0.04, False, 'passed'),
    ('clamp', 0.67, 1.5, 0.7, False, 'passed'),
])
def test_icepop_metric_pass_requires_repair(tmp_path, action, lower, upper, fraction, missing, expected):
    grader = module(ICE / 'evaluations/solution/verifier/tests/grade_submission.py', 'ice_grader')
    spec = grader.load_spec(ICE / 'evaluations/solution/verifier/resources/grader_spec.json')
    run = tmp_path / 'run/current'; run.mkdir(parents=True)
    rows = []
    for i in range(1, 88):
        rows.append(f'Train step {i}/87 done.')
        if not missing or i != 60:
            rows.append(f'│ ppo_actor/update/rs_filtered_fraction │ {fraction} │')
        if i % 29 == 0:
            rows.append('│ eval-rollout/reward │ 0.67 │')
    (run / 'main.log').write_text('\n'.join(rows))
    (run / 'config.yaml').write_text(yaml.safe_dump({'actor': {'rejection_sampling': {
        'action': action, 'metric': 'ratio', 'level': 'token', 'lower': lower, 'upper': upper}}}))
    _, metrics, result = grader.grade(spec, tmp_path)
    assert metrics['final_score'] == 0.67
    assert metrics['mask_repair_verification']['status'] == expected
    assert result['passed'] is (expected == 'passed')


def test_icepop_original_defect_rejected():
    grader = module(ICE / 'evaluations/solution/verifier/tests/grade_submission.py', 'ice_frozen')
    spec = grader.load_spec(ICE / 'evaluations/solution/verifier/resources/grader_spec.json')
    result = grader.mask_repair_check(spec, ICE / 'workspace')
    assert result['status'] == 'failed', result
    assert result['masked_fraction_steps'] == 87
    assert result['masked_fraction_max'] > 0.7
