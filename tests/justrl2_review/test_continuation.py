"""Fifteen collected continuation contracts; explicit, independently failing cases."""
import pytest

from tests.justrl2_review.fixtures import Fixture, inventory, load, write


@pytest.fixture
def case(tmp_path):
    return Fixture(tmp_path)


def test_exact_ids_zero_rewards_and_original_paths_are_accepted(case):
    before = inventory(case.scene)
    result = case.verify()
    assert result['score'] is True, result
    assert result['admitted_count'] == 64 and result['expected_groups'] == case.groups
    assert result['evaluation'] == {'step': 7, 'count': 16, 'reward_mean': 0.0}
    assert result['checkpoint_payloads_rehashed'] == {'actor': True, 'critic': True}
    assert inventory(case.scene) == before


def test_transient_attempts_keep_one_final_read_and_exact_admissions(case):
    case.trace(transient=True)
    result = case.verify()
    assert result['score'] is True, result
    assert result['steps'][0]['read_attempts'] == 16 and result['steps'][0]['abort_count'] == 0
    assert result['read_attempt_trace']['steps']['4'] == {'attempts': 18, 'missing': 2, 'final_reads': 16}


@pytest.mark.parametrize('damage', ['replacement_id', 'content_hash', 'metric_mask', 'rewind'])
def test_read_events_reject_replacement_content_loss_accounting_and_rewind(case, damage):
    def change(events):
        read = next(e for e in events if e['kind'] == 'read')
        if damage == 'replacement_id':
            replacement = case.rows[0]
            for event in events:
                if event.get('id') == case.groups['4'][0]:
                    event['id'] = replacement['id']
                    if event['kind'] == 'read':
                        event.update(sha256=replacement['sha256'], path=replacement['path'])
                    else:
                        event['label'] = case.payloads[replacement['id']]['label']
        elif damage == 'content_hash':
            read['sha256'] = 'f' * 64
        elif damage == 'metric_mask':
            next(e for e in events if e['kind'] == 'rollout_end')['metrics']['data/read_attempts'] = 1
        else:
            for event in events:
                if event['step'] == 4:
                    event['step'] = 0
    case.edit('run/events.jsonl', change, jsonl=True)
    assert case.verify()['score'] is False


def test_missing_terminal_cannot_be_hidden_by_reported_metrics(case):
    case.edit('run/events.jsonl', lambda events: events.remove(next(e for e in events if e['kind'] == 'terminal')), jsonl=True)
    assert case.verify()['score'] is False


def test_native_none_flags_and_list_prompt_data_preserve_config_contract(case):
    case.source_config({'no_load_optim': None, 'no_load_rng': False, 'no_save_rng': None})
    case.edit('run/effective_config.json', lambda config: config.update(no_load_optim=None, no_load_rng=False, no_save_rng=False))
    result = case.verify()
    assert result['score'] is True, result


@pytest.mark.parametrize('key,value', [('no_load_rng', True), ('start_rollout_id', 4)])
def test_enabled_restore_skip_and_explicit_restart_are_rejected(case, key, value):
    case.edit('run/effective_config.json', lambda config: config.update({key: value}))
    assert case.verify()['score'] is False


def test_empty_or_missing_fresh_checkpoint_is_not_success(case):
    (case.home / 'run/actor/iter_0000007/__0_0.distcp').unlink()
    assert case.verify()['score'] is False


def test_reset_optimizer_evidence_is_not_a_continuation(case):
    case.edit('run/optimizer_evidence/actor.jsonl',
              lambda rows: rows[0]['snapshot'].update(optimizer_steps=[0, 0]), jsonl=True)
    assert case.verify()['score'] is False


def test_changed_fixed_evaluation_identity_is_rejected(case):
    case.edit('run/eval_after_step_0000007.json',
              lambda value: value['fixed']['samples'][0]['metadata'].update(source_line=999))
    assert case.verify()['score'] is False


def test_scene_is_rehashed_after_the_audit(case, monkeypatch):
    verifier = load(case.bundle / 'verify_resume.py')
    evaluate = verifier._evaluation
    def change_scene(*args):
        result = evaluate(*args)
        (case.scene / 'checkpoints/critic/iter_0000003/common.pt').write_bytes(b'SYNTHETIC changed during audit')
        return result
    monkeypatch.setattr(verifier, '_evaluation', change_scene)
    proof = verifier.verify_resume(case.scene, case.home)
    assert proof['score'] is False and 'inventory' in proof['errors'][0].lower()


def test_relocated_model_pins_bind_actual_hf_bytes_and_snapshot(case):
    root, pins = case.model_assets()
    write(case.bundle.parent / 'snapshot.json', {'scene_sha256': inventory(case.scene)['SCENE.json']['sha256'],
          'model_revision': 'b' * 40, 'model_files': pins})
    proof = case.verify()
    assert proof['score'] is True, proof
    assert proof['model_assets'] == {'relocated': True, 'pins': 'snapshot', 'files_rehashed': 7}
    (root / 'model.safetensors').write_bytes(b'SYNTHETIC altered weights')
    assert case.verify()['score'] is False
