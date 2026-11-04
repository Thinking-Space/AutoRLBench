# SPDX-License-Identifier: Apache-2.0
"""Four real CPU Adam/DCP contracts; no GPU execution or native model claims."""
import copy
import pickle
from pathlib import Path

import pytest
import torch.distributed.checkpoint as dcp

from tests.justrl2_review.restore.fixtures import native_scene, restore, state_dict, update


def test_cpu_adam_dcp_restore_four_updates_and_final_readback(native_scene):
    scene = native_scene
    assert restore(scene) == (3, 12345)
    loaded = scene.events[0]
    assert loaded['payload_exact'] and loaded['scheduler_exact'] and loaded['rng_exact']
    assert loaded['snapshot']['optimizer_steps'] == [4.0, 4.0]
    assert loaded['scheduler_num_steps'] == 64
    for rollout in range(4, 8):
        update(scene.model, scene.optimizer)
        scene.scheduler.state['num_steps'] += 16
        scene.h.restore_after_optimizer_step(scene.args, scene.optimizer, rollout, step_id=0)
    selected = Path(scene.args.save) / 'iter_0000007'
    dcp.save(copy.deepcopy(state_dict(scene.model, scene.optimizer)), checkpoint_id=selected)
    scene.h.restore_after_save(scene.args, 7)
    saved = scene.events[-1]
    assert saved['saved_iteration'] == 7 and saved['step_delta'] == 4 and saved['payload_exact']
    assert saved['snapshot']['optimizer_steps'] == [8.0, 8.0]
    assert saved['snapshot']['model_sha256'] != loaded['snapshot']['model_sha256']
    assert scene.h.fingerprint(scene.saved) == loaded['snapshot']
    pickle.dumps(scene.args)


@pytest.mark.parametrize('damage', ['skip_rng', 'skip_scheduler', 'wrong_counter'])
def test_cpu_native_readback_rejects_reset_auxiliary_or_adam_state(native_scene, damage):
    scene = native_scene
    scene.flags[damage] = True
    with pytest.raises(ValueError):
        restore(scene)
    assert not scene.events
    assert scene.cp.generate_state_dict is scene.generate and scene.dc.load is scene.reader
    assert scene.args.bench_role not in scene.h._RESTORE_CONTEXT
