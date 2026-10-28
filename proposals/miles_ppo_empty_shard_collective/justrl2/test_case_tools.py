# SPDX-License-Identifier: Apache-2.0
"""Tooling tests use synthetic records; they are never training evidence."""
from datetime import timedelta
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_replay import cp_indices
import runtime_evidence
from sglang_compat import validate_with_aliases
from verify_run import verify_healthy, verify_faulty
from checkpoint_evidence import inspect_checkpoint, convert_bin_to_safetensors


def record(root, rows):
    (root / "evidence").mkdir(parents=True, exist_ok=True)
    (root / "evidence/events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))


def synthetic_success_records():
    rows = []
    # Each role: CP0 group [0,2], CP1 group [1,3]. Values match analytical
    # sample variance, giving [-1,0,1] for the CP1 nonempty rank.
    vals = {0: [-2.0], 1: [], 2: [0.0, 2.0], 3: [-1.0, 0.0, 1.0]}
    for role in ["actor", "critic"]:
        for rank in range(4):
            identity = dict(role=role, rank=rank, pid=(100 if role == "actor" else 200) + rank,
                            cp_size=2, dp_size=2,
                            cp_rank=rank % 2, dp_rank=rank // 2,
                            cp_members=[0, 1] if rank < 2 else [2, 3],
                            dp_members=[rank % 2, rank % 2 + 2])
            row = dict(identity, event="whitening_enter", local_numel=len(vals[rank]),
                       local_advantages=vals[rank], local_masks=[1] * len(vals[rank]))
            rows.append(row)
            rows.append(dict(identity, event="whitening_call", local_numel=len(vals[rank])))
            normalized = [v / 2 for v in vals[rank]] if rank % 2 == 0 else vals[rank]
            rows.append(dict(identity, event="whitening_complete", normalized=normalized))
            rows.append(dict(identity, event="advantages_ready", local_numel=len(vals[rank])))
            rows.append(dict(identity, event="optimizer_step", update_successful=True,
                             grad_norm_finite=True, sampled_finite=True,
                             sampled_changed=rank == 2, rollout_id=0, step_id=0))
    return rows


def success_layout(root, rows):
    record(root, rows)
    (root / "TRAINING_RETURNED.json").write_text("{}")
    for role in ["actor", "critic"]:
        import torch.distributed.checkpoint as dcp
        checkpoint = root / role
        checkpoint.mkdir()
        (checkpoint / "latest_checkpointed_iteration.txt").write_text("0")
        dcp.save({"model": {"weight": torch.ones(2, 2)},
                  "optimizer": {"exp_avg": torch.zeros(2, 2)}},
                 checkpoint_id=checkpoint / "iter_0000000")


def test_geometry():
    assert [len(cp_indices(424, 2, cp)) for cp in range(2)] == [2, 0]
    assert [len(cp_indices(357, 329, cp)) for cp in range(2)] == [149, 180]
    for total, response in [(424, 2), (357, 329)]:
        indices = [x for cp in range(2) for x in cp_indices(total, response, cp)]
        assert sorted(indices) == list(range(response))


def test_sglang_parser_aliases_preserve_validation():
    args = SimpleNamespace(sglang_dp_size=2, sglang_pp_size=1, sglang_ep_size=4)
    calls = []
    validate_with_aliases(args, lambda value: calls.append(value))
    assert calls == [args]
    assert (args.sglang_data_parallel_size, args.sglang_pipeline_parallel_size, args.sglang_expert_parallel_size) == (2, 1, 4)
    args.sglang_data_parallel_size = 9
    with pytest.raises(ValueError, match="Conflicting"):
        validate_with_aliases(args, lambda _: pytest.fail("must not validate conflicting values"))
    with pytest.raises(ValueError, match="missing"):
        validate_with_aliases(SimpleNamespace(), lambda _: pytest.fail("must not invent defaults"))


def test_success_checker_and_corrupted_values(tmp_path):
    rows = synthetic_success_records()
    success_layout(tmp_path, rows)
    assert verify_healthy(tmp_path)["checkpoint_restore_tested"] is False
    row = next(r for r in rows if r["event"] == "whitening_complete" and r["rank"] == 3)
    row["normalized"] = [99, 99, 99]
    record(tmp_path, rows)
    with pytest.raises(ValueError, match="Whitening differs"):
        verify_healthy(tmp_path)


@pytest.mark.parametrize("fault", ["no_update", "missing_rank", "wrong_group", "no_checkpoint", "nan"])
def test_success_checker_rejects_invalid_evidence(tmp_path, fault):
    rows = synthetic_success_records()
    if fault == "no_update":
        for row in rows:
            if row["event"] == "optimizer_step":
                row["sampled_changed"] = False
    elif fault == "missing_rank":
        rows = [r for r in rows if not (r["event"] == "optimizer_step" and r["rank"] == 3)]
    elif fault == "wrong_group":
        rows[0]["dp_members"] = [0, 1]
    elif fault == "nan":
        rows[0]["local_advantages"] = [float("nan")]
    success_layout(tmp_path, rows)
    if fault == "no_checkpoint":
        next((tmp_path / "critic/iter_0000000").glob("*.distcp")).unlink()
    with pytest.raises(ValueError):
        verify_healthy(tmp_path)


def test_faulty_checker_requires_divergence_and_actual_watchdog(tmp_path):
    rows = [r for r in synthetic_success_records()
            if r["event"] == "whitening_enter"
            or (r["event"] == "whitening_complete" and r["rank"] % 2 == 0)
            or (r["event"] == "whitening_call" and r["rank"] in [0, 2, 3])
            or (r["event"] == "advantages_ready" and r["rank"] in [0, 1, 2])]
    record(tmp_path, rows)
    (tmp_path / "stdout.log").write_text("generic timeout; Timeout(ms)=60000")
    with pytest.raises(ValueError, match="target NCCL collective failure"):
        verify_faulty(tmp_path)
    (tmp_path / "stdout.log").write_text("Watchdog caught collective operation timeout")
    with pytest.raises(ValueError, match="correlated"):
        verify_faulty(tmp_path)
    (tmp_path / "stdout.log").write_text(
        "(Actor pid=103) [rank3]: Watchdog caught collective operation timeout\n"
        "(Actor pid=203) [rank3]: Watchdog caught collective operation timeout\n")
    assert verify_faulty(tmp_path)["faulty_evidence_passed"]
    (tmp_path / "stdout.log").write_text(
        "File distributed_c10d.py, line 3068, in all_reduce\n"
        "torch.distributed.DistBackendError: store->get('0') got error: "
        "wait timeout after 60000ms\n")
    assert verify_faulty(tmp_path)["failure_evidence"] == "store_wait_timeout"
    rows.append(next(r for r in synthetic_success_records() if r["event"] == "whitening_complete" and r["rank"] == 3))
    record(tmp_path, rows)
    with pytest.raises(ValueError, match="peer completed"):
        verify_faulty(tmp_path)


def test_checkpoint_requires_metadata_referenced_shards(tmp_path):
    success_layout(tmp_path, synthetic_success_records())
    assert inspect_checkpoint(tmp_path / "actor", expected_iteration=0, require_optimizer=True)["restore_tested"] is False
    shard = next((tmp_path / "actor/iter_0000000").glob("*.distcp"))
    shard.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="truncated referenced"):
        inspect_checkpoint(tmp_path / "actor", expected_iteration=0, require_optimizer=True)


def test_checkpoint_accepts_flat_megatron_tensor_keys(tmp_path):
    import torch.distributed.checkpoint as dcp
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    dcp.save({"embedding.word_embeddings.weight": torch.ones(2, 2),
              "decoder.layers.0.self_attention.linear_qkv.weight": torch.ones(2, 2),
              "output_layer.weight": torch.ones(2, 2)}, checkpoint_id=tmp_path / "release")
    result = inspect_checkpoint(tmp_path, expected_iteration="release")
    assert "embedding.word_embeddings.weight" in result["state_keys"]


def test_lossless_safetensors_conversion(tmp_path):
    from safetensors import safe_open
    source = tmp_path / "pytorch_model.bin"
    shared = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    tensors = {"model.embed_tokens.weight": shared, "lm_head.weight": shared,
               "model.norm.weight": torch.tensor([1., 2., 3.], dtype=torch.float32)}
    torch.save(tensors, source)
    before = source.read_bytes()
    target = tmp_path / "model.safetensors"
    receipt = convert_bin_to_safetensors(source, target)
    assert source.read_bytes() == before
    assert receipt["tensor_count"] == 3 and receipt["all_tensors_exactly_equal"] is True
    with safe_open(str(target), framework="pt", device="cpu") as saved:
        assert set(saved.keys()) == set(tensors)
        for key, original in tensors.items():
            assert torch.equal(original, saved.get_tensor(key))
            assert original.dtype == saved.get_tensor(key).dtype
    with pytest.raises(ValueError, match="already exists"):
        convert_bin_to_safetensors(source, target)


def test_safetensors_rejects_non_tensor_state(tmp_path):
    source = tmp_path / "bad.bin"
    torch.save({"model.weight": torch.ones(2), "description": "not a weight"}, source)
    with pytest.raises(ValueError, match="tensor state dict"):
        convert_bin_to_safetensors(source, tmp_path / "bad.safetensors")


def test_real_adam_snapshot_and_durable_evidence(tmp_path, monkeypatch):
    torch.manual_seed(5)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model(torch.ones(3, 4)).square().mean()
    loss.backward()
    before = runtime_evidence.snapshot_optimizer(opt)
    saved = {k: v[2].clone() for k, v in before.items()}
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
    opt.step()
    rendezvous = tmp_path / "rendezvous"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1,
                            timeout=timedelta(seconds=5))
    group = SimpleNamespace(rank=0, size=1, group=dist.group.WORLD)
    module = ModuleType("miles.backends.training_utils.parallel")
    module.get_parallel_state = lambda: SimpleNamespace(cp=group, intra_dp=group)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv("BENCH_EVIDENCE_DIR", str(tmp_path / "real_evidence"))
    try:
        event = runtime_evidence.after_optimizer(SimpleNamespace(bench_role="actor"), opt, before, True, norm, 0, 0)
        assert event["sampled_changed"] and event["max_delta"] > 0 and event["sampled_finite"]
        assert event["sampled_bytes"] <= 2 * 1024**2
        for key in before:
            torch.testing.assert_close(before[key][2], saved[key])
        logs = list((tmp_path / "real_evidence").glob("*.jsonl"))
        assert len(logs) == 1 and json.loads(logs[0].read_text())["update_successful"] is True
    finally:
        dist.destroy_process_group()
