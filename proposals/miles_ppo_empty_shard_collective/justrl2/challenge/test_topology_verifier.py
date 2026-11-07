# SPDX-License-Identifier: Apache-2.0
"""Offline tests execute a pinned native-function snapshot through real Gloo."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
VERIFIER = HERE / "topology_verifier.py"


def verifier():
    assert VERIFIER.is_file(), "maintainer topology verifier has not been implemented"
    spec = importlib.util.spec_from_file_location("topology_verifier", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def native_source(tmp_path):
    root = tmp_path / "source"
    for relative, text in NATIVE_SOURCE.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_fixed_native_source_all_topologies(native_source, tmp_path):
    result = verifier().run_suite(native_source, tmp_path / "results")
    assert result["status"] == "pass", result
    assert result["passed"] is True
    assert len(result["topologies"]) == 4
    assert {r["cp_size"] for r in result["topologies"]} == {1, 2, 4}
    assert any(r["logical_to_global"] != list(range(r["world_size"]))
               for r in result["topologies"])
    assert all(len(r["cases"]) == len(verifier().SCENARIOS) for r in result["topologies"])
    zero_cases = [c for r in result["topologies"] for c in r["cases"]
                  if c["scenario"] == "global_zero_mask"]
    assert all(c["expected_zero_count_ranks"] for c in zero_cases)
    assert "AST" in result["coverage_limits"]
    assert result["gpu_training_validated"] is False
    assert json.loads((tmp_path / "results/summary.json").read_text()) == result


LOSS = "miles/backends/training_utils/loss.py"
WHITEN = "miles/utils/distributed_utils.py"
PPO = "miles/utils/ppo_utils.py"
CP = "miles/backends/training_utils/cp_utils.py"


def rewrite(source, relative, before, after):
    path = source / relative
    text = path.read_text()
    assert text.count(before) == 1, before
    path.write_text(text.replace(before, after))


def guard_whitening(source, condition):
    path = source / LOSS
    text = path.read_text()
    begin = text.index("        whitened_advs_flat = distributed_masked_whiten(")
    end = text.index("        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))", begin)
    end = text.index("\n", end) + 1
    block = text[begin:end]
    path.write_text(text[:begin] + "        if " + condition + ":\n" +
                    "".join("    " + line for line in block.splitlines(keepends=True)) + text[end:])


def run(source, tmp_path, **kwargs):
    return verifier().run_suite(source, tmp_path / "results", **kwargs)


def all_failures(result):
    return [f for topology in result["topologies"] for f in topology["failures"]]


@pytest.mark.parametrize("condition,scenario", [
    ("all_masks.numel() > 0", "asymmetric_empty"),
    ("all_advs.numel() > 0", "asymmetric_empty"),
    ("all_masks.sum() > 0", "zero_masks_nonempty"),
])
def test_local_guards_prove_collective_divergence(native_source, tmp_path, condition, scenario):
    guard_whitening(native_source, condition)
    result = run(native_source, tmp_path, topologies=["cp2_dp2"], scenarios=[scenario],
                 collective_timeout=1.5)
    assert result["status"] == "fail", result
    assert result["observed_intended_mismatch"] is True
    failures = all_failures(result)
    assert any(f["kind"] == "collective_participation_mismatch" and f["witnesses"] for f in failures)
    assert not any(f["kind"] == "infrastructure_error" for f in failures)


@pytest.mark.parametrize("mutation", ["no_normalization", "identity_whiten", "cp_group", "world_group",
                                     "numel_count", "population_variance", "bad_returns", "bad_slice", "nan"])
def test_real_source_mutations_fail(native_source, tmp_path, mutation):
    if mutation == "no_normalization":
        rewrite(native_source, LOSS, "if args.normalize_advantages:", "if False:")
    elif mutation == "identity_whiten":
        rewrite(native_source, WHITEN, "return whitened_values", "return values")
    elif mutation == "cp_group":
        rewrite(native_source, LOSS, "dp_group = parallel_state.intra_dp.group", "dp_group = parallel_state.cp.group")
    elif mutation == "world_group":
        rewrite(native_source, LOSS, "dp_group = parallel_state.intra_dp.group", "dp_group = dist.group.WORLD")
    elif mutation == "numel_count":
        rewrite(native_source, WHITEN, "local_mask_sum = mask.sum()", "local_mask_sum = values.numel()")
    elif mutation == "population_variance":
        rewrite(native_source, WHITEN, "global_var = global_var * bessel_correction", "global_var = global_var")
    elif mutation == "bad_returns":
        rewrite(native_source, PPO, "return (advantages_list, returns_list)",
                "return (advantages_list, [x + 0.25 for x in returns_list])")
    elif mutation == "bad_slice":
        rewrite(native_source, CP, "return torch.cat([chunk_1, chunk_2], dim=0)",
                "return torch.cat([chunk_1, chunk_2], dim=0).flip(0)")
    else:
        rewrite(native_source, WHITEN, "return whitened_values", "return whitened_values * float('nan')")
    result = run(native_source, tmp_path, topologies=["cp2_dp2"], scenarios=["baseline", "uneven_counts"])
    assert result["status"] == "fail", result
    assert result["passed"] is False
    assert result["observed_intended_mismatch"] is False
    kinds = {f["kind"] for f in all_failures(result)}
    assert "numerical_mismatch" in kinds or "nonfinite" in kinds, kinds
    if mutation in ["cp_group", "world_group"]:
        assert "wrong_process_group" in kinds


def test_global_zero_count_must_raise_on_every_member(native_source, tmp_path):
    rewrite(native_source, WHITEN,
            "raise ValueError('The global mask sum across all participating GPUs is zero.')",
            "return torch.zeros_like(values)")
    result = run(native_source, tmp_path, topologies=["cp2_dp2"],
                 scenarios=["global_zero_mask", "empty_dp_group"])
    assert result["status"] == "fail", result
    assert any(f["kind"] == "expected_zero_count_error" for f in all_failures(result))


@pytest.mark.parametrize('mutation', ['floor_cp_chunks', 'singleton_identity'])
def test_rounding_and_single_token_mutations_fail(native_source, tmp_path, mutation):
    if mutation == 'floor_cp_chunks':
        rewrite(native_source, CP, '(total_length + 2 * cp_size - 1) // (2 * cp_size)',
                'total_length // (2 * cp_size)')
    else:
        rewrite(native_source, WHITEN, 'global_mean = global_sum / global_mask_sum',
                'if global_mask_sum.item() == 1:\n        return values\n    global_mean = global_sum / global_mask_sum')
    result = run(native_source, tmp_path, topologies=['cp2_dp2'])
    assert result['status'] == 'fail', result
    assert result['passed'] is False
    assert any(f['kind'] == 'numerical_mismatch' for f in all_failures(result))


def test_missing_source_is_infrastructure_error(tmp_path):
    result = run(tmp_path / "missing", tmp_path, topologies=["cp1_dp2"])
    assert result["status"] == "infrastructure_error"
    assert result["observed_intended_mismatch"] is False


def test_generic_error_is_not_reproduction(native_source, tmp_path):
    rewrite(native_source, WHITEN, "local_sum = (values * mask).sum()",
            "raise RuntimeError('generic timed out')\n    local_sum = (values * mask).sum()")
    result = run(native_source, tmp_path, topologies=["cp2_dp2"], scenarios=["asymmetric_empty"])
    assert result["status"] == "fail", result
    assert result["observed_intended_mismatch"] is False


def test_worker_exit_and_missing_rank_fail_closed(native_source, tmp_path):
    rewrite(native_source, WHITEN, "local_sum = (values * mask).sum()",
            "__import__('os')._exit(23)\n    local_sum = (values * mask).sum()")
    result = run(native_source, tmp_path, topologies=["cp1_dp2"], scenarios=["baseline"])
    assert result["status"] == "infrastructure_error", result
    assert result["observed_intended_mismatch"] is False


def test_outer_timeout_is_never_reproduction(native_source, tmp_path):
    result = run(native_source, tmp_path, topologies=["cp2_dp2"], scenarios=["asymmetric_empty"],
                 outer_timeout=0.01)
    assert result["status"] == "infrastructure_error", result
    assert result["observed_intended_mismatch"] is False
    assert any("outer timeout" in f["message"].lower() for f in all_failures(result))


def test_output_must_be_fresh_and_cli_error_exit(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(FileExistsError):
        verifier().run_suite(tmp_path / "missing", out)
    process = subprocess.run([sys.executable, str(VERIFIER), "--source", str(tmp_path / "missing"),
                              "--output", str(tmp_path / "cli")], capture_output=True, text=True, timeout=10)
    assert process.returncode == 2, process.stderr
    assert json.loads((tmp_path / "cli/summary.json").read_text())["status"] == "infrastructure_error"


@pytest.mark.skipif(not os.environ.get("JUSTRL2_SOURCE"), reason="optional pinned-checkout integration")
def test_snapshot_matches_pinned_source_and_native_cli(tmp_path):
    import ast
    source = Path(os.environ["JUSTRL2_SOURCE"])
    native, _ = verifier()._source_nodes(source)
    for relative, nodes in native.items():
        snapshot = ast.parse(NATIVE_SOURCE[relative])
        assert ast.dump(ast.Module(body=nodes, type_ignores=[])) == ast.dump(snapshot), relative
    process = subprocess.run([sys.executable, str(VERIFIER), "--source", str(source),
                              "--output", str(tmp_path / "native")], capture_output=True, text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    assert json.loads((tmp_path / "native/summary.json").read_text())["passed"]


# Native code snapshot: merak0514/JustRL-II, commit
# 75814318f243af7e39af337c1173fc3927c9635b. ast.unparse only removes
# comments and formatting; no kernel is reimplemented by these tests.
NATIVE_SOURCE = {
'miles/backends/training_utils/cp_utils.py': r'''
def _parallel_rank(parallel_state, name: str) -> int:
    group_info = getattr(parallel_state, name, None)
    if group_info is not None and hasattr(group_info, 'rank'):
        return group_info.rank
    legacy_name = f'{name}_rank'
    if hasattr(parallel_state, legacy_name):
        return getattr(parallel_state, legacy_name)
    raise AttributeError(f'ParallelState has neither {name}.rank nor {legacy_name}')

def _parallel_size(parallel_state, name: str) -> int:
    group_info = getattr(parallel_state, name, None)
    if group_info is not None and hasattr(group_info, 'size'):
        return group_info.size
    legacy_name = f'{name}_size'
    if hasattr(parallel_state, legacy_name):
        return getattr(parallel_state, legacy_name)
    raise AttributeError(f'ParallelState has neither {name}.size nor {legacy_name}')

def get_logits_and_tokens_offset_with_cp(total_length: int, response_length: int, *args, qkv_format: str='thd', max_seq_len: int | None=None):
    """
    All offsets start from the begining of the prompt.

    Compatible call forms:
    - get_logits_and_tokens_offset_with_cp(total_length, response_length, qkv_format, max_seq_len)
    - get_logits_and_tokens_offset_with_cp(total_length, response_length, parallel_state, qkv_format, max_seq_len)
    """
    parallel_state = None
    if args:
        first_arg = args[0]
        if isinstance(first_arg, str):
            qkv_format = first_arg
            if len(args) > 1:
                max_seq_len = args[1]
            if len(args) > 2:
                raise TypeError('Too many positional arguments for get_logits_and_tokens_offset_with_cp')
        else:
            parallel_state = first_arg
            if len(args) > 1:
                qkv_format = args[1]
            if len(args) > 2:
                max_seq_len = args[2]
            if len(args) > 3:
                raise TypeError('Too many positional arguments for get_logits_and_tokens_offset_with_cp')
    if parallel_state is None:
        parallel_state = get_parallel_state()
    cp_rank = _parallel_rank(parallel_state, 'cp')
    cp_size = _parallel_size(parallel_state, 'cp')
    assert cp_size > 1
    prompt_length = total_length - response_length
    if qkv_format == 'thd':
        chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    else:
        assert max_seq_len is not None, 'max_seq_len must be provided for qkv_format=bshd'
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))
    if logits_0[0] < logits_0[1]:
        token_0 = (logits_0[0] + 1, logits_0[1] + 1)
    else:
        logits_0 = (0, 0)
        token_0 = (0, 0)
    if logits_1[0] < logits_1[1]:
        token_1 = (logits_1[0] + 1, logits_1[1] + 1)
    else:
        logits_1 = (0, 0)
        token_1 = (0, 0)
    return (chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1))

def local_response_to_full(tensor: torch.Tensor, total_length: int, response_length: int, qkv_format: str='thd', max_seq_len: int | None=None) -> torch.Tensor:
    """Scatter this CP rank's local response shard back into full-length response coordinates (missing positions zero-filled, no communication)."""
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length, qkv_format, max_seq_len)
    prompt_length = total_length - response_length
    chunk_0 = tensor[:logits_offset[0][1] - logits_offset[0][0]]
    chunk_1 = tensor[logits_offset[0][1] - logits_offset[0][0]:]
    assert chunk_1.shape[0] == logits_offset[1][1] - logits_offset[1][0]

    def zero(len: int) -> torch.Tensor:
        return torch.zeros([len] + list(tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device, requires_grad=True)
    if chunk_0.shape[0] == 0 and chunk_1.shape[0] == 0:
        full_tensor = zero(response_length)
    elif chunk_0.shape[0] != 0 and chunk_1.shape[0] == 0:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[0][1])
        full_tensor = torch.cat([left, chunk_0, right], dim=0)
    elif chunk_0.shape[0] == 0 and chunk_1.shape[0] != 0:
        left = zero(logits_offset[1][0] - (prompt_length - 1))
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_1, right], dim=0)
    else:
        left = zero(logits_offset[0][0] - (prompt_length - 1))
        mid = zero(logits_offset[1][0] - logits_offset[0][1])
        right = zero(total_length - 1 - logits_offset[1][1])
        full_tensor = torch.cat([left, chunk_0, mid, chunk_1, right], dim=0)
    assert full_tensor.shape[0] == response_length, f'Expected {response_length}, got {full_tensor.shape}'
    return full_tensor

def slice_log_prob_with_cp(log_prob: list[float] | torch.Tensor, total_length: int, response_length: int, qkv_format: str='thd', max_token_len: int | None=None) -> list[float] | torch.Tensor:
    assert len(log_prob) == response_length
    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size
    if cp_size == 1:
        return log_prob
    prompt_length = total_length - response_length
    _, _, logits_offset, _ = get_logits_and_tokens_offset_with_cp(total_length, response_length, qkv_format, max_token_len)
    chunk_1 = log_prob[logits_offset[0][0] - (prompt_length - 1):logits_offset[0][1] - (prompt_length - 1)]
    chunk_2 = log_prob[logits_offset[1][0] - (prompt_length - 1):logits_offset[1][1] - (prompt_length - 1)]
    if isinstance(log_prob, list):
        return chunk_1 + chunk_2
    else:
        return torch.cat([chunk_1, chunk_2], dim=0)
''',
'miles/utils/ppo_utils.py': r'''
def length_adaptive_lambda(k: float, response_lengths, device, dtype=torch.float32) -> torch.Tensor:
    """JustRL2 length-adaptive GAE λ: per-sample ``λ_i = k ** (1 / L_i)``.

    ``k`` is the fraction of terminal credit that reaches the first token: with γ=1 the
    GAE weight of the terminal reward at distance d is ``λ_i**d``, so the first token
    always sees ``λ_i**L_i == k`` regardless of response length. Longer responses get λ
    closer to 1, so credit propagation does not weaken with length. (The earlier
    form ``1 − 1/(α·L)`` is the first-order expansion of this with ``k = exp(−1/α)``.)

    Computed in float32 regardless of ``dtype``: at 128k lengths ``k**(1/L)`` is
    ``1 − 8e-6``-ish, which bf16 rounds to exactly 1.0. ``L_i == 0`` (empty response)
    gives λ=0; that row is fully masked downstream, so the value is irrelevant.
    """
    assert 0.0 < k <= 1.0, f'--gae-lambda-k must be in (0, 1], got {k}'
    lengths = torch.tensor(response_lengths, device=device, dtype=torch.float32)
    lam = torch.where(lengths > 0, float(k) ** (1.0 / lengths.clamp(min=1.0)), torch.zeros_like(lengths))
    out = lam.to(dtype)
    if k < 1.0 and out.numel() and (out[lam > 0] >= 1.0).any():
        raise AssertionError(f'length-adaptive lambda collapsed to 1.0 under dtype={dtype}: the length-adaptive GAE is a no-op (k={k}, max response_len={max(response_lengths)}). {dtype} cannot represent k**(1/L) for L larger than ~355. Keep values/log_probs in float32, or pass an explicit float32 dtype to length_adaptive_lambda().')
    return out

def get_advantages_and_returns_batch(total_lengths, response_lengths, values_list, rewards_list, gamma, lambd, chunked: bool=True, length_adaptive_lambda_k: 'float | None'=None):
    """
    Batched GAE with CP support.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each shape = [resp_len_i]
        rewards_list:      list[Tensor], same shape
        length_adaptive_lambda_k: if set (JustRL2 length-adaptive decoupled GAE), advantages use
            a per-sample λ_i = k^(1/L_i) (the first token always receives k times the terminal
            credit, independent of length), while returns use the unbiased λ=1 target (the
            reward suffix sum when γ=1); the two are decoupled.
    Output:
        advantages_list:   list[Tensor], each shape = [resp_len_i]
        returns_list:      list[Tensor], same shape
    """
    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)
        cp_size = get_parallel_state().cp.size
        device = values_list[0].device
        dtype = values_list[0].dtype
        max_len = max(response_lengths)
        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)
        if cp_size > 1:
            from miles.backends.training_utils.cp_utils import local_response_to_full
            for i, (total_len, resp_len, v, r) in enumerate(zip(total_lengths, response_lengths, values_list, rewards_list, strict=False)):
                full_values[i, :resp_len] = local_response_to_full(v, total_len, resp_len)
                full_rewards[i, :resp_len] = local_response_to_full(r.to(dtype), total_len, resp_len)
            cp_group = get_parallel_state().cp.group
            dist.all_reduce(full_values, group=cp_group)
            dist.all_reduce(full_rewards, group=cp_group)
        else:
            for i in range(B):
                L = response_lengths[i]
                full_values[i, :L] = values_list[i][:L]
                full_rewards[i, :L] = rewards_list[i][:L]
        if length_adaptive_lambda_k:
            assert gamma == 1.0, 'length-adaptive decoupled GAE requires gamma == 1.0 (returns = suffix reward sum)'
            lambd_rowwise = length_adaptive_lambda(length_adaptive_lambda_k, response_lengths, device).to(dtype)
            full_advantages, _ = chunked_gae(rewards=full_rewards, values=full_values, gamma=gamma, lambd=lambd_rowwise)
            full_returns = torch.flip(torch.cumsum(torch.flip(full_rewards, dims=[1]), dim=1), dims=[1])
        elif not chunked:
            full_advantages, full_returns = vanilla_gae(rewards=full_rewards, values=full_values, gamma=gamma, lambd=lambd)
        else:
            full_advantages, full_returns = chunked_gae(rewards=full_rewards, values=full_values, gamma=gamma, lambd=lambd)
        advantages_list = []
        returns_list = []
        if cp_size > 1:
            from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp
            for total_len, resp_len, adv_row, ret_row in zip(total_lengths, response_lengths, full_advantages, full_returns, strict=False):
                adv_full = adv_row
                ret_full = ret_row
                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len)
                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)
        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])
    return (advantages_list, returns_list)

def vanilla_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lambd: float):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype
    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)
    full_advantages = torch.stack(adv_rev[::-1], dim=1)
    full_returns = full_advantages + values
    return (full_advantages, full_returns)

def chunked_gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lambd: 'float | torch.Tensor', chunk_size: int=128):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lambd (float | Tensor): GAE lambda; a scalar, or a [B] tensor for
                          per-sample lambda (length-adaptive GAE).
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)
    device = rewards.device
    dtype = rewards.dtype
    rowwise_lambda = isinstance(lambd, torch.Tensor)
    if rowwise_lambda:
        assert lambd.shape == (B,), f'per-sample lambd must be [B], got {lambd.shape}'
        lambd = lambd.to(device=device, dtype=dtype)
    next_values = torch.cat([values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)], dim=1)
    deltas = rewards + gamma * next_values - values
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])
    if T % chunk_size != 0:
        pad = chunk_size - T % chunk_size
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0
    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size
    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row
    mask = diff >= 0
    if rowwise_lambda:
        diff_f = diff.clamp(min=0).to(dtype)
        M = torch.where(mask, w[:, None, None] ** diff_f, torch.zeros((), device=device, dtype=dtype))
        pow_vec = w[:, None] ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)
    else:
        M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)
        if w == 0.0:
            M[mask & (diff == 0)] = 1.0
        else:
            M[mask] = w ** diff[mask].to(dtype)
        if w == 0.0:
            pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
        else:
            pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)
    if rowwise_lambda:
        S_local_chunks = torch.einsum('bnc,bcd->bnd', deltas_chunks, M)
    else:
        deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
        S_local_flat = deltas_flat @ M
        S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)
    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc
        S_local = S_local_chunks[:, c, :Lc]
        if rowwise_lambda:
            S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:, :Lc]
        else:
            S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]
        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]
    if pad > 0:
        S_rev = S_rev[:, :T]
    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values
    return (advantages, returns)
''',
'miles/utils/distributed_utils.py': r'''
def distributed_masked_whiten(values: torch.Tensor, mask: torch.Tensor, process_group: dist.ProcessGroup | None=None, shift_mean: bool=True, epsilon: float=1e-08):
    """
    Performs whitening on a tensor using global statistics from all participating GPUs.

    It calculates the global mean and variance across all ranks in the default
    process group (the WORLD) and uses these global statistics to normalize the
    local data on each rank.

    Args:
        values (torch.Tensor): The local tensor of values to whiten.
        mask (torch.Tensor): The local mask corresponding to the values.
        process_group: The process group for all_reduce.
                      If None, uses the default world group.
        shift_mean (bool): If True, the output is zero-mean. Defaults to True.
        epsilon (float): A small value for numerical stability.

    Returns:
        torch.Tensor: The locally whitened tensor using global statistics.
    """
    local_sum = (values * mask).sum()
    local_sum_sq = (values ** 2 * mask).sum()
    local_mask_sum = mask.sum()
    stats_tensor = torch.tensor([local_sum, local_sum_sq, local_mask_sum], device=values.device, dtype=torch.float32)
    dist.all_reduce(stats_tensor, group=process_group)
    global_sum, global_sum_sq, global_mask_sum = stats_tensor
    if global_mask_sum.item() == 0:
        raise ValueError('The global mask sum across all participating GPUs is zero.')
    global_mean = global_sum / global_mask_sum
    global_mean_sq = global_sum_sq / global_mask_sum
    global_var = global_mean_sq - global_mean ** 2
    if global_mask_sum.item() >= 2:
        bessel_correction = global_mask_sum / (global_mask_sum - 1)
        global_var = global_var * bessel_correction
    whitened_values = (values - global_mean) * torch.rsqrt(global_var + epsilon)
    if not shift_mean:
        whitened_values += global_mean
    return whitened_values
''',
'miles/backends/training_utils/loss.py': r'''
def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    parallel_state = get_parallel_state()
    log_probs: list[torch.Tensor] = rollout_data.get('rollout_log_probs' if args.use_rollout_logprobs else 'log_probs')
    ref_log_probs: list[torch.Tensor] = rollout_data.get('ref_log_probs')
    rewards: list[float] = rollout_data.get('rewards')
    values: None | list[torch.Tensor] = rollout_data.get('values')
    response_lengths: list[int] = rollout_data.get('response_lengths')
    loss_masks: list[torch.Tensor] = rollout_data.get('loss_masks')
    total_lengths: list[int] = rollout_data.get('total_lengths')
    max_seq_lens: list[int] | None = rollout_data.get('max_seq_lens', None)
    if log_probs is None and values is None:
        return
    if args.kl_coef == 0 or not log_probs:
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [compute_approx_kl(log_probs[i], ref_log_probs[i], kl_loss_type=args.kl_loss_type) for i in range(len(log_probs))]
    if args.advantage_estimator in ['grpo', 'gspo']:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        advantages = [r for r in returns]
    elif args.advantage_estimator == 'ppo':
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = parallel_state.cp.rank
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0 and k.numel() > 0:
                k[-1] += reward
            rewards.append(k)
        _diag_logger.info('[adv-diag] rank=%s enter GAE B=%s max_resp=%s', dist.get_rank(), len(response_lengths), max(response_lengths) if response_lengths else 0)
        advantages, returns = get_advantages_and_returns_batch(total_lengths, response_lengths, values, rewards, args.gamma, args.lambd, length_adaptive_lambda_k=args.gae_lambda_k)
        _diag_logger.info('[adv-diag] rank=%s GAE done', dist.get_rank())
    elif args.advantage_estimator == 'reinforce_plus_plus':
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(rewards=rewards, kl=kl, loss_masks=loss_masks, response_lengths=response_lengths, total_lengths=total_lengths, kl_coef=args.kl_coef, gamma=args.gamma)
        advantages = [r for r in returns]
    elif args.advantage_estimator == 'reinforce_plus_plus_baseline':
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(rewards=rewards, kl=kl, loss_masks=loss_masks, kl_coef=args.kl_coef)
        returns = advantages
    elif args.advantage_estimator == 'on_policy_distillation':
        student_log_probs = log_probs
        teacher_log_probs = rollout_data.get('teacher_log_probs')
        response_lengths = rollout_data.get('response_lengths')
        device = student_log_probs[0].device
        teacher_log_probs = [t_log_prob.to(device=device) for t_log_prob in teacher_log_probs]
        teacher_log_probs = [t_log_prob[-response_length:] for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)]
        advantages = [teacher_log_prob - student_log_prob for teacher_log_prob, student_log_prob in zip(teacher_log_probs, student_log_probs, strict=False)]
        returns = advantages
    else:
        raise NotImplementedError(f'advantage_estimator {args.advantage_estimator} is not supported. ')
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = parallel_state.cp.size
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len, args.qkv_format, max_seq_len)
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = (max(0, s0 - prompt_len), max(0, e0 - prompt_len))
                res_s1, res_e1 = (max(0, s1 - prompt_len), max(0, e1 - prompt_len))
                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])
                local_mask_chunk = torch.cat(local_mask_parts) if local_mask_parts else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                mask_chunks.append(local_mask_chunk)
            all_masks = torch.cat(mask_chunks)
        assert all_advs.size() == all_masks.size(), f'Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}'
        dp_group = parallel_state.intra_dp.group
        _diag_logger.info('[adv-diag] rank=%s enter whiten adv_numel=%s mask_numel=%s', dist.get_rank(), all_advs.numel(), all_masks.numel())
        whitened_advs_flat = distributed_masked_whiten(all_advs, all_masks, process_group=dp_group, shift_mean=True)
        _diag_logger.info('[adv-diag] rank=%s whiten done', dist.get_rank())
        chunk_lengths = [chunk.size(0) for chunk in advantages]
        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))
    rollout_data['advantages'] = advantages
    rollout_data['returns'] = returns
''',
}
