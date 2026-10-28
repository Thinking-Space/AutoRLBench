# SPDX-License-Identifier: Apache-2.0
# Function/block excerpts from Miles, with a small standalone adapter.
# See source_manifest.json for exact revisions, source ranges and hashes.
# These reviewer-facing excerpts intentionally contain both versions.
from argparse import Namespace
from types import SimpleNamespace

import torch
import torch.distributed as dist

_parallel_state = None


def get_parallel_state():
    return _parallel_state


def _parallel_rank(parallel_state, name: str) -> int:
    group_info = getattr(parallel_state, name, None)
    if group_info is not None and hasattr(group_info, "rank"):
        return group_info.rank

    legacy_name = f"{name}_rank"
    if hasattr(parallel_state, legacy_name):
        return getattr(parallel_state, legacy_name)

    raise AttributeError(f"ParallelState has neither {name}.rank nor {legacy_name}")


def _parallel_size(parallel_state, name: str) -> int:
    group_info = getattr(parallel_state, name, None)
    if group_info is not None and hasattr(group_info, "size"):
        return group_info.size

    legacy_name = f"{name}_size"
    if hasattr(parallel_state, legacy_name):
        return getattr(parallel_state, legacy_name)

    raise AttributeError(f"ParallelState has neither {name}.size nor {legacy_name}")


def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    *args,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
):
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
                raise TypeError("Too many positional arguments for get_logits_and_tokens_offset_with_cp")
        else:
            parallel_state = first_arg
            if len(args) > 1:
                qkv_format = args[1]
            if len(args) > 2:
                max_seq_len = args[2]
            if len(args) > 3:
                raise TypeError("Too many positional arguments for get_logits_and_tokens_offset_with_cp")

    if parallel_state is None:
        parallel_state = get_parallel_state()

    cp_rank = _parallel_rank(parallel_state, "cp")
    cp_size = _parallel_size(parallel_state, "cp")
    assert cp_size > 1

    prompt_length = total_length - response_length
    if qkv_format == "thd":
        chunk_size = (total_length + 2 * cp_size - 1) // (2 * cp_size)
    else:
        assert max_seq_len is not None, "max_seq_len must be provided for qkv_format=bshd"
        chunk_size = (max_seq_len + 2 * cp_size - 1) // (2 * cp_size)

    # the offset of 2 chunks
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)

    # the offset of 2 logits, note that the logits need a "-1".
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))

    # when the sequence is empty, make an empty slice to continue the gradient flow.
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

    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)


def distributed_masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
    shift_mean: bool = True,
    epsilon: float = 1e-8,
):
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
    # Calculate local intermediate statistics
    local_sum = (values * mask).sum()
    local_sum_sq = ((values**2) * mask).sum()
    local_mask_sum = mask.sum()

    stats_tensor = torch.tensor(
        [local_sum, local_sum_sq, local_mask_sum],
        device=values.device,
        dtype=torch.float32,
    )

    # Aggregate via all_reduce within the DP group
    dist.all_reduce(stats_tensor, group=process_group)

    # Calculate global stats from aggregated results
    global_sum, global_sum_sq, global_mask_sum = stats_tensor

    if global_mask_sum.item() == 0:
        raise ValueError("The global mask sum across all participating GPUs is zero.")

    global_mean = global_sum / global_mask_sum
    global_mean_sq = global_sum_sq / global_mask_sum
    global_var = global_mean_sq - global_mean**2

    # Bessel's correction for unbiased estimate
    if global_mask_sum.item() >= 2:
        bessel_correction = global_mask_sum / (global_mask_sum - 1)
        global_var = global_var * bessel_correction

    # Whiten local data using global stats
    whitened_values = (values - global_mean) * torch.rsqrt(global_var + epsilon)

    if not shift_mean:
        whitened_values += global_mean

    return whitened_values


def normalize_original(advantages, loss_masks, total_lengths, response_lengths, cp_rank=1, cp_size=2):
    global _parallel_state
    parallel_state = SimpleNamespace(
        cp=SimpleNamespace(rank=cp_rank, size=cp_size),
        intra_dp=SimpleNamespace(group=dist.group.WORLD),
    )
    _parallel_state = parallel_state
    args = Namespace(normalize_advantages=True, qkv_format="thd")
    max_seq_lens = None
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

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = parallel_state.intra_dp.group

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))
    return advantages


def normalize_repaired(advantages, loss_masks, total_lengths, response_lengths, cp_rank=1, cp_size=2):
    global _parallel_state
    parallel_state = SimpleNamespace(
        cp=SimpleNamespace(rank=cp_rank, size=cp_size),
        intra_dp=SimpleNamespace(group=dist.group.WORLD),
    )
    _parallel_state = parallel_state
    args = Namespace(normalize_advantages=True, qkv_format="thd")
    max_seq_lens = None
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

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        # 集合通信不变量：distributed_masked_whiten 含 DP-group allreduce，所有 rank
        # 必须无条件参与。CP 切片下本地分片可能为空，空 tensor 参与是安全的
        # （贡献 [0,0,0] 统计量）；若按本地是否为空跳过，会死锁整个 DP group。
        assert (
            all_advs.size() == all_masks.size()
        ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
        dp_group = parallel_state.intra_dp.group

        whitened_advs_flat = distributed_masked_whiten(
            all_advs,
            all_masks,
            process_group=dp_group,
            shift_mean=True,
        )
        chunk_lengths = [chunk.size(0) for chunk in advantages]
        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))
    return advantages
