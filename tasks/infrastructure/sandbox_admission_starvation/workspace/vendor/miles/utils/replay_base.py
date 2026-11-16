import logging
import os

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


UINT8_PAD = 255


def storage_dtype(num_experts: int | None) -> np.dtype:
    """Storage dtype for rollout routed-experts arrays.

    The dtype depends on the model, never on the ids in an individual request.
    """
    if num_experts is not None and 0 < num_experts <= UINT8_PAD:
        return np.dtype(np.uint8)
    if num_experts is not None and UINT8_PAD < num_experts <= 32768:
        return np.dtype(np.int16)
    return np.dtype(np.int32)


def compact_routed_experts(array: np.ndarray, num_experts: int | None) -> np.ndarray:
    """Validate decoded expert ids and store them without retaining the wire buffer.

    Signed input uses -1 for missing routes; uint8 input uses 255. Checking
    before casting prevents invalid ids from silently wrapping into valid ones.
    """
    dtype = storage_dtype(num_experts)
    if array.dtype.kind not in "iu":
        raise ValueError("routed experts must contain integer ids")
    sentinel = pad_value(array.dtype)
    limit = np.iinfo(np.int32).max + 1
    if num_experts is not None:
        limit = min(limit, num_experts)
    if array.size:
        minimum = int(array.min())
        maximum = int(array.max(where=array != sentinel, initial=0)) if array.dtype == np.uint8 else int(array.max())
        if minimum < -1 or maximum >= limit:
            raise ValueError(f"routed expert ids must be -1 or in [0, {limit})")
    if array.dtype == dtype:
        return array
    result = array.astype(dtype)
    if array.dtype == np.uint8 and dtype != np.uint8:
        result[array == sentinel] = -1
    return result


def pad_value(dtype) -> int:
    """Sentinel for "no routing data" in the given storage dtype."""
    is_uint8 = dtype == torch.uint8 if isinstance(dtype, torch.dtype) else np.dtype(dtype) == np.uint8
    return UINT8_PAD if is_uint8 else -1


class Replay:
    def __init__(self, stream_idx: int | None = None):
        self.stream_idx = stream_idx
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list: list[torch.Tensor] = []

    def record(self, top_indices: torch.Tensor):
        buf = torch.empty_like(top_indices, device="cpu", pin_memory=True)
        buf.copy_(top_indices)
        self.top_indices_list.append(buf)

    def pop_forward(self) -> torch.Tensor:
        if self.forward_index >= len(self.top_indices_list):
            shapes = [
                t.shape if isinstance(t, torch.Tensor) else f"non-tensor({type(t)})" for t in self.top_indices_list
            ]
            raise IndexError(
                f"pop_forward out of range: forward_index={self.forward_index}, "
                f"len(top_indices_list)={len(self.top_indices_list)}, shapes={shapes}"
            )
        top_indices = self.top_indices_list[self.forward_index]
        self.forward_index += 1
        return self._to_device(top_indices)

    def pop_backward(self) -> torch.Tensor:
        top_indices = self.top_indices_list[self.backward_index]
        self.backward_index += 1
        return self._to_device(top_indices)

    @staticmethod
    def _to_device(top_indices: torch.Tensor) -> torch.Tensor:
        """Widen compact routes on the GPU and restore the missing-route sentinel."""
        out = top_indices.to(device=torch.cuda.current_device())
        if out.dtype in (torch.uint8, torch.int16):
            out = out.long()
        if top_indices.dtype == torch.uint8:
            out[out == UINT8_PAD] = -1
        return out

    def clear(self):
        self.forward_index = 0
        self.backward_index = 0
        self.top_indices_list = []

    def clear_forward(self):
        self.forward_index = 0


class BaseReplayManager:
    name: str = ""
    filename: str = ""
    replay_check_min_overlap_ratio = 0.0  # 0.0 = mismatch only on zero overlap
    # True when replayed indices are token/KV positions (indexer): rebase the
    # per-sample 0-based rollout indices onto the packed training sequence.
    replay_indices_are_token_positions = False

    def __init__(self):
        self.replays: list[Replay] = []
        self.current: Replay | None = None
        self.enabled = False
        self.stage = "fallthrough"
        self.register_replay_list_func = None

    def create_replay(self, stream_idx: int | None = None) -> Replay:
        replay = Replay(stream_idx=stream_idx)
        self.replays.append(replay)
        return replay

    def set_current(self, replay: Replay):
        self.current = replay

    def get_current(self) -> Replay | None:
        return self.current

    def reset_replays(self):
        """Drop queues registered by an earlier model build."""
        self.replays = []
        self.current = None

    def clear_all(self):
        for replay in self.replays:
            replay.clear()

    def clear_all_forward(self):
        for replay in self.replays:
            replay.clear_forward()

    def replay_topk(self, replay, old_topk_fn, scores, topk, *args, return_probs, **kwargs):
        """Run one top-k call under the manager's current stage.

        Shared by both wiring strategies: the legacy source patch reaches this
        through `get_topk_fn`, while Megatron's native `RouterReplay` reaches it
        through a per-instance binding. `replay` is the queue owned by the MoE
        layer being routed; it is unused while falling through.
        """
        if not self.enabled:
            return old_topk_fn(scores, topk, *args, **kwargs)

        if self.stage == "record":
            result = old_topk_fn(scores, topk, *args, **kwargs)
            top_indices = result[1] if return_probs else result
            replay.record(top_indices)
            return result

        if self.stage == "replay_forward":
            top_indices = replay.pop_forward()
        elif self.stage == "replay_backward":
            top_indices = replay.pop_backward()
        else:
            return old_topk_fn(scores, topk, *args, **kwargs)

        return self._build_replay_result(
            top_indices, old_topk_fn, scores, topk, *args, return_probs=return_probs, **kwargs
        )

    def _build_replay_result(self, top_indices, old_topk_fn, scores, topk, *args, return_probs, **kwargs):
        assert (
            top_indices.shape[0] == scores.shape[0]
        ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {scores.shape[0]}"

        assert (
            top_indices.shape[1] == topk
        ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"

        if self.enable_check_replay_result:
            self.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)

        # fill padding tokens with arange to avoid invalid reading
        all_invalid = (top_indices == -1).all(dim=-1)
        if all_invalid.any():
            ar = (
                torch.arange(top_indices.shape[1], device=top_indices.device, dtype=top_indices.dtype)
                % scores.shape[1]
            )
            top_indices = torch.where(all_invalid.unsqueeze(-1), ar, top_indices)

        if return_probs:
            return scores.gather(1, top_indices), top_indices
        else:
            return top_indices

    def get_topk_fn(self, old_topk_fn, return_probs):
        manager = self

        def _get_replay_result(top_indices, scores, topk, *args, **kwargs):
            assert (
                top_indices.shape[0] == scores.shape[0]
            ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {scores.shape[0]}"

            assert (
                top_indices.shape[1] == topk
            ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"

            if self.enable_check_replay_result:
                self.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)

            # fill padding tokens with arange to avoid invalid reading
            all_invalid = (top_indices == -1).all(dim=-1)
            if all_invalid.any():
                ar = (
                    torch.arange(top_indices.shape[1], device=top_indices.device, dtype=top_indices.dtype)
                    % scores.shape[1]
                )
                top_indices = torch.where(all_invalid.unsqueeze(-1), ar, top_indices)

            if return_probs:
                return scores.gather(1, top_indices), top_indices
            else:
                return top_indices

        def new_topk_fn(scores, topk, *args, **kwargs):
            if not manager.enabled:
                return old_topk_fn(scores, topk, *args, **kwargs)

            stage = manager.stage
            replay = manager.get_current()

            if stage == "fallthrough":
                return old_topk_fn(scores, topk, *args, **kwargs)

            elif stage == "record":
                result = old_topk_fn(scores, topk, *args, **kwargs)
                if return_probs:
                    probs, top_indices = result
                else:
                    top_indices = result
                replay.record(top_indices)
                return result

            elif stage == "replay_forward":
                return _get_replay_result(replay.pop_forward(), scores, topk, *args, **kwargs)

            elif stage == "replay_backward":
                return _get_replay_result(replay.pop_backward(), scores, topk, *args, **kwargs)

            else:
                return old_topk_fn(scores, topk, *args, **kwargs)

        return new_topk_fn

    def register_to_module(self, module, attr_name: str, stream_idx: int | None = None):
        if not self.enabled:
            return
        replay = self.create_replay(stream_idx=stream_idx)
        setattr(module, attr_name, replay)
        manager = self

        def pre_forward_hook(*args, **kwargs):
            manager.set_current(replay)

        module.register_forward_pre_hook(pre_forward_hook)

    def check_replay_result(self, old_topk_fn, scores, topk, top_indices, *args, **kwargs):
        """
        CI checker for R3. Only enable when enable_check_replay_result=True.
        Per token, measure the overlap between the training engine's recomputed
        topk and the replayed topk (ignoring -1 padding). A token is mismatched
        when overlap < replay_check_min_overlap_ratio of its valid picks (ratio 0.0 =
        only zero overlap). Raise when the mismatched fraction exceeds
        replay_check_max_mismatch_fraction.
        """
        orig_top_indices = old_topk_fn(scores, topk, *args, **kwargs)
        if isinstance(orig_top_indices, tuple):
            _, orig_top_indices = orig_top_indices

        orig_flat = orig_top_indices.view(-1, orig_top_indices.shape[-1])  # [n_tokens, topk]
        replay_flat = top_indices.view(-1, top_indices.shape[-1])
        valid_orig = orig_flat != -1
        valid_replay = replay_flat != -1

        # token-wise set overlap via a membership mask, avoiding the O(n*topk^2)
        # all-pairs tensor; -1 padding is routed to a sentinel column
        n_kv = int(torch.maximum(orig_flat.max(), replay_flat.max()).clamp_min(0)) + 1
        membership = orig_flat.new_zeros((orig_flat.shape[0], n_kv + 1), dtype=torch.bool)
        membership.scatter_(1, torch.where(valid_orig, orig_flat, n_kv).long(), True)
        replay_idx = torch.where(valid_replay, replay_flat, n_kv).long()
        hit = membership.gather(1, replay_idx) & valid_replay  # [n_tokens, topk]

        overlap = hit.sum(dim=1)
        valid_count = valid_replay.sum(dim=1)
        is_padding = valid_count == 0
        if self.replay_check_min_overlap_ratio == 0.0:
            required = torch.ones_like(overlap)  # legacy: mismatch only on zero overlap
        else:
            required = (valid_count * self.replay_check_min_overlap_ratio).ceil().to(overlap.dtype)
        is_mismatch = (overlap < required) & ~is_padding

        mismatch_count = is_mismatch.sum().item()
        if mismatch_count == 0:
            return

        threshold = float(os.environ.get("MILES_TEST_R3_THRESHOLD", self.replay_check_max_mismatch_fraction))
        mismatch_threshold = threshold * orig_flat.shape[0]
        mismatch_indices = is_mismatch.nonzero(as_tuple=False).squeeze(1)
        for idx in mismatch_indices[:10]:
            i = idx.item()
            lines = []
            for j in range(max(0, i - 3), min(len(orig_flat), i + 4)):
                marker = " <<<" if j == i else ""
                lines.append(f"  token {j}: orig={orig_flat[j].tolist()}, replay={replay_flat[j].tolist()}{marker}")
            logger.warning(
                f"Replay check (rank {_get_rank()}, stage {self.stage}): "
                f"token {i} overlap {overlap[i].item()}/{valid_count[i].item()}, topk={topk}\n" + "\n".join(lines)
            )

        if mismatch_count > mismatch_threshold:
            raise AssertionError(f"R3 mismatch tokens ({mismatch_count}) > threshold ({mismatch_threshold:.0f})")


class RoutingReplayManager(BaseReplayManager):
    name = "routing"
    filename = "routing_replay.pt"
    data_key = "rollout_routed_experts"
    if_sp_region = True
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2


class IndexerReplayManager(BaseReplayManager):
    name = "indexer"
    filename = "indexer_replay.pt"
    data_key = "rollout_indexer_topk"
    if_sp_region = False
    enable_check_replay_result = False
    replay_check_max_mismatch_fraction = 1e-2
    replay_check_min_overlap_ratio = 0.8
    replay_indices_are_token_positions = True


routing_replay_manager = RoutingReplayManager()
indexer_replay_manager = IndexerReplayManager()
all_replay_managers = [routing_replay_manager, indexer_replay_manager]
