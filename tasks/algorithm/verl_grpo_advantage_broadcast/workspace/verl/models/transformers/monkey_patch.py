# Copyright 2024 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runtime patches applied to HuggingFace causal LM modules.

The FSDP engine's remove-padding path packs the micro-batch into a single
 ``(1, total_nnz)`` stream and deliberately passes ``attention_mask=None``
(see ``prepare_model_inputs``: "only pass input_ids and position_ids to
enable flash_attn_varlen"). A stock HuggingFace causal attention treats that
stream as ONE sequence, so whenever the stream actually concatenates several
samples, attention would leak across sample boundaries and corrupt the
log-probs/gradients. The full verl distribution solves this by patching the
attention cores to a varlen flash kernel.

This snapshot runs ``attn_implementation=sdpa`` with
``ppo_micro_batch_size_per_gpu=1`` and ``use_dynamic_bsz=False``, so each
forward sees exactly one packed sequence and the stock causal forward is
already exact. To keep the engine correct for any packed batch size, the
patch installed here wraps the model's ``forward``: single-sequence streams
are delegated to the stock forward untouched, while multi-sequence streams
are split at their ``position_ids`` resets and run per sequence, which is
numerically identical to a per-sample padded batch.
"""

import functools
import logging

import torch

logger = logging.getLogger(__name__)

# Forward kwargs that carry no per-sequence state, so they are safe to pass
# through unchanged when splitting a packed stream. Anything else that is
# non-default makes the wrapper bail out to the stock forward.
_PASSTHROUGH_DEFAULTS = {
    "past_key_values": None,
    "inputs_embeds": None,
    "labels": None,
    "cache_position": None,
    "logits_to_keep": 0,
}

_PATCHED_CLASSES = set()


def apply_monkey_patch(
    model,
    use_remove_padding: bool = True,
    ulysses_sp_size: int = 1,
    use_fused_kernels: bool = False,
    fused_kernels_backend=None,
):
    """Apply runtime patches to ``model`` and return it.

    With ``use_remove_padding=False`` the engine builds proper padded batches
    with real attention masks, so no patch is needed. With sequence
    parallelism or fused kernels the packed stream must be handled by the
    dedicated kernels from the full verl distribution, which are not part of
    this snapshot.
    """
    if use_fused_kernels:
        raise NotImplementedError(
            "use_fused_kernels=True requires the fused-kernel patches from the full verl distribution; "
            "this snapshot runs the standard HF forward path."
        )
    if ulysses_sp_size > 1:
        raise NotImplementedError(
            "ulysses_sp_size > 1 requires the sequence-parallel patches from the full verl distribution; "
            "this snapshot runs tensor_model_parallel_size=1 without sequence parallelism."
        )
    if not use_remove_padding:
        return model

    _patch_packed_forward(model)
    return model


def _patch_packed_forward(model):
    """Install the packed-stream guard on ``type(model).forward`` (idempotent)."""
    model_cls = type(model)
    if model_cls in _PATCHED_CLASSES:
        return
    orig_forward = model_cls.forward
    model_cls.forward = _make_packed_forward(orig_forward)
    _PATCHED_CLASSES.add(model_cls)
    logger.info("installed packed-stream forward guard on %s", model_cls.__name__)


def _make_packed_forward(orig_forward):
    @functools.wraps(orig_forward)
    def packed_forward(self, *args, **kwargs):
        slices = _packed_sequence_slices(kwargs)
        if slices is None or args:
            # Not a packed multi-sequence stream (or positional-call form):
            # hand the call to the stock forward unchanged.
            return orig_forward(self, *args, **kwargs)

        input_ids = kwargs["input_ids"]
        logits_parts = []
        output = None
        for start, end in slices:
            output = orig_forward(
                self,
                input_ids=input_ids[:, start:end],
                attention_mask=None,
                position_ids=kwargs["position_ids"][:, start:end],
                past_key_values=None,
                inputs_embeds=None,
                use_cache=False,
                cache_position=None,
                logits_to_keep=0,
            )
            logits_parts.append(output.logits)
        # Only ``logits`` is consumed downstream (log-probs / entropy); keep the
        # remaining fields of the last per-sequence output (all trivial here).
        output.logits = torch.cat(logits_parts, dim=1)
        return output

    return packed_forward


def _packed_sequence_slices(kwargs):
    """Return ``[(start, end), ...]`` per sequence for a packed stream, else None.

    A stream is treated as packed only when it matches the engine's
    remove-padding call exactly: ``input_ids``/``position_ids`` of shape
    ``(1, L)``, no attention mask, and ``position_ids`` restarting at 0 for
    every packed sample. Anything else returns None so the call is delegated
    to the stock forward.
    """
    for key, default in _PASSTHROUGH_DEFAULTS.items():
        if kwargs.get(key, default) != default:
            return None
    if kwargs.get("use_cache") not in (None, False):
        return None
    for key, value in kwargs.items():
        if key in _PASSTHROUGH_DEFAULTS or key in ("use_cache", "input_ids", "position_ids", "attention_mask"):
            continue
        if value is not None:
            return None

    input_ids = kwargs.get("input_ids")
    position_ids = kwargs.get("position_ids")
    attention_mask = kwargs.get("attention_mask")
    if attention_mask is not None or input_ids is None or position_ids is None:
        return None
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 2 or input_ids.shape[0] != 1:
        return None
    if not isinstance(position_ids, torch.Tensor) or tuple(position_ids.shape) != tuple(input_ids.shape):
        return None
    if input_ids.dtype not in (torch.long, torch.int) or position_ids.dtype not in (torch.long, torch.int):
        return None

    positions = position_ids[0]
    starts = (positions == 0).nonzero(as_tuple=True)[0]
    if starts.numel() < 2 or int(starts[0]) != 0:
        # 0 resets -> a single sequence: the stock causal forward is exact.
        # A stream that does not begin at position 0 is not one the engine
        # produces; leave it to the stock forward.
        return None
    ends = torch.cat([starts, positions.new_tensor([positions.shape[0]])])
    return [(int(s), int(e)) for s, e in zip(starts.tolist(), ends[1:].tolist())]
