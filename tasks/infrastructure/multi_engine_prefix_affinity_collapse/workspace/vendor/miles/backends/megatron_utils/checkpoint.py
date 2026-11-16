import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from miles.utils import megatron_bridge_utils

from .lora_utils import is_lora_enabled, is_lora_model, load_lora_adapter, save_lora_checkpoint

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint", "save_checkpoint_with_lora", "load_checkpoint"]


def _iter_scalar_value_head_params(ddp_model):
    """Yield ``(name, param)`` for the critic's scalar value head.

    Structural match, not a role/attribute lookup: the head is the ``[1, hidden]``
    ``output_layer.weight`` (plus its ``[1]`` bias) installed by
    ``model_provider.LinearForLastLayer``. Two earlier attempts at this predicate
    silently matched nothing —
      * ``ddp_model[0].role`` is stamped by ``model.py`` only AFTER
        ``setup_model_and_optimizer()`` returns, i.e. after this load runs, so it
        always read "actor" (698572);
      * peeling a single ``.module`` misses ``DDP(Float16Module(GPTModel))`` (699321).
    ``named_parameters()`` is wrapper-depth independent, so it survives both.
    """
    for chunk in ddp_model or []:
        for name, p in chunk.named_parameters():
            if name.endswith("output_layer.weight") and p.dim() == 2 and p.shape[0] == 1:
                yield name, p
            elif name.endswith("output_layer.bias") and p.dim() == 1 and p.shape[0] == 1:
                yield name, p


def _value_head_main_param_absmax(optimizer, ddp_model):
    """Best-effort |main param| for the value head, across optimizer variants.

    The bf16 model param and the fp32 master copy can disagree, and only the
    master one survives ``optimizer.step()`` — logging just the model param would
    print ``0`` even with a polluted master, i.e. exactly the blind signal that let
    this bug through three times. Returns ``None`` when the mapping cannot be
    resolved (fp32 optimizer, precision-aware optimizer, unexpected megatron
    version); the caller degrades to "unavailable" rather than failing the load.
    """
    if optimizer is None:
        return None
    ids = {id(p) for _, p in _iter_scalar_value_head_params(ddp_model)}
    if not ids:
        return None
    found = []
    for opt in getattr(optimizer, "chained_optimizers", None) or [optimizer]:
        # DistributedOptimizer shards the master copy (``shard_fp32_from_float16``);
        # the non-distributed MixedPrecisionOptimizer keeps whole tensors.
        model_groups = getattr(opt, "model_float16_groups", None) or getattr(opt, "float16_groups", None)
        main_groups = getattr(opt, "shard_fp32_from_float16_groups", None) or getattr(
            opt, "fp32_from_float16_groups", None
        )
        if not model_groups or not main_groups:
            continue
        # Best-effort diagnostic: tolerate group-length mismatches across Megatron versions.
        for model_group, main_group in zip(model_groups, main_groups, strict=False):
            for model_param, main_param in zip(model_group, main_group, strict=False):
                if id(model_param) in ids and main_param is not None:
                    # A DP rank may own no slice of this param under the distributed
                    # optimizer; an empty shard is vacuously clean.
                    found.append(float(main_param.detach().abs().max()) if main_param.numel() else 0.0)
    return max(found) if found else None


def _log_critic_value_head(ddp_model, args, tag, optimizer=None):
    """Diagnostic: the value head's magnitude around checkpoint load."""
    for name, p in _iter_scalar_value_head_params(ddp_model):
        if not name.endswith("weight"):
            continue
        main_absmax = _value_head_main_param_absmax(optimizer, ddp_model)
        logger.info(
            "[critic-value-head] %s: %s shape=%s w_absmax=%.6g main_absmax=%s finetune=%s",
            tag,
            name,
            tuple(p.shape),
            float(p.detach().abs().max()),
            "unavailable" if main_absmax is None else f"{main_absmax:.6g}",
            getattr(args, "finetune", None),
        )
        return


def _rezero_critic_value_head(ddp_model, optimizer, args):
    """Re-zero the critic's scalar value head after loading a POLICY checkpoint.

    Why re-zero instead of filtering the load: the policy ckpt stores
    ``output_layer.weight`` as ``[vocab, hidden]`` while the critic's head is
    ``[1, hidden]``, and megatron's dist-ckpt reader fills the overlapping region
    instead of raising — the head ends up holding row 0 of the LM head
    (699468: before-load w_absmax=0 -> after-load 0.189, i.e. V ~ +8 against
    group-centered targets in [-1, 1]). The previous attempt to drop the key from
    the request could never work: megatron's ``load_checkpoint`` calls
    ``unwrap_model(ddp_model)`` first, so patching ``sharded_state_dict`` on the
    DDP wrapper is never seen (``megatron/training/checkpointing.py:1504,903``).
    Re-zeroing afterwards is independent of which load path ran and is directly
    verifiable from the post-load log line.

    The optimizer resync is not optional: megatron's ``load_checkpoint`` ends with
    ``optimizer.reload_model_params()`` under ``--finetune``/``--no-load-optim``,
    which has ALREADY copied the polluted value into the fp32 master weights.
    Zeroing only the bf16 model param would be undone by the first
    ``optimizer.step()`` (master - lr*grad is copied back into the model param).
    ``reload_model_params`` is a plain local model->main copy, no collective, so
    calling it a second time is safe and idempotent.

    Only for ``finetune=True`` (starting from a policy/base ckpt). On resume
    (``finetune=False``) the critic loads its own ckpt, whose ``[1, hidden]`` head
    is a trained parameter that must be kept.
    """
    if not getattr(args, "finetune", False):
        return
    zeroed = []
    with torch.no_grad():
        for name, p in _iter_scalar_value_head_params(ddp_model):
            p.zero_()
            zeroed.append(name)
    if not zeroed:
        return
    if optimizer is not None and hasattr(optimizer, "reload_model_params"):
        optimizer.reload_model_params()
        resync = "master params resynced"
    else:
        # optimizer=None is the ref-model load path (actor.load_other_checkpoint);
        # there are no master weights to keep in sync.
        resync = "no optimizer, nothing to resync"
    logger.info("[critic-value-head] re-zeroed %s after policy-ckpt load (%s)", zeroed, resync)


def _resolve_megatron_iter_dir(load_dir) -> str | None:
    """Resolve the iteration directory megatron's tracker points at, or None."""
    if not load_dir:
        return None
    tracker = os.path.join(load_dir, "latest_checkpointed_iteration.txt")
    if not os.path.isfile(tracker):
        return None
    with open(tracker) as f:
        content = f.read().strip()
    name = "release" if content == "release" else f"iter_{int(content):07d}"
    iter_dir = os.path.join(load_dir, name)
    return iter_dir if os.path.isdir(iter_dir) else None


def _ckpt_contains_scalar_value_head(load_dir, ddp_model=None) -> bool:
    """Whether the dist-ckpt at ``load_dir`` already holds a critic-style head.

    A policy ckpt stores ``output_layer.weight`` as ``[vocab, hidden]`` and no
    bias; a trained critic ckpt stores ``[1, hidden]`` plus ``output_layer.bias``.
    Reads only the dist-ckpt tensor metadata (no tensor data).

    On metadata read failure the fallback depends on whether the model has a scalar
    value head:
    - No value head: the model cannot be harmed; return False safely.
    - Has value head: we cannot tell whether this is a policy or trained-critic
      checkpoint, but silently re-zeroing a trained critic is worse than a loud
      failure at startup.  Raise immediately so the operator can verify the
      checkpoint path rather than losing a trained critic silently.
    If no ddp_model is provided the old safe-fallback (return False) is preserved
    for callers that have no model context (e.g. standalone validation tools).
    """
    iter_dir = _resolve_megatron_iter_dir(load_dir)
    if iter_dir is None:
        return False
    try:
        from megatron.core import dist_checkpointing

        metadata = dist_checkpointing.load_tensors_metadata(iter_dir)
    except Exception as exc:
        model_has_value_head = ddp_model is not None and any(
            True for _ in _iter_scalar_value_head_params(ddp_model)
        )
        if model_has_value_head:
            raise RuntimeError(
                f"[critic-value-head] could not read dist-ckpt tensor metadata at {iter_dir!r} "
                f"({exc}). The model HAS a scalar value head, so proceeding with a policy-ckpt "
                "assumption risks silently re-zeroing a trained critic. "
                "Fix the checkpoint path or resolve the metadata read error before continuing."
            ) from exc
        logger.warning(
            "[critic-value-head] could not read dist-ckpt tensor metadata at %s (%s); "
            "model has no scalar value head so assuming a policy checkpoint (safe fallback).",
            iter_dir,
            exc,
        )
        return False
    for key, sharded in metadata.items():
        if key.endswith("output_layer.bias"):
            return True
        if key.endswith("output_layer.weight") and tuple(getattr(sharded, "global_shape", ()))[:1] == (1,):
            return True
    return False


@contextmanager
def _critic_value_head_load_guard(ddp_model, args):
    """Drop the critic's scalar value head from the load request (finetune only).

    A policy checkpoint has no ``output_layer.bias`` at all, and its
    ``output_layer.weight`` is ``[vocab, hidden]`` against the critic's
    ``[1, hidden]``. Megatron marks the output layer ``allow_shape_mismatch``, and
    the pinned dist-ckpt reader then looks up ``tensor_metadata[key]`` for every
    such tensor -- the absent bias raises ``KeyError: 'output_layer.bias'`` before
    a single weight is read. (Older Megatron tolerated the missing key, which is
    why the same checkpoint loads fine on the old stack.)

    The value head is re-zeroed immediately after the load regardless, so nothing
    is lost by never reading it: this removes the pollution at the source instead
    of repairing it afterwards. ``_rezero_critic_value_head`` stays as the second
    layer of defense (and still does the fp32 master resync).

    Patching on the object megatron actually calls: ``load_checkpoint`` runs
    ``unwrap_model(ddp_model)`` first, so the DDP/Float16Module wrappers are gone
    by then -- a patch installed on the wrapper would never be seen (699468).
    """
    if not getattr(args, "finetune", False):
        yield
        return

    if os.environ.get("VALIDATE_SKIP_LOAD_GUARD") == "1":
        # Negative control for tools/validate_critic_value_head.py: without the
        # guard the pinned Megatron raises KeyError on the absent
        # 'output_layer.bias', which is exactly what this context manager exists
        # to prevent.
        logger.warning("[critic-value-head] VALIDATE_SKIP_LOAD_GUARD=1: value head NOT excluded from the load")
        yield
        return

    from megatron.core.utils import unwrap_model

    critic_chunks = [
        chunk
        for chunk in unwrap_model(ddp_model or [])
        if any(True for _ in _iter_scalar_value_head_params([chunk]))
    ]
    if critic_chunks and _ckpt_contains_scalar_value_head(getattr(args, "load", None), ddp_model=ddp_model):
        # The checkpoint already holds a trained critic value head, but finetune=True
        # would exclude it from the load and re-zero it (silently discarding the
        # trained critic).  finetune=True is also set globally whenever the actor's
        # --load dir has no tracker file, so a warm-start from a previously trained
        # critic would hit this path too.  Refuse loudly in all cases.
        raise RuntimeError(
            f"[critic-value-head] the checkpoint at {args.load!r} already contains a "
            "scalar value head (output_layer.bias / [1, hidden] output_layer.weight): "
            "this is a TRAINED CRITIC checkpoint, but finetune=True would exclude that "
            "head from the load and re-zero it, silently discarding the trained critic. "
            "To keep the trained head, load it with resume semantics (finetune=False). "
            "To re-initialize the critic from a base model, point --critic-load at a "
            "policy checkpoint instead."
        )

    patched = []
    for chunk in critic_chunks:
        original = chunk.sharded_state_dict

        def _make(original_fn):
            def _sharded_state_dict_without_value_head(*a, **kw):
                sharded = original_fn(*a, **kw)
                for key in [k for k in sharded if k.endswith(("output_layer.weight", "output_layer.bias"))]:
                    sharded.pop(key)
                return sharded

            return _sharded_state_dict_without_value_head

        chunk.sharded_state_dict = _make(original)
        patched.append((chunk, original))

    if patched:
        logger.info("[critic-value-head] excluding the scalar value head from the checkpoint load request")
    try:
        yield
    finally:
        for chunk, original in patched:
            chunk.sharded_state_dict = original


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()

    load_path = args.load

    has_local_checkpoint_manager = "local_checkpoint_manager" in (checkpointing_context or {})
    if has_local_checkpoint_manager:
        logger.info("Skipping disk path validation: using in-memory checkpoint via local_checkpoint_manager")
    else:
        assert Path(load_path).exists() and _is_dir_nonempty(
            load_path
        ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    _log_critic_value_head(ddp_model, args, "before-load", optimizer)

    with _critic_value_head_load_guard(ddp_model, args):
        if has_local_checkpoint_manager or _is_megatron_checkpoint(load_path):
            result = _load_checkpoint_megatron(
                ddp_model=ddp_model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            )
        else:
            result = _load_checkpoint_hf(
                ddp_model=ddp_model,
                optimizer=optimizer,
                args=args,
                load_path=load_path,
            )

    # A critic starting from a policy checkpoint silently inherits row 0 of the
    # [vocab, hidden] LM head into its [1, hidden] value head; re-zero it here,
    # after whichever load path ran. The three log lines are the only direct
    # evidence this worked -- three earlier "fixes" were no-ops that nobody caught
    # because they were never logged (698572 / 699321 / 699468).
    _log_critic_value_head(ddp_model, args, "after-load", optimizer)
    _rezero_critic_value_head(ddp_model, optimizer, args)
    _log_critic_value_head(ddp_model, args, "after-rezero", optimizer)

    # Load LoRA adapter weights if available
    if is_lora_enabled(args):
        adapter_path = getattr(args, "lora_adapter_path", None)
        if adapter_path is not None:
            loaded, iteration = load_lora_adapter(
                ddp_model,
                adapter_path,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
            )
            if loaded:
                logger.info(f"Successfully loaded LoRA adapter from {adapter_path}")
                if iteration is not None:
                    result = (iteration, result[1])
            else:
                logger.warning(
                    f"LoRA is enabled and --lora-adapter-path={adapter_path} was specified, "
                    f"but adapter weights could not be loaded. "
                    f"Training will start with freshly initialized adapter weights."
                )

    return result


def save_checkpoint_with_lora(iteration, model, optimizer, opt_param_scheduler):
    """Extended save that handles LoRA adapters separately."""
    args = get_args()

    if is_lora_model(model):
        save_dir = Path(args.save) / f"iter_{iteration:07d}" / "adapter"
        logger.info(f"Saving LoRA checkpoint to {save_dir}")
        save_lora_checkpoint(
            model,
            args,
            str(save_dir),
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
        )
    else:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
