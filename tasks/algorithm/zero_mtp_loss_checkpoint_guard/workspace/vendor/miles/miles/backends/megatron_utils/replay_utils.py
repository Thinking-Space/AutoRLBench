"""Wire miles' rollout routing replay (R3) into Megatron's MoE routers.

R3 needs a hook inside Megatron's top-k routing. Two Megatron generations expose
that hook differently, and miles supports both:

* Legacy (core_v0.16 plus ``docker/patch/latest/megatron.patch``): the patch
  calls ``routing_replay_manager.register_to_module`` from ``TopKRouter``, so
  the manager's replay queues are created during model build.
* Native (core_v0.18+): Megatron owns a ``RouterReplay`` per MoE layer, built
  when ``config.moe_enable_routing_replay`` is set. miles binds one replay queue
  onto each instance here instead of patching Megatron's source.

Without either wiring the manager holds no queues, and replay filling would fail
with an opaque ``IndexError``; ``register_replay_list_moe`` reports that instead.
"""

import logging

from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

logger = logging.getLogger(__name__)

try:
    from megatron.core.transformer.moe.router_replay import RouterReplay
except ImportError:  # Megatron predates native router replay; legacy patch applies.
    RouterReplay = None


def native_router_replay_supported() -> bool:
    """Whether this Megatron owns the per-layer replay instances miles binds to."""
    return RouterReplay is not None


def reset_native_router_replays() -> None:
    """Drop instances left by an earlier model build.

    Megatron tracks its ``RouterReplay`` objects in a class-level list that
    survives model teardown, so a rebuild (fault-tolerance heal, reload) would
    otherwise stack a second model's layers onto the first's.
    """
    if RouterReplay is not None:
        RouterReplay.clear_global_router_replay_instances()


def bind_native_router_replays(manager, models) -> int:
    """Give each of this rank's MoE layers a miles replay queue.

    Megatron's own implementation keeps replayed indices on the GPU and cannot
    express miles' padding and cross-check semantics, so the bound instances get
    their ``get_replay_topk`` rebound to the manager's shared stage dispatch.

    Megatron also builds a ``RouterReplay`` for MTP layers, which R3 does not
    cover. Those trail the decoder's instances in build order, so only the
    leading ones are bound; the rest keep Megatron's default pass-through.
    Returns the number of layers bound.
    """
    instances = RouterReplay.global_router_replay_instances
    num_moe_layers = len(local_moe_layer_indices(models))
    if len(instances) < num_moe_layers:
        raise RuntimeError(
            f"Megatron built {len(instances)} RouterReplay instances but this rank has {num_moe_layers} "
            f"MoE layers. Routing replay cannot be bound; check that config.moe_enable_routing_replay "
            f"was set before the model was built."
        )

    for instance in instances[:num_moe_layers]:
        replay = manager.create_replay()
        instance.get_replay_topk = _make_native_replay_topk(manager, replay)
    logger.info(
        f"Bound {num_moe_layers} of {len(instances)} native Megatron RouterReplay "
        f"instances to {manager.name} replay"
    )
    return num_moe_layers


def _make_native_replay_topk(manager, replay):
    def get_replay_topk(scores, topk, num_groups=None, group_topk=None, default_compute_topk=None):
        return manager.replay_topk(
            replay, default_compute_topk, scores, topk, num_groups, group_topk, return_probs=True
        )

    return get_replay_topk


def local_moe_layer_indices(models) -> list[int]:
    """Global layer indices of the MoE layers this rank builds, in build order."""
    layer_indices = []
    for vp_stage, model in enumerate(models):
        config = model.module.config
        num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        for layer_id in range(offset, offset + num_layers_to_build):
            if isinstance(config.moe_layer_freq, int):
                if layer_id % config.moe_layer_freq != 0:
                    continue
            elif isinstance(config.moe_layer_freq, list):
                assert len(config.moe_layer_freq) == config.num_layers
                if config.moe_layer_freq[layer_id] == 0:
                    continue
            layer_indices.append(layer_id)
    return layer_indices


def register_replay_list_moe(replay_list, replay_data, *, models, **_kwargs):
    """Map replay streams to Megatron MoE layers using the local model layout."""
    layer_indices = local_moe_layer_indices(models)
    if len(replay_list) != len(layer_indices):
        raise RuntimeError(
            f"Routing replay is enabled but {len(replay_list)} replay queues were registered for "
            f"{len(layer_indices)} local MoE layers. The replay hook did not reach Megatron's router: "
            f"on core_v0.18+ check that config.moe_enable_routing_replay is set and "
            f"bind_native_router_replays ran after the model build; on older Megatron check that "
            f"docker/patch/latest/megatron.patch was applied (setup.sh logs when it skips the patch)."
        )

    for replay_idx, layer_idx in enumerate(layer_indices):
        layer_data = replay_data[:, layer_idx]
        replay_list[replay_idx].record(layer_data)
