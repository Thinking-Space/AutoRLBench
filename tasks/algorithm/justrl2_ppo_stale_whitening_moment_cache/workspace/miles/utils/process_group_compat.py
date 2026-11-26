"""Keep cached Megatron collective entry points aligned with native-group unwrap."""
import functools


def install():
    import torch.distributed as dist
    import megatron.core.distributed.param_and_grad_buffer as buffers
    from miles.utils.reloadable_process_group import ReloadableProcessGroup

    def unwrap(value):
        if isinstance(value, ReloadableProcessGroup):
            if value.group is None:
                raise ValueError("Collective called on an unloaded process group")
            return value.group
        return value

    for name in ("all_gather_single", "reduce_scatter_single"):
        original = getattr(dist, name, None)
        if original is None or getattr(original, "_miles_native_group_forwarder", False):
            continue

        def wrap(fn):
            @functools.wraps(fn)
            def forwarded(*args, **kwargs):
                return fn(*(unwrap(v) for v in args), **{k: unwrap(v) for k, v in kwargs.items()})

            forwarded._miles_native_group_forwarder = True
            return forwarded

        setattr(dist, name, wrap(original))

    for dst, src in (
        ("_coalescing_manager", "_coalescing_manager"),
        ("dist_all_gather_func", "all_gather_into_tensor"),
        ("dist_reduce_scatter_func", "reduce_scatter_tensor"),
    ):
        setattr(buffers, dst, getattr(dist, src))
