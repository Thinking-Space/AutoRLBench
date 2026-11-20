# SPDX-License-Identifier: Apache-2.0
"""Keep cached Megatron collective entry points aligned with native-group unwrap.

JustRL2 patches torch.distributed after Megatron has imported these callables.
PyTorch2.13 renamed the tensor collectives and can return no Work for a
synchronous coalesced call; routing that through a Python ProcessGroup trampoline
can create a null PyWorkHolder. Bind the cached functions to JustRL2's existing
native-group forwarding wrappers. Collective semantics are unchanged.
"""
import functools


def bind_cached_collectives(buffer_module, distributed):
    bindings = {
        '_coalescing_manager': '_coalescing_manager',
        'dist_all_gather_func': 'all_gather_into_tensor',
        'dist_reduce_scatter_func': 'reduce_scatter_tensor',
    }
    for destination, source in bindings.items():
        if not callable(getattr(buffer_module, destination, None)) or not callable(getattr(distributed, source, None)):
            raise ValueError('Unsupported Megatron/distributed collective API: ' + destination)
    for destination, source in bindings.items():
        setattr(buffer_module, destination, getattr(distributed, source))


def install():
    import torch.distributed as dist
    import megatron.core.distributed.param_and_grad_buffer as buffers
    from miles.utils.reloadable_process_group import ReloadableProcessGroup

    def unwrap(value):
        if isinstance(value, ReloadableProcessGroup):
            if value.group is None:
                raise ValueError('Collective called on an unloaded process group')
            return value.group
        return value

    for name in ('all_gather_single', 'reduce_scatter_single'):
        original = getattr(dist, name, None)
        if original is None or getattr(original, '_justrl2_native_group_forwarder', False):
            continue
        def wrap(function):
            @functools.wraps(function)
            def forwarded(*args, **kwargs):
                return function(*(unwrap(v) for v in args), **{k:unwrap(v) for k,v in kwargs.items()})
            forwarded._justrl2_native_group_forwarder = True
            return forwarded
        setattr(dist, name, wrap(original))
    bind_cached_collectives(buffers, dist)
