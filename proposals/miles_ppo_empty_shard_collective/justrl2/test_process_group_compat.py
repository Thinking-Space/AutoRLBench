"""Compatibility tests for cached native-group collective bindings."""
from contextlib import contextmanager
from types import SimpleNamespace
import importlib
import sys
from types import ModuleType

import pytest


def test_cached_manager_and_collective_use_same_native_group_after_reload():
    helper = importlib.import_module('process_group_compat')
    operations, calls = {}, []

    class Reloadable:
        def __init__(self, group):
            self.group = group

    @contextmanager
    def native_manager(group, async_ops=False):
        operations[group] = []
        try:
            yield
            calls.append((group, operations[group], async_ops))
        finally:
            del operations[group]

    def native_reduce(output, value, *, group, async_op=False):
        operations[group].append((output,value,async_op))

    def unwrap(fn):
        def wrapped(*args, **kw):
            args=tuple(x.group if isinstance(x,Reloadable) else x for x in args)
            kw={k:(v.group if isinstance(v,Reloadable) else v) for k,v in kw.items()}
            return fn(*args,**kw)
        return wrapped

    dist=SimpleNamespace(_coalescing_manager=unwrap(native_manager),
                         reduce_scatter_tensor=unwrap(native_reduce),
                         all_gather_into_tensor=unwrap(native_reduce))
    buffer=SimpleNamespace(_coalescing_manager=native_manager,
                           dist_reduce_scatter_func=native_reduce,
                           dist_all_gather_func=native_reduce)
    helper.bind_cached_collectives(buffer,dist)
    wrapper=Reloadable(object())
    for async_op in (False,True):
        with buffer._coalescing_manager(wrapper,async_ops=async_op):
            buffer.dist_reduce_scatter_func('out','in',group=wrapper,async_op=async_op)
        assert calls[-1]==(wrapper.group,[('out','in',async_op)],async_op)
        wrapper.group=object()
    helper.bind_cached_collectives(buffer,dist)
    assert buffer._coalescing_manager is dist._coalescing_manager


def test_unsupported_cached_api_fails_without_partial_rebinding():
    helper=importlib.import_module('process_group_compat')
    old=lambda:None
    buffer=SimpleNamespace(_coalescing_manager=old,dist_reduce_scatter_func=old,dist_all_gather_func=old)
    with pytest.raises(ValueError):
        helper.bind_cached_collectives(buffer,SimpleNamespace(_coalescing_manager=lambda:None))
    assert buffer._coalescing_manager is old


def test_installer_aliases_preserve_returns_reload_identity_and_idempotence(monkeypatch):
    helper=importlib.import_module('process_group_compat')
    names=('torch','torch.distributed','megatron','megatron.core','megatron.core.distributed',
           'megatron.core.distributed.param_and_grad_buffer','miles','miles.utils',
           'miles.utils.reloadable_process_group')
    modules={name:ModuleType(name) for name in names}
    for name,value in modules.items():
        monkeypatch.setitem(sys.modules,name,value)
        if '.' in name:
            parent,child=name.rsplit('.',1)
            setattr(modules[parent],child,value)
    dist=modules['torch.distributed']
    buffer=modules['megatron.core.distributed.param_and_grad_buffer']
    class Reloadable:
        def __init__(self,group):self.group=group
    modules['miles.utils.reloadable_process_group'].ReloadableProcessGroup=Reloadable
    result=object()
    calls=[]
    def native(*args,**kwargs):
        calls.append((args,kwargs));return result
    for name in ('all_gather_single','reduce_scatter_single','_coalescing_manager',
                 'all_gather_into_tensor','reduce_scatter_tensor'):
        setattr(dist,name,native)
    buffer._coalescing_manager=buffer.dist_all_gather_func=buffer.dist_reduce_scatter_func=native
    helper.install()
    first=dist.reduce_scatter_single
    group=Reloadable(object())
    assert first('out','in',group=group,async_op=True) is result
    assert calls[-1]==(('out','in'),{'group':group.group,'async_op':True})
    group.group=object()
    assert dist.all_gather_single('out','in',group,False) is result
    assert calls[-1]==(('out','in',group.group,False),{})
    helper.install()
    assert dist.reduce_scatter_single is first
    group.group=None
    with pytest.raises(ValueError,match='unloaded'):
        first('out','in',group=group)
