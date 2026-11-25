"""Detached boundary observation; PPO alignment uses the production response extraction."""
import json
import os
from pathlib import Path
import threading
_state=threading.local()

def prepare(data,model_inputs,output_args,sp):
    if not os.getenv('BENCH_AUDIT_DIR'): return
    import torch
    from verl.workers.utils.padding import no_padding_2_padding
    ids=data['input_ids']
    assert ids.is_nested
    total=ids.values().numel()
    expected=output_args['input_ids_rmpad_rolled'].flatten()
    # Response-field layouts differ from full-token input_ids. Use exactly the
    # production no_padding_2_padding mapping, not an SFT-style rolled loss mask.
    indices=no_padding_2_padding(torch.arange(total,device=expected.device),data)
    response=data['response_mask']
    if response.is_nested: response=torch.nested.to_padded_tensor(response,0)
    assert indices.shape==response.shape,(indices.shape,response.shape)
    active=torch.zeros(total,device=expected.device,dtype=torch.bool)
    active[indices[response.bool()].long()]=True
    advantage=data.get('advantages')
    nonzero=None
    if advantage is not None:
        if advantage.is_nested: advantage=torch.nested.to_padded_tensor(advantage,0)
        assert advantage.shape==response.shape
        nonzero=torch.zeros_like(active)
        nonzero[indices[response.bool() & advantage.ne(0)].long()]=True
    _state.expected=expected.detach()
    _state.active=active
    _state.nonzero=nonzero
    _state.sp=sp
    _state.record=None

def head(hidden_states,input_ids):
    directory=os.getenv('BENCH_AUDIT_DIR')
    expected=getattr(_state,'expected',None)
    if not directory or expected is None:return
    import torch
    from verl.utils.ulysses import get_ulysses_sequence_parallel_rank
    # Capped per process, with additional records whenever loss gradients are observed.
    count=getattr(_state,'count',0)
    if count>=2048:return
    _state.count=count+1
    rank=get_ulysses_sequence_parallel_rank()
    n=hidden_states.numel()//hidden_states.shape[-1]
    provided=input_ids.flatten()
    # The production fused loop consumes only the first T labels. Before-fix
    # VLM adapters can pass full global IDs while hidden states are SP-local.
    used=provided[:n]
    assert used.numel()==n and expected.numel()==n,(provided.shape,expected.shape,n)
    mismatch=used!=expected
    def local(t):
        if t is None:return None
        t=torch.nn.functional.pad(t,(0,n*_state.sp-t.numel()))
        return t[rank*n:(rank+1)*n]
    active=local(_state.active);nonzero=local(_state.nonzero)
    positions=mismatch.nonzero().flatten()
    row={'pid':os.getpid(),'rank':rank,'ordinal':count,'hidden_tokens':n,'provided_labels':provided.numel(),
         'mismatched_labels':int(mismatch.sum()),'mismatches_in_response_loss':int((mismatch&active).sum()),
         'mismatches_nonzero_advantage':None if nonzero is None else int((mismatch&active&nonzero).sum()),
         'local_mismatch_positions':positions.cpu().tolist(),
         'expected_labels':expected[positions].cpu().tolist(),'actual_labels':used[positions].cpu().tolist(),
         'grad_enabled':torch.is_grad_enabled(),'alignment':'production no_padding_2_padding'}
    _state.record=(row,mismatch.detach(),active.detach())
    emit(row,'labels')

def watch(log_probs):
    record=getattr(_state,'record',None)
    if record is None or not log_probs.requires_grad:return
    row,mismatch,active=record
    def hook(gradient):
        flat=gradient.flatten().detach()
        assert flat.numel()==mismatch.numel()
        emit({**row,'mismatch_nonzero_logprob_gradient':int((mismatch&active&flat.ne(0)).sum()),
              'mismatch_abs_logprob_gradient_sum':float(flat[mismatch&active].abs().sum())},'loss_gradient')
        # A detached observation hook returns no replacement gradient.
    log_probs.register_hook(hook)

def emit(row,kind):
    directory=Path(os.environ['BENCH_AUDIT_DIR']);directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'tensor-alignment-{os.getpid()}.jsonl').open('a') as out:
        out.write(json.dumps({'kind':kind,**row})+'\n')
