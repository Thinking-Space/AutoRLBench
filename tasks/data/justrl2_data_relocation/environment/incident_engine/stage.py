"""Pinned native source with independent checkpoint-load evidence."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def instrument_load(text):
    anchor='    iteration, _ = load_checkpoint(\n'
    if text.count(anchor)!=1:
        raise ValueError('Native checkpoint load anchor changed')
    text='from restore_evidence import restore_load_checkpoint\n'+text
    text=text.replace(anchor,'    iteration, _ = restore_load_checkpoint(args, load_checkpoint,\n',1)
    advance='    if opt_param_scheduler is not None:\n        opt_param_scheduler.step(increment=iteration * args.global_batch_size)\n'
    if advance in text:
        text=text.replace(advance,'    # Native load already restored the exact incident scheduler state.\n',1)
    ast.parse(text)
    return text


def stage_source(bundle, prepared, output):
    bundle,prepared,output=map(Path,(bundle,prepared,output))
    # Copies are pinned as a whole into read-only AUTORLBENCH_PREPARED at setup.
    subprocess.run([sys.executable,str(bundle/'stage_runtime.py'),
                    '--source-archive',str(prepared/'source.tar.gz'),
                    '--helpers',str(bundle/'runtime_support'),'--output',str(output)],check=True)
    root=output/'JustRL-II-75814318f243af7e39af337c1173fc3927c9635b'
    model=root/'miles/backends/megatron_utils/model.py'
    text=instrument_load(model.read_text())
    text='from restore_evidence import restore_after_optimizer_step, restore_after_save\n'+text
    anchor='                        grad_norm, rollout_id, step_id)\n'
    if text.count(anchor)!=1:raise ValueError('Optimizer evidence hook changed')
    text=text.replace(anchor,anchor+'        restore_after_optimizer_step(args, optimizer, rollout_id, step_id)\n',1)
    save_tree=ast.parse(text)
    method=next(n for n in ast.walk(save_tree) if isinstance(n,ast.FunctionDef) and n.name=='save')
    # Put readback immediately after synchronous native save, before cache/offload.
    save_text=''.join(text.splitlines(keepends=True)[method.lineno-1:method.end_lineno])
    save_anchor='    if hashes is not None:\n'
    if save_text.count(save_anchor)!=1:raise ValueError('Native save completion anchor changed')
    changed=save_text.replace(save_anchor,'    if iteration == 7:\n        restore_after_save(args, iteration)\n\n'+save_anchor,1)
    text=text.replace(save_text,changed,1)
    model.write_text(text)
    receipt_path=output/'source_receipt.json'
    receipt=json.loads(receipt_path.read_text())
    receipt['instrumented_files']['miles/backends/megatron_utils/model.py']=hashlib.sha256(model.read_bytes()).hexdigest()
    receipt['incident_restore_instrumentation']=True
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return root
