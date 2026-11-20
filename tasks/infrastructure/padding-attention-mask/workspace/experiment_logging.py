"""Scalar JSONL plus native offline W&B, shared by future controlled RL runs."""
from __future__ import annotations
import json,math,os
from pathlib import Path

class ExperimentLogger:
    def __init__(self,directory,config,*,name,project='autorlbench-contributions',enable_wandb=True):
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.history=(self.root/'metrics.jsonl').open('x',encoding='utf-8',buffering=1)
        self.steps=[];self.run=None;self.closed=False
        self.config=config
        if enable_wandb:
            import wandb
            os.environ['WANDB_MODE']='offline'
            os.environ['WANDB_SILENT']='true'
            self.run=wandb.init(mode='offline',project=project,name=name,
                dir=str(self.root.resolve()),config=config,
                settings=wandb.Settings(console='off',disable_code=True,disable_git=True))
            self.run.define_metric('training/global_step')
            self.run.define_metric('*',step_metric='training/global_step')
        (self.root/'logging_config.json').write_text(json.dumps(dict(config=config,
            wandb_mode='offline' if self.run else 'disabled_for_local_contract_test',
            cloud_upload=False),ensure_ascii=False,indent=2)+'\n')

    def log(self,step,metrics):
        if self.closed:raise RuntimeError('Logger is closed')
        if not isinstance(step,int) or (self.steps and step<=self.steps[-1]):
            raise ValueError('Steps must be strictly increasing integers')
        values={}
        for key,value in metrics.items():
            if hasattr(value,'item'):value=value.item()
            if not isinstance(value,(int,float,bool)):
                raise TypeError('Scalar logger received non-scalar: '+key)
            if not math.isfinite(value):raise ValueError('Nonfinite scalar: '+key)
            values[key]=value
        for key in ('step', '_step', 'training/global_step'):
            if key in values and values[key] != step:
                raise ValueError('Conflicting step: ' + key)
        values['training/global_step']=step
        self.history.write(json.dumps(dict(values,_step=step),allow_nan=False,sort_keys=True)+'\n')
        self.history.flush()
        if self.run:self.run.log(values,step=step)
        self.steps.append(step)
        print(json.dumps(dict(values,step=step),allow_nan=False,sort_keys=True),flush=True)

    def close(self):
        if self.closed:return
        self.closed=True;self.history.close()
        if self.run:self.run.finish()
        receipt=dict(rows=len(self.steps),first_step=self.steps[0] if self.steps else None,
                     last_step=self.steps[-1] if self.steps else None,
                     wandb_mode='offline' if self.run else 'disabled_for_local_contract_test',
                     cloud_upload=False)
        (self.root/'logging_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
