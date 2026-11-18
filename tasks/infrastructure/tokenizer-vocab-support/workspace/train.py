"""Portable entry to the recorded upstream GRPO configuration.

The issue exercise uses frozen history. An exact continuation additionally
requires the original step-725 actor/optimizer/dataloader checkpoint, which is
not distributed as a public HF model. --check never imports CUDA or trains.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--from-base',action='store_true',help='A new experiment, not the recorded continuation')
    args,overrides=parser.parse_known_args()
    root=Path(__file__).resolve().parent
    # The packaged file lives at workspace/train.py.
    config_path=root/'configs'/'resolved_config.yaml'
    if not config_path.is_file():
        raise FileNotFoundError('Run the packaged workspace/train.py, or build the task first')
    import yaml
    config=yaml.safe_load(config_path.read_text())
    assets={'model':root/'models/actor','train':root/'data/train.parquet',
            'validation':root/'data/AIME24.parquet','framework':root/'framework/verl'}
    status={name:{'path':str(path),'exists':path.exists()} for name,path in assets.items()}
    status['historical_resume']={'supplied':args.resume is not None,
        'exists':bool(args.resume and args.resume.is_dir()),'distributed':False}
    if args.check:
        print(json.dumps({'mode':'read_only_preflight','assets':status,
            'frozen_history_steps':[726,1000],'automatic_training':False},indent=2))
        return
    if bool(args.resume)==bool(args.from_base):
        parser.error('Choose --resume <original-step-725-checkpoint> or --from-base for a separately labelled experiment')
    for name in assets:
        if not assets[name].exists():raise FileNotFoundError(name+': run environment/prepare.sh first')
    if args.resume and not args.resume.is_dir():raise FileNotFoundError(args.resume)
    os.environ['WANDB_MODE']='offline'
    sys.path.insert(0,str(root/'framework'))
    config['actor_rollout_ref']['model']['path']=str(assets['model'])
    config['data']['train_files']=[str(assets['train'])]
    config['data']['val_files']=[str(root/'data'/name) for name in ['AIME24.parquet','AIME25.parquet','AMC23.parquet']]
    config['reward']['custom_reward_function']['path']=str(root/'reward.py')
    trainer=config['trainer'];trainer['logger']=['console','wandb']
    trainer['project_name']='autorlbench-contributions';trainer['experiment_name']='recorded-grpo'
    trainer['default_local_dir']=str(root/'outputs/checkpoints')
    trainer['validation_data_dir']=str(root/'outputs/validation')
    trainer['resume_mode']='resume_path' if args.resume else 'disable'
    trainer['resume_from_path']=str(args.resume.resolve()) if args.resume else None
    runtime=config['ray_kwargs']['ray_init'];runtime['address']=None
    runtime['runtime_env'].pop('worker_process_setup_hook',None)
    runtime['runtime_env'].pop('py_executable',None)
    # codex: retain prepared Python overlays in Ray workers.
    inherited_pythonpath=os.environ.get('PYTHONPATH','')
    runtime['runtime_env'].setdefault('env_vars',{})['PYTHONPATH']=os.pathsep.join(filter(None,[str(root/'framework'),inherited_pythonpath]))
    prepared=root/'outputs/portable_config';prepared.mkdir(parents=True,exist_ok=True)
    (prepared/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    sys.argv=['verl.trainer.main_ppo','--config-path',str(prepared),'--config-name','config',*overrides]
    from verl.trainer.main_ppo import main as ppo_main
    ppo_main()


if __name__=='__main__':main()
