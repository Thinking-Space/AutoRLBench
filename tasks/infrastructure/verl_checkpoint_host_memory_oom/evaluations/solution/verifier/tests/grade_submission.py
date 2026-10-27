"""Fresh training, independent checkpoint inspection, and a separate resume run."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

from omegaconf import OmegaConf
from check_run import check_checkpoint, check_updates, rows


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    verifier = Path(os.environ['AUTORLBENCH_VERIFIER'])
    workspace = Path(os.environ['AUTORLBENCH_WORKSPACE']).resolve()
    results = Path(os.environ['AUTORLBENCH_RESULTS']).resolve()
    spec = json.loads((verifier/'resources/grader_spec.json').read_text())
    report = {'passed':False,'score':0.0,'metrics':{}}
    try:
        assert int(Path('/sys/fs/cgroup/memory.max').read_text()) == spec['memory_limit_bytes']
        for rel,digest in spec['protected_sha256'].items():
            file = workspace/rel
            assert file.is_file() and not file.is_symlink() and sha(file)==digest, 'protected evidence or data changed: '+rel
        for asset in spec['model_assets']:
            file = workspace/'model'/asset['path']
            assert file.is_file() and not file.is_symlink() and file.stat().st_size==asset['size'] and sha(file)==asset['sha256']
        actual = OmegaConf.load(workspace/'configs/experiment.yaml')
        fixed = OmegaConf.create(spec['fixed_config'])
        for config in [actual,fixed]:
            config.actor_rollout_ref.actor.fsdp_config.param_offload = None
            config.actor_rollout_ref.actor.fsdp_config.optimizer_offload = None
        assert OmegaConf.to_container(actual)==OmegaConf.to_container(fixed), 'fixed training/checkpoint recipe changed'
        env = dict(os.environ, PYTHONPATH=str(workspace), PYTHONDONTWRITEBYTECODE='1')
        def execute(name, resume=None):
            output = results/(name+'-'+uuid.uuid4().hex)
            report['metrics'][name+'_dir'] = str(output)
            argv = [sys.executable,'launch.py','--output',str(output)]
            if resume is not None:
                argv += ['--resume',str(resume),'--validate-only']
            with (results/(name+'.stdout.log')).open('w') as stdout, (results/(name+'.stderr.log')).open('w') as stderr:
                process = subprocess.run(argv,cwd=workspace,env=env,stdout=stdout,stderr=stderr,timeout=1200)
            report['metrics'][name+'_exit_code'] = process.returncode
            assert process.returncode==0, name+' failed; inspect runtime logs'
            return output
        run = execute('training')
        updates,events = check_updates(run,1,spec['final_step'],spec['memory_limit_bytes'])
        assert [e['step'] for e in events if e['phase']=='save_complete']==[spec['checkpoint_step']]
        change = max(r['actor/parameter_probe_change'] for r in updates)
        assert change >= spec['minimum_parameter_change'], 'no actual actor parameter change'
        checkpoints=[]
        for step in [spec['checkpoint_step']]:
            item,_ = check_checkpoint(run,step,workspace/'model')
            checkpoints.append(item)
        resumed = execute('resume',run/'checkpoints'/f"global_step_{spec['checkpoint_step']}")
        resumed_events = rows(resumed,'checkpoint_events')
        loads = [e for e in resumed_events if e['phase']=='load_complete']
        assert len(loads)==1 and loads[0]['step']==spec['checkpoint_step']
        assert loads[0]['optimizer_step_min']==loads[0]['optimizer_step_max']==spec['checkpoint_step']
        assert all(e['host_memory_limit_bytes']==spec['memory_limit_bytes'] for e in resumed_events)
        validation = rows(resumed,'metrics')
        assert len(validation)==1 and validation[0]['step']==spec['checkpoint_step']
        assert 'val/test_score/openai/gsm8k' in validation[0], 'missing restored-model validation'
        assert not (resumed/'worker_events.jsonl').exists(), 'resume check must not perform extra training steps'
        completed = json.loads((resumed/'completed.json').read_text())
        assert completed['mode']=='validation' and completed['steps']==spec['checkpoint_step']
        report['metrics'].update(completed_steps=len(updates),restored_step=spec['checkpoint_step'],resumed_training_steps=0,parameter_change=change,checkpoints=checkpoints)
        report.update(passed=True,score=1.0)
    except Exception as error:
        report['metrics']['error']=str(error)
    (results/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
