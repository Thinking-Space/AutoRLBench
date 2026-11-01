"""Trusted incident continuation entry; the only submission input is loader.py.

Use a read-only prepared bundle/scene under the benchmark runtime. This module
does not invoke code, configs, launch scripts or rewards from the submitted tree
other than its location-resolver function.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scene import verify_scene
from loader_contract import expected_groups
from stage import stage_source

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def run(args):
    incident=Path(args.scene).resolve()
    prepared=Path(args.prepared).resolve()
    home=Path(args.output).resolve()
    if home.exists():raise FileExistsError('Use a fresh continuation output')
    if home.is_relative_to(incident):raise ValueError('Output must not alter sealed incident')
    receipt=verify_scene(incident)
    pin_path=Path(args.scene_lock) if args.scene_lock else HERE/'snapshot.json'
    if not pin_path.is_file():raise ValueError('Missing trusted incident snapshot pin')
    pin=json.loads(pin_path.read_text())
    if sha(incident/'SCENE.json')!=pin['scene_sha256']:
        raise ValueError('Incident differs from the sealed benchmark snapshot')
    loader=Path(args.loader).resolve()
    if not loader.is_file():raise ValueError('Missing submitted loader.py')
    home.mkdir(parents=True)
    checkpoints=incident/'checkpoints'
    rows=json.loads((incident/'dataset/manifest.json').read_text())['rows']
    (home/'submitted').mkdir()
    shutil.copyfile(loader,home/'submitted/loader.py')
    before=sha(incident/'SCENE.json')
    write(home/'INPUTS.json',{'schema_version':2,'scene_sha256':before,
          'checkpoint_iteration':3,'start_step':4,'final_step':7,
          'loader_sha256':sha(home/'submitted/loader.py'),
          'expected_groups':expected_groups(rows,receipt['sample_cursor'],8,4,7),
          'samples_per_prompt':2,'source_revision':'75814318f243af7e39af337c1173fc3927c9635b',
          'model_revision':'b558bbc71d9433e7c5869effcbac07d4c7f2c5b4'})
    code=stage_source(HERE,prepared,home/'source')
    env={k:v for k,v in os.environ.items() if not k.startswith(('RELOCATION_','BENCH_RESTORE_'))
         and k not in ('PYTHONPATH','WORLD_SIZE','RANK','LOCAL_RANK','MASTER_ADDR','MASTER_PORT')}
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',RELOCATION_CODE=str(code),RELOCATION_PLUGIN=str(HERE),
               RELOCATION_OUTPUT=str(home/'run'),RELOCATION_DATA=str(incident/'dataset'),
               RELOCATION_LOADER=str(home/'submitted/loader.py'),RELOCATION_ARM='incident',
               RELOCATION_MODEL=str(prepared/'hf-safetensors'),RELOCATION_NATIVE=str(args.native),
               RELOCATION_STEPS='8',RELOCATION_AT='4',RELOCATION_BATCH='8',RELOCATION_SAMPLES='2',
               RELOCATION_MAX_RESPONSE='32768',RELOCATION_MAX_TOTAL_TOKENS='262144',
               BENCH_RESTORE_ROOT=str(checkpoints),BENCH_RESTORE_ITERATION='3',PYTHONUNBUFFERED='1')
    result={'returncode':1,'scene_sha256':before}
    try:
        with (home/'stdout.log').open('w') as log:
            proc=subprocess.Popen(['bash',str(HERE/'train.sh')],env=env,stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in proc.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            rc=proc.wait()
        if rc:raise RuntimeError('Native continuation failed: '+str(rc))
        from runtime_support.checkpoint_evidence import inspect_checkpoint
        outputs={role:inspect_checkpoint(home/'run'/role,expected_iteration=7,require_optimizer=True)
                 for role in ('actor','critic')}
        write(home/'output_checkpoints.json',outputs)
        verify_scene(incident)
        if sha(incident/'SCENE.json')!=before:raise ValueError('Sealed scene changed')
        result={'returncode':0,'scene_sha256':before,'original_scene_unchanged':True,'restored_iteration':3,
                'completed_steps':[4,5,6,7]}
    finally:
        write(home/'RESULT.json',result)
        print('INCIDENT_RESUME_RESULT='+json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('scene','prepared','native','loader','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--scene-lock',type=Path)
    run(p.parse_args())


if __name__=='__main__':main()
