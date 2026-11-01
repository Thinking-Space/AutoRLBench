"""One-time authoring: run the real four-update prefix and seal an incident.

Requires the pinned public runtime with three GPUs. Evaluating a submitted patch
uses run.py and reuses this checkpoint; it never rebuilds the prefix.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

from prefix import stop_after_checkpoint
from scene import create_scene, pack_scene, verify_scene


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-native',type=Path)
    p.add_argument('--pack',type=Path)
    args=p.parse_args()
    proposal=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(proposal))
    import run_case
    from prepare_assets import verify_prepared,write_json,file_sha
    from runtime_support.checkpoint_evidence import inspect_checkpoint
    if args.output.exists():raise FileExistsError('Use a new authoring output')
    assets,converter=verify_prepared(args.prepared)
    args.output.mkdir(parents=True)
    home=args.output/'prefix'
    home.mkdir();(home/'plugin').mkdir()
    for name in run_case.PLUGINS:shutil.copyfile(proposal/'runtime'/name,home/'plugin'/name)
    shutil.copytree(args.prepared/'dataset',home/'dataset')
    code,source=run_case.stage_source(home,args.prepared,assets)
    train=code/'train.py';train.write_text(stop_after_checkpoint(train.read_text(),3))
    source['instrumented_files']['train.py']=file_sha(train)
    write_json(home/'source/source_receipt.json',source)
    if args.reuse_native:
        run_case.verify_conversion(args.reuse_native,args.prepared/'hf-safetensors',converter,assets,source)
        shutil.copytree(args.reuse_native,home/'native')
    else:
        run_case.convert_native(home,code,args.prepared/'hf-safetensors',converter,assets,source,29365)
    config=run_case.parse_args(['--prepared',str(args.prepared),'--output',str(home),'--arm','healthy'])
    env=run_case.training_env(config,home,code,args.prepared/'hf-safetensors')
    write_json(home/'INPUTS.json',{'schema_version':2,'stage':'prefix','arm':'healthy','steps':8,
           'completed_through':3,'resume_start':4,'model_revision':assets['model_revision'],
           'source_revision':assets['justrl2_revision'],
           'manifest_sha256':file_sha(home/'dataset/manifest.json'),'eval_sha256':file_sha(home/'dataset/eval.jsonl'),
           'case_sha256':file_sha(home/'dataset/CASE.json')})
    rc=run_case.run_logged(['bash',str(home/'plugin/train.sh')],cwd=code,env=env,log=home/'stdout.log')
    if rc:raise RuntimeError('Real prefix training failed')
    checkpoints={role:inspect_checkpoint(home/'run'/role,expected_iteration=3,require_optimizer=True)
                 for role in ('actor','critic')}
    write_json(home/'provenance/output_checkpoints.json',checkpoints)
    scene=args.output/'scene'
    receipt=create_scene(home,scene)
    verify_scene(scene)
    result={'scene':str(scene),'checkpoint_iteration':receipt['checkpoint_iteration'],
            'scene_sha256':file_sha(scene/'SCENE.json'),'restore_tested':False}
    if args.pack:result['archive']=pack_scene(scene,args.pack)
    write_json(args.output/'BUILD_RESULT.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
