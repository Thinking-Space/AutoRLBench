"""Inactive persistent CPU supervisor for the engineering-to-continuous handoff.

The current healthy owner is always followed.  An absent deployment admission,
uncertain ownership, unaccepted checkpoint or incomplete cleanup preserves the
current allocation and reports a pending state; none permits a second job.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
import time

from launch_continuous import ROOT, HERE, POD, small, atomic, process_identity, read_control
from supervisor_contract import Pending, classify

STATE = ROOT/'work/runtime/libero-continuous'


def obj(path):
    value=json.loads(small(path))
    if not isinstance(value,dict):raise ValueError('Expected a bounded JSON object')
    return value


def followed_owner():
    continuous=ROOT/'evidence/eight-node-continuous-owner-launch.json'
    if continuous.exists():
        return 'eight-node-continuous',obj(continuous)
    return 'eight-node-engineering10',obj(ROOT/'evidence/eight-node-engineering10-owner-launch.json')


def owner_liveness(owner):
    if owner.get('status')=='spawn_intent' or 'holder_pid' not in owner:
        raise Pending('A persisted spawn intent requires owner/GCS reconciliation before another launch')
    current=process_identity(owner['holder_pid'])
    if current is None or current['state']=='Z':return False
    if current['startticks']!=owner['holder_startticks']:return False
    process=Path('/proc')/str(owner['holder_pid'])
    if process.stat().st_uid!=os.getuid() or str((process/'cwd').readlink())!=str(ROOT):
        raise Pending('The live owner no longer has the admitted UID/cwd identity')
    return True


def terminal_verified(controller):
    # Original exact jobs/actors/runtime owner verification, with no Ray init.
    path=ROOT/'source/pear/pear/trainer/formal_gcs.py'
    spec=importlib.util.spec_from_file_location('continuous_release_validation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.check_export_release(ROOT,controller)
    return True


def accepted(marker,prefix,ticket):
    if prefix=='eight-node-continuous':
        path=ROOT/'evidence/libero-continuous'/ticket/f'step-{marker}.json'
        if not path.exists():
            # The attempt may have restored a previously accepted checkpoint
            # and failed before its first update; use its exact source binding.
            launch=obj(ROOT/'evidence/libero-continuous'/ticket/'launch.json')
            if (launch.get('ticket')!=ticket
                    or launch.get('starting_checkpoint',{}).get('global_step')!=marker):
                raise ValueError('The failed continuous attempt has another starting checkpoint or ticket')
            reference=launch['source_acceptance'];path=ROOT/reference['path']
            raw=small(path)
            if hashlib.sha256(raw).hexdigest()!=reference['sha256']:
                raise ValueError('The accepted continuous predecessor changed')
            return json.loads(raw)
        return obj(path)
    name=('eight-node-engineering-step5-acceptance.json' if marker==5
          else f'eight-node-engineering10-step{marker}-acceptance.json')
    return obj(ROOT/'evidence'/name)


def poll():
    control_path=STATE/'control.json'
    control=read_control(control_path)
    prefix,owner=followed_owner();ticket=owner['attempt_id']
    alive=owner_liveness(owner)
    base={'at':time.time(),'attempt_id':ticket,'controller_prefix':prefix}
    if control is not None and control.get('action')=='stop':
        return {**base,'status':'stop_latched','action':'stop_latched',
                'meaning':('The continuous trainer consumes the durable stop at its safe boundary; no new launch'
                           if prefix=='eight-node-continuous' else
                           'No new launch; the existing finite owner requires verified safe-boundary stop handling')}
    if alive:
        return {**base,'status':'following_current','action':'follow','owner_live':True}
    controller=obj(ROOT/'evidence'/f'{prefix}-controller.json')
    marker=int(small(ROOT/'work/runs/clean-rl-smoke/ckpt/latest_checkpointed_iteration.txt',32))
    acceptance=accepted(marker,prefix,ticket)
    driver=obj(ROOT/'evidence/formal-gcs'/f'{ticket}-driver.json')
    gcs=obj(ROOT/'evidence/eight-node-current-gcs.json')
    hosts=obj(ROOT/'config/cluster-eight-h20-20261118.json')['hosts']
    ready=HERE/'activation-ready.json';deployment=obj(ready) if ready.exists() else None
    decision=classify(control=control,owner_live=False,controller=controller,acceptance=acceptance,
                      marker=marker,gcs=gcs,expected_ticket=ticket,hosts=hosts,now=time.time(),
                      deployment=deployment,driver_claim=driver,terminal_verified=terminal_verified(controller))
    return {**base,'status':decision.action,'action':decision.action,'source_step':decision.source_step,
            'reason':decision.reason,'owner_live':False}


def act(state):
    """Only the deployment's exact executable may perform the one safe cutover."""
    action=state['action']
    if action not in ('start_continuous','resume_engineering'):return
    control=read_control(STATE/'control.json')
    if control is not None and control['action']=='stop':
        raise Pending('A user stop arrived after the last poll; no new launch is allowed')
    if action=='start_continuous':
        admission=obj(HERE/'activation-ready.json')
        reference=admission['deployment_entrypoint'];entry=ROOT/reference['path']
        if entry.parent!=HERE or hashlib.sha256(small(entry)).hexdigest()!=reference['sha256']:
            raise ValueError('The admitted deployment entrypoint differs')
        command=[sys.executable,'-B',str(entry),'--execute','--follow-ticket',state['attempt_id']]
    else:
        command=[sys.executable,'-B',str(ROOT/'work/dev/libero-engineering10-launch/launch.py'),'--execute']
    intent=STATE/'next-launch-intent.json'
    if intent.exists():
        prior=obj(intent)
        if prior.get('followed_ticket')==state['attempt_id']:
            raise Pending('An existing next-launch intent requires reconciliation; do not double-spawn')
    atomic(intent,{'at':time.time(),'followed_ticket':state['attempt_id'],
                   'source_step':state['source_step'],'action':action})
    # The child launchers persist their own immutable intent, holder identity,
    # source admission and canonical owner.  They use private_runtime and setsid.
    control=read_control(STATE/'control.json')
    if control is not None and control['action']=='stop':
        raise Pending('A user stop arrived before child launch; preserve the persisted intent')
    result=subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT,text=True,timeout=1200,
                          env={**os.environ,'CUDA_VISIBLE_DEVICES':'','NO_GPU':'1',
                               'PYTHONDONTWRITEBYTECODE':'1'})
    output=result.stdout.encode()[-8192:]
    (STATE/'last-launch-output.log').write_bytes(output)
    atomic(STATE/'last-launch-result.json',{'at':time.time(),'returncode':result.returncode,
              'followed_ticket':state['attempt_id'],'action':action,
              'output_sha256':hashlib.sha256(output).hexdigest()})
    if result.returncode:
        raise Pending('The admitted child launch failed; preserve guard and reconcile its persisted intent')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if (os.environ.get('POD_NAME')!=POD or os.environ.get('CUDA_VISIBLE_DEVICES')!=''
            or os.environ.get('NO_GPU')!='1'):
        parser.error('Use the verified GPU launcher CPU environment')
    if not (args.once or args.execute):parser.error('Choose --once or --execute')
    os.umask(0o077);STATE.mkdir(parents=True,exist_ok=True)
    if STATE.resolve()!=STATE:raise ValueError('The supervisor directory is not canonical')
    handler=RotatingFileHandler(STATE/'supervisor.log',maxBytes=2*1024*1024,backupCount=2)
    logger=logging.getLogger('libero-continuous-supervisor');logger.setLevel(logging.INFO);logger.addHandler(handler)
    with (STATE/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=process_identity(os.getpid())
        atomic(STATE/'supervisor-owner.json',{'at':time.time(),'pid':os.getpid(),**identity})
        previous=None
        while True:
            try:
                state=poll()
                if args.execute:act(state)
            except (OSError,ValueError,KeyError,TypeError,Pending,subprocess.SubprocessError) as error:
                state={'at':time.time(),'status':'pending_or_blocked','error_type':type(error).__name__,
                       'reason':str(error)[:1000],'gpu_state':'not_assumed_safe'}
            atomic(STATE/'supervisor-state.json',state)
            summary={k:v for k,v in state.items() if k!='at'}
            if summary!=previous:
                logger.info(json.dumps(state,ensure_ascii=False));print(json.dumps(state,ensure_ascii=False),flush=True)
                previous=summary
            if args.once:return
            time.sleep(30)


if __name__=='__main__':main()
