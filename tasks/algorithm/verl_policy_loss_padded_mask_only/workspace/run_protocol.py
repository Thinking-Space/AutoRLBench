#!/usr/bin/env python3
"""Fresh, foreground veRL rerun. No training data or hyperparameters are changed."""
import argparse,datetime,json,os,signal,subprocess,sys,tempfile,time
from pathlib import Path
import yaml

def relocate(x,workspace):
 if isinstance(x,dict):return {k:relocate(v,workspace) for k,v in x.items()}
 if isinstance(x,list):return [relocate(v,workspace) for v in x]
 if isinstance(x,str) and (x=='/workspace' or x.startswith('/workspace/')):return str(workspace)+x[len('/workspace'):]
 return x

def start_cluster(cpus,gpus_per_node=4,test=False):
 import ray
 from ray.cluster_utils import Cluster
 cluster=Cluster(shutdown_at_exit=True)
 try:
  for node in range(2):
   devices=','.join(str(i) for i in range(node*4,node*4+4))
   cluster.add_node(num_cpus=cpus,num_gpus=0 if test else gpus_per_node,include_dashboard=False,object_store_memory=256*1024*1024,node_ip_address=ray._private.services.get_node_ip_address(),resources={f'partition_{node}':1},env_vars={'CUDA_VISIBLE_DEVICES':devices,'AUTORLBENCH_NODE_PARTITION':str(node)})
  cluster.wait_for_nodes(timeout=90)
  return cluster,cluster.address
 except BaseException:
  cluster.shutdown();raise

def main():
 p=argparse.ArgumentParser();p.add_argument('--preflight-only',action='store_true');p.add_argument('--cluster-test',action='store_true');a=p.parse_args();w=Path(__file__).resolve().parent
 if a.cluster_test:
  import ray
  c,address=start_cluster(1,test=True)
  try:
   ray.init(address=address,logging_level='ERROR')
   @ray.remote
   def probe():return {'partition':os.environ.get('AUTORLBENCH_NODE_PARTITION'),'node':ray.get_runtime_context().get_node_id()}
   rows=ray.get([probe.options(resources={f'partition_{i}':1}).remote() for i in range(2)])
   assert len({x['node'] for x in rows})==2 and {x['partition'] for x in rows}=={'0','1'}
   print(json.dumps({'two_ray_nodes_verified':True,'gpu_partitioning_verified':False,'nodes':rows}),flush=True)
  finally:ray.shutdown();c.shutdown()
  return
 results=Path(os.environ['AUTORLBENCH_RESULTS']).resolve();logs=results/'experiment_logs';logs.mkdir(parents=True,exist_ok=True)
 if (logs/'metrics.jsonl').exists():raise ValueError('Fresh metrics already exist; harness must clear experiment_logs before rerun')
 cfg=relocate(yaml.safe_load((w/'case/recorded_config.yaml').read_text()),w)
 # These are output-location/backend wiring changes, not fixes to the selected policy or reward.
 cfg['trainer'].update(default_local_dir=str(logs/'checkpoints'),rollout_data_dir=str(logs/'rollouts'),validation_data_dir=str(logs/'fresh_eval'),logger=['console','file'])
 cfg['ray_kwargs']['ray_init']={}
 env={**os.environ,'PYTHONPATH':str(w),'VERL_FILE_LOGGER_PATH':str(logs/'metrics.jsonl'),'CASE_SYNC_TRACE':str(logs/'sync_trace.jsonl'),'WANDB_MODE':'offline','HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1','TOKENIZERS_PARALLELISM':'false','VLLM_USE_V1':'1','PYTHONUNBUFFERED':'1','PYTHONHASHSEED':'42'}
 for key in ['SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP','MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP']:env.pop(key,None)
 runtime=w/'.autorlbench-runtime';runtime.mkdir(exist_ok=True)
 config=runtime/'effective.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
 command=[sys.executable,'-m','verl.trainer.main_ppo','--config-path',str(runtime),'--config-name','effective']
 # Hydra resolves the same config the actual training command will receive.
 pre=subprocess.run(command+['--cfg','job','--resolve'],cwd=w,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (logs/'hydra_preflight.log').write_text(pre.stdout)
 if pre.returncode:raise RuntimeError('Hydra composition failed; see hydra_preflight.log')
 lines=pre.stdout.splitlines();start=next(i for i,s in enumerate(lines) if s=='actor_rollout_ref:');effective=yaml.safe_load('\n'.join(lines[start:]));(logs/'effective_config.json').write_text(json.dumps(effective,indent=2))
 if a.preflight_only:
  print(json.dumps({'hydra_composed':True,'gpu_training_started':False,'nnodes':effective['trainer']['nnodes'],'prompt_cap':effective['data']['max_prompt_length']}));return
 from importlib.metadata import version
 assert version('torch').split('+')[0]=='2.8.0' and version('vllm')=='0.11.0'
 import torch,ray
 cluster=None;address=os.environ.get('RAY_ADDRESS');mode='external_two_nodes'
 try:
  if not address:
   if torch.cuda.device_count()!=8:raise ValueError('Local 2x4 rerun requires eight visible GPUs; supply an external two-node RAY_ADDRESS otherwise')
   cpus=max(2,(os.cpu_count() or 8)//2);cluster,address=start_cluster(cpus);mode='single_host_two_logical_nodes'
  ray.init(address=address,logging_level='ERROR')
  nodes=[n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU',0)>0]
  if len(nodes)!=2 or any(n['Resources']['GPU']!=4 for n in nodes):raise ValueError('Expected exactly two Ray workers with four GPUs each')
  topology={'mode':mode,'nodes':[{'node_id':n['NodeID'],'gpus':n['Resources']['GPU']} for n in nodes],'historical_topology':'two physical nodes x four GPUs','physical_topology_equivalence_tested':False}
  (logs/'topology.json').write_text(json.dumps(topology,indent=2));ray.shutdown()
  # Workers run identical source/assets at the same workspace path. Agent-installed packages may be repaired legitimately.
  worker_env={k:env[k] for k in ['PYTHONPATH','VERL_FILE_LOGGER_PATH','CASE_SYNC_TRACE','WANDB_MODE','HF_HUB_OFFLINE','TOKENIZERS_PARALLELISM','VLLM_USE_V1','PYTHONUNBUFFERED','PYTHONHASHSEED']}
  cfg['ray_kwargs']['ray_init']={'address':address,'runtime_env':{'env_vars':worker_env}}
  config.write_text(yaml.safe_dump(cfg,sort_keys=False))
  with (logs/'train.log').open('w') as f:
   code=subprocess.call(command,cwd=w,env=env,stdout=f,stderr=subprocess.STDOUT)
  if code:raise RuntimeError(f'veRL exited with code {code}; see train.log')
 finally:
  if ray.is_initialized():ray.shutdown()
  if cluster is not None:cluster.shutdown()

if __name__=='__main__':main()
