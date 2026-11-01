import argparse,os,json,threading,time,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--validate-only',action='store_true');a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
os.environ['RUN_OUTPUT']=str(out)
os.environ['VLLM_USE_V1']='0'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['RAY_DEDUP_LOGS']='0'
os.environ['RAY_memory_usage_threshold']='0.95'
os.environ['RAY_memory_monitor_refresh_ms']='100'
os.environ['PYTHONUNBUFFERED']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='2'
os.environ['WANDB_MODE']='offline'
assert Path('/sys/fs/cgroup/memory.max').read_text().strip() == os.environ['CONTAINER_MEMORY_LIMIT_BYTES'], 'container memory limit mismatch'
import psutil
stop=threading.Event()
def monitor():
 with (out/'host_memory.jsonl').open('w',buffering=1) as f:
  while not stop.is_set():
   files={}
   for name in ['memory.current','memory.peak','memory.max','memory.events','memory.stat']:
    q=Path('/sys/fs/cgroup')/name
    if q.exists():files[name]=q.read_text().strip()
   procs=[]
   for proc in psutil.process_iter(['pid','name','memory_info']):
    try:procs.append({'pid':proc.pid,'name':proc.info['name'],'rss':proc.info['memory_info'].rss})
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
   f.write(json.dumps({'time':time.time(),'cgroup':files,'processes':procs})+'\n');stop.wait(.2)
threading.Thread(target=monitor,daemon=True).start()
try:
 from omegaconf import OmegaConf
 import ray
 config=OmegaConf.merge(OmegaConf.load('verl/trainer/config/ppo_trainer.yaml'),OmegaConf.load('configs/experiment.yaml'))
 config.trainer.default_local_dir=str(out/'checkpoints')
 if a.validate_only:
  assert a.resume, '--validate-only requires --resume'
  config.trainer.val_only=True
 if a.resume:
  config.trainer.resume_mode='resume_path'
  config.trainer.resume_from_path=a.resume
 OmegaConf.save(config,out/'resolved_config.yaml')
 ray.init(num_cpus=4,include_dashboard=False,object_store_memory=134217728,_temp_dir='/tmp/ray',runtime_env={'env_vars':{'RUN_OUTPUT':str(out),'PYTHONPATH':'/workspace','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}})
 from verl.trainer.main_ppo import run_ppo
 run_ppo(config)
 (out/'completed.json').write_text(json.dumps({'completed':True,'steps':int(config.trainer.total_training_steps),'mode':'validation' if a.validate_only else 'training'}))
finally:
 stop.set()
 for name in ['memory.current','memory.max','memory.events']:
  q=Path('/sys/fs/cgroup')/name
  if q.exists():(out/('final_'+name.replace('.','_')+'.txt')).write_text(q.read_text())
 import shutil
 logroot=Path('/tmp/ray/session_latest/logs')
 if logroot.exists():
  dest=out/'ray_logs';dest.mkdir(exist_ok=True)
  for q in logroot.iterdir():
   if q.is_file() and (q.name.startswith('raylet.') or q.name.startswith('worker-')):
    shutil.copy2(q,dest/q.name)
 if 'ray' in globals():
  children = psutil.Process().children(recursive=True)
  ray.shutdown()
  _, alive = psutil.wait_procs(children, timeout=10)
  for child in alive:
   try:child.terminate()
   except psutil.NoSuchProcess:pass
  _, alive = psutil.wait_procs(alive, timeout=5)
  for child in alive:
   try:child.kill()
   except psutil.NoSuchProcess:pass
  psutil.wait_procs(alive, timeout=5)
