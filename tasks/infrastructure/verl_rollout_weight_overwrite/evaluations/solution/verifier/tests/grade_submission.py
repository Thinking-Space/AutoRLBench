#!/usr/bin/env python3
"""Strict fresh-run grading with independent rescoring of all final AIME outputs."""
import argparse,collections,hashlib,json,math,os,statistics,sys,zipfile
from pathlib import Path

def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def load_rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def jsonfile(p):return json.loads(p.read_text())
def nested(x,path):
 for key in path.split('.'):x=x[key]
 return x

_CHECKPOINTS = {}
def verify_checkpoint(root):
 actor=root/'checkpoints/global_step_150/actor'
 fsdp=jsonfile(actor/'fsdp_config.json')
 if fsdp.get('world_size')!=8 or fsdp.get('FSDP_version')!=2:raise ValueError('Expected an eight-rank FSDP2 final checkpoint')
 paths=[actor/f'{kind}_world_size_8_rank_{rank}.pt' for kind in ['model','optim','extra_state'] for rank in range(8)]
 paths.append(root/'checkpoints/global_step_150/data.pt')
 key=tuple((str(p.resolve()),p.stat().st_size,p.stat().st_mtime_ns) for p in paths)
 if key not in _CHECKPOINTS:
  for p in paths:
   with zipfile.ZipFile(p) as z:
    if z.testzip() is not None:raise ValueError('Corrupt checkpoint archive')
  from safetensors import safe_open
  hf=actor/'huggingface';count=0
  for p in hf.glob('*.safetensors'):
   with safe_open(str(p),framework='pt',device='cpu') as f:
    for name in f.keys():count+=math.prod(f.get_slice(name).get_shape())
  if count<1_000_000_000:raise ValueError('Missing/incomplete HF checkpoint weights')
  _CHECKPOINTS[key]=True

def evaluate(results,spec,items,*,replay=False,workspace=None):
 errors=[];value=None;detail={};root=Path(results)/'experiment_logs'
 try:
  if not replay:
   report=Path(workspace)/'report.md'
   if not report.is_file() or not report.read_text().strip():raise ValueError('Missing nonempty report.md')
   verify_checkpoint(root)
   config=jsonfile(root/'effective_config.json');contract=jsonfile(Path(__file__).resolve().parents[1]/'resources/contract.json')
   resources=Path(__file__).resolve().parents[1]/'resources'
   registry=jsonfile(resources/'asset_registry.json')
   for key,name in [('data.train_files','train.parquet'),('data.val_files','val.parquet')]:
    data_path=Path(nested(config,key))
    if data_path.resolve()!=(Path(workspace)/'data'/name).resolve():raise ValueError('Unexpected data path')
    h=hashlib.sha256()
    with data_path.open('rb') as stream:
     for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=registry['derived'][name]['sha256']:raise ValueError('Prepared data modified')
   for key,wanted in contract.items():
    if nested(config,key)!=wanted:raise ValueError(f'Effective configuration contract differs: {key}')
  rows=load_rows(root/'metrics.jsonl');steps=[x['step'] for x in rows]
  if steps!=list(range(151)):raise ValueError('Expected exactly one ordered metric record at every step 0..150')
  for row in rows:
   for k,v in row['data'].items():
    if isinstance(v,(int,float)) and not finite(v):raise ValueError(f'Nonfinite metric: {k}')
  for step in range(0,151,10):
   d=rows[step]['data']
   for year in ['aime24','aime25']:
    key=f'val-core/{year}/acc/mean@32'
    if key not in d or not finite(d[key]) or not 0<=d[key]<=1:raise ValueError('Missing/invalid scheduled AIME evaluation')
  final=rows[-1]['data'];samples=load_rows(root/'fresh_eval/150.jsonl')
  if len(samples)!=1920:raise ValueError('Final evaluation must contain 1920 completions')
  expected={x['problem_id']:x for x in items};counts=collections.Counter();correct=collections.defaultdict(list)
  from score_math import score
  for x in samples:
   identity=x['problem_id']
   if identity not in expected:raise ValueError('Unknown evaluation question')
   e=expected[identity]
   if x['input']!=e['input'] or str(x['gts'])!=str(e['ground_truth']) or x['data_source']!=e['source'] or x['step']!=150:raise ValueError('Final evaluation question/answer identity mismatch')
   counts[identity]+=1;correct[e['source']].append(float(score(x['output'],x['gts'])))
  if set(counts)!=set(expected) or set(counts.values())!={32}:raise ValueError('Each of the 60 questions must have exactly 32 completions')
  for year in ['aime24','aime25']:
   acc=statistics.mean(correct[year]);logged=final[f'val-core/{year}/acc/mean@32']
   if abs(acc-logged)>1e-9:raise ValueError('Logged AIME accuracy differs from independent text rescoring')
   detail[year]=acc
  if spec['metric_key']=='training_reward_late30':
   values=[x['data']['critic/score/mean'] for x in rows if 121<=x['step']<=150]
   if len(values)!=30 or any(not finite(v) or not 0<=v<=1 for v in values):raise ValueError('Invalid final training reward window')
   if not replay:
    from score_math import score as score_training
    for step,expected_reward in zip(range(121,151),values):
     rollout=load_rows(root/f'rollouts/{step}.jsonl')
     if len(rollout)!=4096 or any(x['step']!=step for x in rollout):raise ValueError('Incomplete training rollout witness')
     actual=statistics.mean(float(score_training(x['output'],x['gts'])) for x in rollout)
     if abs(actual-expected_reward)>1e-9:raise ValueError('Training reward differs from independent answer rescoring')
   value=statistics.mean(values)
  else:value=statistics.mean(detail.values())
  if not finite(value) or value<spec['required_final_score'] or value>spec['required_final_score_max']:errors.append('Metric outside accepted interval')
 except Exception as e:errors.append(f'{type(e).__name__}: {e}')
 return {'passed':not errors,'score':float(not errors),'case_score':float(not errors),'value':value,'independent_final_accuracy':detail,'errors':errors,'historical_replay':replay}

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(os.environ.get('AUTORLBENCH_RESULTS','/results')));a=p.parse_args();resources=Path(__file__).resolve().parents[1]/'resources'
 result=evaluate(a.results,jsonfile(resources/'grader_spec.json'),jsonfile(resources/'eval_items.json'),workspace=Path(os.environ.get('AUTORLBENCH_WORKSPACE','/workspace')))
 a.results.mkdir(parents=True,exist_ok=True)
 for n in ['result.json','metrics.json']:(a.results/n).write_text(json.dumps(result,indent=2))
 print(json.dumps(result));return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
