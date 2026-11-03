#!/usr/bin/env python3
"""Reproduce the recorded parquet bytes; retain all repeated/conflicting train rows."""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import prepare_dapo as train_builder
import prepare_aime_eval as eval_builder

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def check(p,spec):
 if not p.is_file() or p.stat().st_size!=spec['size'] or digest(p)!=spec['sha256']:raise ValueError(f'Asset identity mismatch: {p}')

def run(workspace,registry):
 w=Path(workspace).resolve();r=json.loads(Path(registry).read_text());out=w/'data';model=w/'model';inputs=w/'assets/raw'
 for a in r['datasets']:
  for f in a['files']:check(inputs/a['id']/f['path'],f)
 # Model bytes are verified by the official fetch helper; verify tokenizer identities again here.
 for f in r['model']['files']:
  if not f['path'].endswith('.safetensors'):check(model/f['path'],f)
 def verify_outputs():
  for name,s in r['derived'].items():
   p=out/name;check(p,s)
   if pq.ParquetFile(p).metadata.num_rows!=s['rows']:raise ValueError(f'Row count differs: {p}')
 if (out/'.complete').exists():
  verify_outputs();return {'status':'reused','verified':True}
 if out.exists() and list(out.iterdir()):raise ValueError('Nonempty data directory without completion marker; do not overwrite')
 out.mkdir(parents=True,exist_ok=True)
 tok=train_builder.load_tokenizer(model)
 with tempfile.TemporaryDirectory(dir=out,prefix='.preparing-') as temp:
  stage=Path(temp)
  train_builder.copy_split(inputs/'dapo/data/dapo-math-17k.parquet','train',stage/'train.parquet',tok,{}, {})
  sources={a['id']:pq.read_table(inputs/a['id']/a['data_file']).to_pylist() for a in r['datasets'] if a['id']!='dapo'}
  validation,coverage=eval_builder.build_validation(sources,tok)
  pq.write_table(pa.Table.from_pylist(validation),stage/'val.parquet',compression='zstd')
  for name,s in r['derived'].items():check(stage/name,s)
  for name in r['derived']:
   os.replace(stage/name,out/name);(out/name).chmod(0o444)
 verify_outputs()
 manifest={'registry_sha256':digest(registry),'outputs':r['derived'],'training_rows_preserved':1791700,'validation_questions':60,'deduplicated':False,'train_prompt_text_or_labels_modified':False}
 (out/'prepared_manifest.json').write_text(json.dumps(manifest,indent=2));(out/'.complete').write_text(json.dumps(manifest,sort_keys=True))
 return {'status':'prepared','verified':True,'outputs':r['derived']}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path(os.environ.get('AUTORLBENCH_WORKSPACE','/workspace')));p.add_argument('--registry',type=Path,default=Path(__file__).with_name('asset_registry.json'));a=p.parse_args();print(json.dumps(run(a.workspace,a.registry),indent=2))
