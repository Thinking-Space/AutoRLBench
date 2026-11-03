#!/usr/bin/env python3
"""Offline CPU source/evidence checks. Does not build an image or train a model."""
from pathlib import Path
import ast,json,os,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
checks=[]
def check(name,ok,detail=''):
 checks.append({'check':name,'passed':bool(ok),'detail':detail})
def run(args,cwd=ROOT):
 return subprocess.run(args,cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
index=json.loads((ROOT/'benchmark-index.json').read_text())
for row in index['tasks']:
 task=ROOT/row['path'];label=row['task_id']
 p=run([sys.executable,'-m','autorlbench','validate',str(task)])
 check(label+': official validate',p.returncode==0,p.stdout.strip())
 for f in (ROOT/'submission_validation/upstream_scaffold').rglob('*'):
  if f.is_file():
   target=task/'evaluations'/f.relative_to(ROOT/'submission_validation/upstream_scaffold')
   check(label+': scaffold '+str(f.relative_to(ROOT/'submission_validation/upstream_scaffold')),target.exists() and target.read_bytes()==f.read_bytes())
 for f in (task/'environment').rglob('*.sh'):
  p=run(['bash','-n',str(f)]);check(label+': shell '+f.name,p.returncode==0,p.stdout.strip())
 for directory in [task/'environment',task/'workspace/case',task/'evaluations/solution/verifier/tests']:
  for f in directory.rglob('*.py'):
   try:ast.parse(f.read_text());valid=True;err=''
   except (SyntaxError,UnicodeError) as e:valid=False;err=str(e)
   check(label+': syntax '+str(f.relative_to(task)),valid,err)
 with tempfile.TemporaryDirectory() as d:
  shutil.copytree(task/'workspace/case',Path(d)/'case')
  patch=task/'evaluations/solution/oracle/fix.patch'
  p=run(['patch','--dry-run','-p1','-i',str(patch)],Path(d))
  check(label+': reference patch applies',p.returncode==0,p.stdout.strip())
p=run([sys.executable,'-m','unittest','discover','-s','submission_validation/tests','-v'])
check('independent grader evidence and mutation tests',p.returncode==0,p.stdout.strip())
marker='# ---- AutoRLBench tooling:'
new=(ROOT/'docker/images/verl-v061-grpo.Dockerfile').read_text();official=(ROOT/'docker/images/verl.Dockerfile').read_text()
check('Dockerfile official tooling identical',new[new.index(marker):]==official[official.index(marker):])
check('base image pinned by SHA256 digest','@sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a' in new)
pins=[s for s in (ROOT/'docker/verl-v061-runtime/runtime.lock.txt').read_text().splitlines() if s.strip() and not s.startswith('#')]
check('runtime package snapshot pinned',len(pins)==217 and all('==' in s for s in pins),str(len(pins))+' packages')
report={'offline_checks_passed':all(x['passed'] for x in checks),'checks':checks,'scope':'Source checks and recorded evidence replay only. No container build, GPU training or live diagnosis judge.'}
(ROOT/'submission_validation/source-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':sum(x['passed'] for x in checks),'total':len(checks),'ok':report['offline_checks_passed']},indent=2))
for c in checks:
 if not c['passed']:print(json.dumps(c))
sys.exit(0 if report['offline_checks_passed'] else 1)
