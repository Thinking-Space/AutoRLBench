"""Install pinned wheels in the task-prepared directory without replacing the image stack."""
import hashlib,json,os,pathlib,shutil,subprocess,sys,tempfile,urllib.request

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 root=pathlib.Path(sys.argv[1]);root.mkdir(parents=True,exist_ok=True);spec=json.loads(pathlib.Path(__file__).with_name('dependencies.json').read_text());wheels=root/'wheel_cache';wheels.mkdir(exist_ok=True)
 for item in spec:
  p=wheels/item['filename']
  if not p.exists():
   temp=p.with_suffix('.partial')
   try:
    with urllib.request.urlopen(item['url'],timeout=120) as response,temp.open('wb') as out:shutil.copyfileobj(response,out)
    if temp.stat().st_size!=item['size'] or digest(temp)!=item['sha256']:raise ValueError(('download mismatch',item['filename']))
    os.replace(temp,p)
   finally:temp.unlink(missing_ok=True)
  if p.stat().st_size!=item['size'] or digest(p)!=item['sha256']:raise ValueError(('cached wheel mismatch',item['filename']))
 target=root/'dependencies';marker=target/'.autorl-pins.json'
 if target.exists():
  if not marker.is_file() or json.loads(marker.read_text())!=spec:raise ValueError('dependency target exists with different pins')
 else:
  tmp=pathlib.Path(tempfile.mkdtemp(prefix='.dependencies-',dir=root))
  try:
   subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-index','--target',str(tmp),*[str(wheels/x['filename']) for x in spec]],check=True)
   (tmp/'.autorl-pins.json').write_text(json.dumps(spec)+'\n');os.replace(tmp,target)
  finally:
   if tmp.exists():shutil.rmtree(tmp)
 env=dict(os.environ,PYTHONPATH=str(target),SAFE_TARGET=str(target.resolve()))
 subprocess.run([sys.executable,'-c',"import os,torch,vllm,flashinfer,modelopt.torch.opt; from modelopt.torch.quantization.nn import TensorQuantizer; assert all(not os.path.realpath(x.__file__).startswith(os.environ['SAFE_TARGET']+'/') for x in (torch,vllm,flashinfer)); print('image frameworks retained; ModelOpt available')"],env=env,check=True)
 envfile=root/'env';lines=envfile.read_text().splitlines() if envfile.exists() else [];value='PYTHONPATH='+str(target);existing=[x for x in lines if x.startswith('PYTHONPATH=')]
 if existing and existing!=[value]:raise ValueError('prepared PYTHONPATH already differs')
 if not existing:envfile.write_text('\n'.join(lines+[value])+'\n')
if __name__=='__main__':main()
