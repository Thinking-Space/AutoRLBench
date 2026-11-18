"""Adapt task asset manifests to the unmodified upstream HF fetch helper."""
import argparse,hashlib,json,pathlib,re,subprocess,sys,tempfile
from pathlib import PurePosixPath

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True,type=pathlib.Path);parser.add_argument('--dest',required=True,type=pathlib.Path);args=parser.parse_args()
 manifest=json.loads(args.manifest.read_text());assert manifest['schema_version']==1
 assert manifest['repo_type'] in ('model','dataset') and re.fullmatch(r'[0-9a-f]{40}',manifest['revision']) and manifest['repo_id']
 files={}; sizes={}
 for item in manifest['files']:
  name=item['path'];p=PurePosixPath(name);assert name and not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name and name not in files
  assert re.fullmatch(r'[0-9a-f]{64}',item['sha256']);size=item['size_bytes'];assert isinstance(size,int) and not isinstance(size,bool) and size>=0
  files[name]=item['sha256'];sizes[name]=size
 assert files
 with tempfile.TemporaryDirectory(prefix='autorlbench-official-manifest-') as tmp:
  pinned=pathlib.Path(tmp)/'official.json';pinned.write_text(json.dumps({'files':files},sort_keys=True)+'\n')
  subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('fetch_hf.py')),'--type',manifest['repo_type'],'--repo',manifest['repo_id'],'--revision',manifest['revision'],'--manifest',str(pinned),'--dest',str(args.dest)],check=True)
 for name,expected in files.items():
  f=args.dest/name;assert f.is_file() and f.stat().st_size==sizes[name],('size mismatch',name)
  digest=hashlib.sha256()
  with f.open('rb') as h:
   for block in iter(lambda:h.read(1<<24),b''):digest.update(block)
  assert digest.hexdigest()==expected,('digest mismatch',name)
 print(f'[prepare] verified {len(files)} file sizes and SHA-256 hashes')
if __name__=='__main__':main()
