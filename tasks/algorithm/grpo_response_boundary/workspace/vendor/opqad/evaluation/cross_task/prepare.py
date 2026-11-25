#!/usr/bin/env python3
"""Prepare a new cross-task workspace from explicitly supplied, formatted data."""
import argparse, hashlib, json, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ['workdir','data-dir','model','arm']:p.add_argument('--'+x,required=True)
    p.add_argument('--kv',choices=['auto','turboquant_k2v3_nc','oscar_int2'],required=True)
    p.add_argument('--rotation',help='OSCAR directory containing model-specific K/V matrices')
    p.add_argument('--max-num-seqs',type=int,default=32)
    p.add_argument('--benches',nargs='+',choices=['aime24','aime25','aime26','ifeval','humaneval','lcbv6'],default=['aime25','aime26','ifeval','humaneval','lcbv6'])
    a=p.parse_args();root=Path(a.workdir).expanduser().resolve();data=Path(a.data_dir).expanduser().resolve()
    if root.exists():p.error('Workdir already exists; choose a new directory to protect prior results')
    if a.max_num_seqs<1:p.error('--max-num-seqs must be positive')
    if '/' in a.arm or a.arm in ['.','..']:p.error('Arm must be a simple directory name')
    arm={'model':str(Path(a.model).expanduser().resolve()),'method':a.kv,'max_num_seqs':a.max_num_seqs}
    if a.kv=='oscar_int2':
        if not a.rotation:p.error('OSCAR requires --rotation')
        r=Path(a.rotation).expanduser().resolve()
        for name in ['k_rotation_qqt_r_h_pbr.pt','v_rotation_sst_r_h_pbr.pt']:
            if not (r/name).is_file():p.error('Missing rotation '+str(r/name))
        arm['rotation']=str(r)
    hashes={};protocol={};sources=[]
    for b in a.benches:
        f=data/(b+'.json');raw=f.read_bytes();rows=json.loads(raw)
        if not rows or len({r['id'] for r in rows})!=len(rows):p.error(b+': empty or duplicate IDs')
        for r in rows:
            if not isinstance(r['id'],str) or not isinstance(r['prompt'],str):p.error(b+': id and prompt must be strings')
            needed=['answer'] if b.startswith('aime') else (['key','instruction_id_list','kwargs'] if b=='ifeval' else (['entry_point'] if b=='humaneval' else []))
            if any(k not in r for k in needed):p.error(b+': missing task-specific fields')
        hashes[b]=hashlib.sha256(raw).hexdigest();sources.append(f)
        math=b.startswith('aime')
        protocol[b]=dict(n=16 if math else 2,temperature=.7 if math else .9,top_p=.95,top_k=-1 if math else 20,max_output_tokens=32768 if math else 16384)
    if 'lcbv6' in a.benches:
        f=data/'livecodebench_v6.jsonl'
        if not f.is_file():p.error('LCB requires livecodebench_v6.jsonl with official test cases')
        sources.append(f)
    (root/'data').mkdir(parents=True)
    for f in sources:shutil.copy2(f,root/'data'/f.name)
    manifest=dict(arms={a.arm:arm},protocol=dict(benches=protocol,seed=42,thinking=True,num_shards=2,data_hashes=hashes))
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(root/'manifest.json')
if __name__=='__main__':main()
