"""Rebuild the historical Skywork selection from pinned raw data on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--assets',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    root=args.workspace;dest=root/'data';dest.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(root/'models/actor',local_files_only=True)
    table=pq.read_table(args.raw/'data/math-00000-of-00001.parquet')
    difficulty=table['extra_info'].combine_chunks().field('model_difficulty').field('DeepSeek-R1-Distill-Qwen-7B').to_numpy()
    rows=table.take(np.flatnonzero((difficulty>0)&(difficulty<16))).to_pylist()
    selected=len(rows);kept=[]
    for start in range(0,len(rows),256):
        part=rows[start:start+256]
        tokens=tokenizer.apply_chat_template([row['prompt'] for row in part],add_generation_prompt=True,tokenize=True,return_dict=False)
        kept.extend(row for row,ids in zip(part,tokens,strict=True) if len(ids)<=1024)
    if table.num_rows!=105055 or selected!=48371 or len(kept)!=48353:
        raise ValueError('Data/model preprocessing differs from the recorded selection')
    pq.write_table(pa.Table.from_pylist(kept,schema=table.schema),dest/'train.parquet',compression='zstd')
    manifest=json.loads((args.assets/'manifest.json').read_text())
    for name,item in manifest['evaluation'].items():
        source=args.assets/name
        if source.stat().st_size!=item['bytes'] or hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Frozen validation asset integrity mismatch: '+name)
        shutil.copyfile(source,dest/name)
    logical=hashlib.sha256()
    for row in kept:logical.update((json.dumps(row,sort_keys=True,ensure_ascii=False,default=str)+'\n').encode())
    (dest/'.complete').write_text(json.dumps({'source_rows':table.num_rows,'difficulty_selected':selected,
        'prompt_length_retained':len(kept),'logical_rows_sha256':logical.hexdigest(),
        'eval_rows':{'AIME24':30,'AIME25':30,'AMC23':40},
        'byte_identity_note':'Parquet bytes can differ by Arrow version; selection, ordering, prompt and GT are retained.'},indent=2)+'\n')

if __name__=='__main__':main()
