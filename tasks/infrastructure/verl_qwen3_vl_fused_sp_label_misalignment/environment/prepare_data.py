"""Materialize fixed recorded rows from immutable raw image MC data."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import pyarrow as pa
import pyarrow.parquet as pq

p=argparse.ArgumentParser()
p.add_argument('--raw-root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a=p.parse_args()
spec=json.loads((Path(__file__).parent/'selection.json').read_text())
assets=json.loads((Path(__file__).parent/'assets.json').read_text())
dataset=next(x for x in assets if x['kind']=='dataset')
def sha(b): return hashlib.sha256(b).hexdigest()
def normalized(text): return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC',str(text))).strip().lower()
a.output.mkdir(parents=True,exist_ok=True)
report={}
for split in ['train','validation']:
    raw=next(x for x in dataset['files'] if x['path'].startswith('data/'+split+'-'))
    source=a.raw_root/raw['path']
    if source.stat().st_size!=raw['bytes'] or sha(source.read_bytes())!=raw['sha256']:
        raise ValueError('raw dataset checksum: '+split)
    rows=pq.read_table(source).to_pylist(); selected=[]
    s=spec['splits'][split]
    for entry in s['selection']:
        row=rows[entry['original_index']]
        key=normalized(row['question'])+'\n'+'\n'.join(sorted(normalized(c) for c in row['choices']))
        if sha(key.encode())!=entry['question_sha256'] or sha(row['image']['bytes'])!=entry['image_sha256']:
            raise ValueError('row identity')
        options='\n'.join(f'{chr(65+i)}. {choice}' for i,choice in enumerate(row['choices']))
        hint=str(row.get('hint') or '').strip()
        text='<image>\n'+str(row['question']).strip()+('\nContext: '+hint if hint else '')+'\n'+options+'\nExplain your reasoning, then put the final option letter in \\boxed{}, for example \\boxed{A}.'
        if sha(text.encode())!=entry['prompt_sha256']:raise ValueError('prompt identity')
        selected.append({'data_source':'scienceqa_image_mc','prompt':[{'role':'user','content':text}],
                         'images':[{'bytes':row['image']['bytes'],'path':None}],'ability':'science',
                         'reward_model':{'style':'rule','ground_truth':chr(65+int(row['answer']))},
                         'extra_info':{'split':split,'index':entry['original_index'],'num_choices':len(row['choices'])}})
    target=a.output/f'{split}.parquet'
    if target.exists():
        if target.stat().st_size!=s['bytes'] or sha(target.read_bytes())!=s['sha256']:raise ValueError('existing output mismatch')
    else:
        temp=target.with_suffix('.parquet.partial')
        pq.write_table(pa.Table.from_pylist(selected),temp)
        if temp.stat().st_size!=s['bytes'] or sha(temp.read_bytes())!=s['sha256']:raise ValueError('prepared parquet checksum')
        temp.replace(target)
    report[split]={'count':len(selected),'bytes':target.stat().st_size,'sha256':sha(target.read_bytes())}
print(json.dumps(report,indent=2))
