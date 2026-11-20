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
def file_sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def normalized(text): return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC',str(text))).strip().lower()
a.output.mkdir(parents=True,exist_ok=True)
report={}
for split in ['train','validation']:
    raw=next(x for x in dataset['files'] if x['path'].startswith('data/'+split+'-'))
    source=a.raw_root/raw['path']
    if source.stat().st_size!=raw['bytes'] or file_sha(source)!=raw['sha256']:
        raise ValueError('raw dataset checksum: '+split)
    s=spec['splits'][split]
    entries=s['selection']
    by_index={entry['original_index']:(position,entry) for position,entry in enumerate(entries)}
    if len(by_index)!=len(entries):raise ValueError('duplicate selection index')
    selected=[None]*len(entries)
    offset=0
    for batch in pq.ParquetFile(source, pre_buffer=False).iter_batches(
            batch_size=8, columns=['question','choices','image','hint','answer'], use_threads=False):
        for row in batch.to_pylist():
            match=by_index.get(offset)
            offset+=1
            if match is None:continue
            position,entry=match
            key=normalized(row['question'])+'\n'+'\n'.join(sorted(normalized(c) for c in row['choices']))
            if sha(key.encode())!=entry['question_sha256'] or sha(row['image']['bytes'])!=entry['image_sha256']:
                raise ValueError('row identity')
            options='\n'.join(f'{chr(65+i)}. {choice}' for i,choice in enumerate(row['choices']))
            hint=str(row.get('hint') or '').strip()
            text='<image>\n'+str(row['question']).strip()+('\nContext: '+hint if hint else '')+'\n'+options+'\nExplain your reasoning, then put the final option letter in \\boxed{}, for example \\boxed{A}.'
            if sha(text.encode())!=entry['prompt_sha256']:raise ValueError('prompt identity')
            selected[position]={'data_source':'scienceqa_image_mc','prompt':[{'role':'user','content':text}],
                                'images':[{'bytes':row['image']['bytes'],'path':None}],'ability':'science',
                                'reward_model':{'style':'rule','ground_truth':chr(65+int(row['answer']))},
                                'extra_info':{'split':split,'index':entry['original_index'],'num_choices':len(row['choices'])}}
    if any(row is None for row in selected):raise ValueError('missing selected row')
    target=a.output/f'{split}.parquet'
    row_count=len(selected)
    table=pa.Table.from_pylist(selected)
    del selected
    def verify_output(path):
        # Parquet's writer version is embedded in its bytes, so compare the
        # rows and order as well as recording the actual file digest.
        digest=file_sha(path)
        if path.stat().st_size==s['bytes'] and digest==s['sha256']:
            return digest
        offset=0
        for batch in pq.ParquetFile(path, pre_buffer=False).iter_batches(batch_size=8, use_threads=False):
            actual=pa.Table.from_batches([batch])
            if not actual.equals(table.slice(offset, batch.num_rows)):
                raise ValueError('prepared parquet content or order mismatch')
            offset+=batch.num_rows
        if offset!=row_count:raise ValueError('prepared parquet row count')
        return digest
    if target.exists():
        digest=verify_output(target)
    else:
        temp=target.with_suffix('.parquet.partial')
        pq.write_table(table,temp,row_group_size=32)
        digest=verify_output(temp)
        temp.replace(target)
    report[split]={'count':row_count,'bytes':target.stat().st_size,'sha256':digest,
                   'recorded_sha256':s['sha256'],'matches_recorded_sha256':digest==s['sha256'],
                   'pyarrow_version':pa.__version__}
print(json.dumps(report,indent=2))
