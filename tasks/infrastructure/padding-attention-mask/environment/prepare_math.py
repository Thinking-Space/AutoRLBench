"""Prepare genuine DAPO math training rows and fixed AIME24 evaluation rows."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

PREFIX='Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n'
SUFFIXES=['\n\nRemember to put your answer on its own line after "Answer:".',
          '\n\nPlease reason step by step, and put your final answer within \\boxed{}.']


def normalize_problem(value):
    if isinstance(value,list):
        messages=[item['content'] for item in value if item.get('role')=='user']
        if len(value)!=1 or len(messages)!=1:raise ValueError('Expected exactly one mathematical user question, without dropping other roles')
        value=messages[0]
    if not isinstance(value,str) or not value.strip():raise ValueError('Missing mathematical question')
    value=value.strip().removeprefix(PREFIX)
    while True:
        before=value
        for suffix in SUFFIXES:value=value.removesuffix(suffix)
        if value==before:break
    return value.strip()


def normalize_answer(value):
    if value is None:raise ValueError('Missing independent mathematical answer')
    value=str(value).strip()
    boxed=re.fullmatch(r'\\boxed\{\s*([-+]?\d+)\s*\}',value)
    if boxed:value=boxed[1]
    if not re.fullmatch(r'[-+]?\d+',value):raise ValueError('Expected an exact integer ground truth, got '+repr(value))
    return str(int(value))


def normalize_record(record):
    problem=record.get('problem',record.get('question',record.get('prompt')))
    answer=record.get('answer')
    if answer is None:
        reward=record.get('reward_model')
        if isinstance(reward,str):reward=json.loads(reward)
        if isinstance(reward,dict):answer=reward.get('ground_truth')
    problem=normalize_problem(problem);answer=normalize_answer(answer)
    identity=hashlib.sha256((problem+'\0'+answer).encode()).hexdigest()[:24]
    return dict(id=identity,problem=problem,answer=answer)


def format_problem(problem):
    return problem+'\n\nPlease reason step by step, and put your final answer within \\boxed{}.'


def encode_problem(tokenizer,problem,valid_vocab=None):
    ids=tokenizer.apply_chat_template([dict(role='user',content=format_problem(problem))],
                                      tokenize=True,add_generation_prompt=True,return_dict=False)
    if valid_vocab is None:valid_vocab=len(tokenizer)
    if not isinstance(ids,list) or not ids or any(type(x) is not int or x<0 or x>=valid_vocab for x in ids):
        raise ValueError('Chat template must produce valid integer token IDs')
    return ids


def parquet_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
        yield from batch.to_pylist()


def load_examples(path):
    path=Path(path)
    if path.suffix=='.jsonl':rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    elif path.suffix=='.parquet':rows=list(parquet_rows(path))
    else:rows=json.loads(path.read_text())
    return [normalize_record(row) for row in rows]


def prepare(dapo_path,aime_path,model_path,output,seed=42,max_prompt_tokens=1024):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    valid_vocab=len(tokenizer)
    evaluation=load_examples(aime_path)
    if len(evaluation)!=30 or len({row['id'] for row in evaluation})!=30:
        raise ValueError('Pinned AIME24 must contain exactly30 distinct questions')
    evaluation.sort(key=lambda row:row['id'])
    eval_questions={re.sub(r'\s+',' ',row['problem']).strip() for row in evaluation}
    unique={};raw_rows=0;raw_unique=set()
    for record in parquet_rows(dapo_path):
        raw_rows+=1
        row=normalize_record(record)
        raw_problem=record.get('problem',record.get('question',record.get('prompt')))
        if isinstance(raw_problem,list):raw_problem=raw_problem[0]['content']
        raw_answer=record.get('answer')
        if raw_answer is None:
            reward=record.get('reward_model')
            if isinstance(reward,str):reward=json.loads(reward)
            raw_answer=reward['ground_truth']
        raw_unique.add((raw_problem,str(raw_answer)))
        unique.setdefault((row['problem'],row['answer']),row)
    retained=[];overlong=0;overlap=0
    for row in unique.values():
        if re.sub(r'\s+',' ',row['problem']).strip() in eval_questions:
            overlap+=1;continue
        if len(encode_problem(tokenizer,row['problem'],valid_vocab))>max_prompt_tokens:
            overlong+=1;continue
        retained.append(row)
    random.Random(seed).shuffle(retained)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('train.jsonl',retained),('aime24.jsonl',evaluation)]:
        (output/name).write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in rows))
    manifest=dict(schema_version=1,training='BytedTsinghua-SIA/DAPO-Math-17k',
        training_revision='65877096c24ffa7abc4e4fa5edb95cf3413a5674',
        evaluation='HuggingFaceH4/aime_2024',evaluation_revision='58e32b4b387f34192acdacbdf602510c459a6525',
        raw_training_rows=raw_rows,unique_question_answer_pairs=len(unique),
        exact_raw_question_answer_pairs=len(raw_unique),normalization_merged_pairs=len(raw_unique)-len(unique),
        prompt_overflow_removed=overlong,exact_eval_question_overlap_removed=overlap,
        training_rows=len(retained),evaluation_questions=len(evaluation),seed=seed,
        prompt_token_limit=max_prompt_tokens,normalization='exact independent integer GT; original user question; no solution text',
        prompt_template='native model chat template, generation prompt enabled, fixed boxed-answer instruction',
        files={name:{'sha256':hashlib.sha256((output/name).read_bytes()).hexdigest(),
                     'size_bytes':(output/name).stat().st_size} for name in ['train.jsonl','aime24.jsonl']})
    if not retained:raise ValueError('No DAPO training rows survived preparation')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dapo',required=True);parser.add_argument('--aime',required=True)
    parser.add_argument('--model',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--max-prompt-tokens',type=int,default=1024)
    args=parser.parse_args();print(json.dumps(prepare(args.dapo,args.aime,args.model,args.output,args.seed,args.max_prompt_tokens),indent=2))
