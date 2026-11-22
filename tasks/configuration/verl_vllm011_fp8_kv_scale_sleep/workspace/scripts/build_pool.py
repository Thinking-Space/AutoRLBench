import pandas as pd, json, re, random
R="/home/ben/task/raw/"
rows=[]
def isint(a):
    a=str(a).strip().replace(",","")
    return re.fullmatch(r"-?\d+", a) is not None
s=pd.read_parquet(R+"Skywork-OR1-RL-Data/data/math-00000-of-00001.parquet")
for _,r in s.iterrows():
    g=json.loads(r.reward_model['ground_truth'])
    md=r.extra_info['model_difficulty']
    if len(g)==1 and isint(g[0]):
        rows.append(dict(problem=r.prompt[0]['content'].strip(), answer=g[0].strip().replace(",",""), src="skywork:"+r.data_source,
                         diff=md['DeepSeek-R1-Distill-Qwen-7B']/16, d32=md['DeepSeek-R1-Distill-Qwen-32B']/16))
p=pd.read_json(R+"Polaris-Dataset-53K/polaris-data-53K.jsonl",lines=True)
for _,r in p.iterrows():
    if isint(r.answer):
        n,dn=r.difficulty.split("/"); rows.append(dict(problem=r.problem.strip(), answer=str(r.answer).strip().replace(",",""), src="polaris", diff=int(n)/int(dn), d32=None))
d=pd.read_parquet(R+"DAPO-Math-17k/data/dapo-math-17k.parquet").drop_duplicates(subset=None if False else ["extra_info"]) if False else pd.read_parquet(R+"DAPO-Math-17k/data/dapo-math-17k.parquet")
d['q']=d.prompt.apply(lambda x:x[0]['content']); d=d.drop_duplicates('q')
pre="Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
suf='\n\nRemember to put your answer on its own line after "Answer:".'
for _,r in d.iterrows():
    q=r.q
    assert q.startswith(pre) and q.endswith(suf)
    q=q[len(pre):-len(suf)]
    a=r.reward_model['ground_truth']
    if isint(a): rows.append(dict(problem=q.strip(), answer=str(a).strip(), src="dapo", diff=None, d32=None))
df=pd.DataFrame(rows)
print(len(df), df.src.value_counts().to_dict())
df['key']=df.problem.str.lower().str.replace(r"\s+","",regex=True)
df=df.drop_duplicates('key')
print("dedup", len(df))
df[['problem','answer','src','diff','d32']].to_json("/home/ben/task/data/pool_all.jsonl",orient="records",lines=True)
df[['problem']].rename(columns={'problem':'text'}).to_json("/home/ben/task/data/pool_text.jsonl",orient="records",lines=True)
