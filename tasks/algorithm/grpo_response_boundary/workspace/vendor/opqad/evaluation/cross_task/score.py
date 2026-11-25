import argparse,dataclasses,fcntl,hashlib,json,os,subprocess,sys
from pathlib import Path
S=Path(__file__).resolve().parent
H=Path(os.environ["OPQAD_EVAL_WORKDIR"]).expanduser().resolve()
N=H
M=S

def final(text):return text.rsplit('</think>',1)[-1].strip()

def sandbox(arm,bench):
    out=H/'outputs'/arm
    env=Path(sys.executable).parent.parent
    cmd=['/usr/bin/bwrap','--die-with-parent','--new-session','--unshare-all']
    for path in ['/usr','/lib','/lib64','/bin']:
        if Path(path).exists():cmd+=['--ro-bind',path,path]
    for path in [str(env),str(H),str(S)]:cmd+=['--ro-bind',path,path]
    cmd+=['--tmpfs','/cache','--dir','/cache/evalplus']
    for cached in (N/'data/cache/evalplus').iterdir():
        if cached.is_file():cmd+=['--ro-bind',str(cached),'/cache/evalplus/'+cached.name]
    cmd+=['--bind',str(out),str(out),'--dev','/dev','--proc','/proc','--tmpfs','/tmp','--dir','/home','--chdir',str(H),
          '--setenv','OPQAD_SCORING_SANDBOX','1','--setenv','OPQAD_EVAL_WORKDIR',str(H),'--setenv','NLTK_DATA',str(H/'data/nltk'),'--setenv','HOME','/tmp','--setenv','XDG_CACHE_HOME','/cache','--setenv','PATH',str(env/'bin')+':/usr/bin:/bin',
          '--setenv','PYTHONPATH',str(S/'vendor')+':'+str(S/'vendor/LiveCodeBench'),'--setenv','OPENBLAS_NUM_THREADS','1','--setenv','OMP_NUM_THREADS','1',
          str(env/'bin/python'),str(S/'code_score.py'),arm,bench]
    subprocess.run(cmd,check=True,env={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'})

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('arm');ap.add_argument('--bench')
    ap.add_argument('--num-shards',type=int,choices=(2,4),default=2)
    args=ap.parse_args();arm=args.arm;out=H/'outputs'/arm
    p=json.loads((H/'manifest.json').read_text())['protocol']
    if args.bench is None:
        for s in range(args.num_shards):assert (out/f'shard{s}/GENERATION_DONE').exists()
    else:assert args.bench in p['benches']
    summary={}
    for b,setting in p['benches'].items():
        if args.bench and b!=args.bench:continue
        lock=(out/f'.score-{b}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
        reference_bytes=(H/f'data/{b}.json').read_bytes()
        if b in p.get('data_hashes',{}):
            assert hashlib.sha256(reference_bytes).hexdigest()==p['data_hashes'][b],('Dataset changed',b)
        refs=json.loads(reference_bytes);ref={r['id']:r for r in refs}
        expected={(r['id'],i) for r in refs for i in range(setting['n'])};seen=set();rows=[]
        for s in range(args.num_shards):
            for r in map(json.loads,(out/f'shard{s}/{b}.jsonl').open()):
                key=(r['id'],r['sample']);assert key in expected and key not in seen
                seen.add(key);rows.append(r)
        assert seen==expected,(b,len(seen),len(expected))
        rows.sort(key=lambda r:(r['id'],r['sample']))
        merged=out/f'{b}.jsonl';tmp=merged.with_suffix('.tmp')
        prior_path=out/f'{b}.summary.json'
        if prior_path.exists() and merged.exists():
            prior=json.loads(prior_path.read_text())
            expected_digest=hashlib.sha256()
            for r in rows:expected_digest.update((json.dumps(r,ensure_ascii=False)+'\n').encode())
            with merged.open('rb') as f:actual_digest=hashlib.file_digest(f,'sha256').hexdigest()
            protocol_matches=True
            if b=='ifeval':
                from ifeval_grading import PROTOCOL
                protocol_matches=prior.get('grader_protocol')==PROTOCOL
            if actual_digest==expected_digest.hexdigest() and protocol_matches:
                assert prior['samples']==len(rows) and prior['questions']==len(ref)
                summary[b]=prior
                print('REUSED_VERIFIED_SCORE',arm,b,flush=True)
                fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
                continue
        with tmp.open('w') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        tmp.replace(merged)
        score=dict(samples=len(rows),questions=len(ref),truncated=sum(r['finish_reason']=='length' for r in rows),mean_response_tokens=sum(len(r['token_ids']) for r in rows)/len(rows))
        if b.startswith('aime'):
            sys.path[:0]=[str(S/'vendor/simplelr_math_eval'),str(S)]
            from parser import extract_answer
            from aime_grader import answers_equal
            graded=[]
            for r in rows:
                pred=extract_answer(final(r['response']),b)
                graded.append(dict(id=r['id'],sample=r['sample'],prediction=pred,answer=ref[r['id']]['answer'],correct=answers_equal(pred,ref[r['id']]['answer'])))
            score.update(correct=sum(r['correct'] for r in graded),accuracy=sum(r['correct'] for r in graded)/len(rows))
            (out/f'{b}.graded.json').write_text(json.dumps(graded,indent=2))
        elif b=='ifeval':
            from ifeval_grading import grade, PROTOCOL
            graded=[grade(ref[r['id']],r) for r in rows]
            score['grader_protocol']=PROTOCOL
            for label in ['strict','loose']:
                score['prompt_'+label]=sum(r[label]['follow_all_instructions'] for r in graded)/len(graded)
                score['instruction_'+label]=sum(sum(r[label]['follow_instruction_list']) for r in graded)/sum(len(r[label]['follow_instruction_list']) for r in graded)
            (out/f'{b}.graded.json').write_text(json.dumps(graded,indent=2))
        else:
            sandbox(arm,b)
            score.update(json.loads((out/f'{b}.code_summary.json').read_text()))
        (out/f'{b}.summary.json').write_text(json.dumps(score,indent=2));summary[b]=score
        print('SCORE',arm,b,score,flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if args.bench is None:
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
        (out/'DONE').write_text('All configured generations checked and scored\n')

if __name__=='__main__':main()
