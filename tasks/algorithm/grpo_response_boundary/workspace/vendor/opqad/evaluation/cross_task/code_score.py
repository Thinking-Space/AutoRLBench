"""Only executed inside the isolated, networkless CPU scoring sandbox."""
import json,os,sys,warnings
from pathlib import Path
S=Path(__file__).resolve().parent
H=Path(os.environ["OPQAD_EVAL_WORKDIR"]).expanduser().resolve()

def sanitize_solution(item):
    from evalplus.sanitize import sanitize
    row,entrypoint=item
    # ast.parse repeatedly visits overlapping line spans. Invalid escape
    # sequences can emit the same non-fatal warning millions of times.
    # Suppress diagnostics only while sanitizing; syntax decisions and test
    # execution are unchanged.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        solution=sanitize(row['response'].rsplit('</think>',1)[-1],entrypoint=entrypoint)
    return dict(task_id=row['id'],solution=solution)

def main():
    if os.environ.get('OPQAD_SCORING_SANDBOX') != '1':
        raise SystemExit('Use score.py: generated code must run inside its bubblewrap sandbox')
    arm,bench=sys.argv[1:3];out=H/'outputs'/arm
    if bench=='selftest':
        from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
        from evalplus.data import get_human_eval_plus
        from evalplus.eval import untrusted_check
        from evalplus.gen.util import trusted_exec
        sample=dict(input_output=json.dumps(dict(inputs=['2\n'],outputs=['4\n'],fn_name=None)))
        good,_=check_correctness(sample,'print(int(input())*2)',6,False)
        bad,_=check_correctness(sample,'print(9)',6,False)
        assert all(v is True or v==1 for v in good) and not all(v is True or v==1 for v in bad),(good,bad)
        r=next(iter(get_human_eval_plus().values()));code=r['prompt']+r['canonical_solution']
        inputs=r['plus_input'][:5]
        expected,times=trusted_exec(code,inputs,r['entry_point'],record_time=True)
        res=untrusted_check('humaneval',code,inputs,r['entry_point'],expected=expected,atol=r['atol'],ref_time=times,fast_check=True)
        assert res[0]=='pass',res
        assert not Path('/home/.ssh').exists()
        print('SANDBOX_LCB_CORRECT_WRONG_AND_HUMANEVAL_REFERENCE_PASS',flush=True)
        return
    rows=[json.loads(l) for l in (out/f'{bench}.jsonl').open()]
    if bench=='humaneval':
        from evalplus.sanitize import sanitize
        from evalplus.evaluate import evaluate
        refs={r['id']:r for r in json.loads((H/'data/humaneval.json').read_text())}
        sf=out/'humaneval_solutions.jsonl'
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=8,initializer=os.chdir,initargs=('/tmp',)) as pool:
            solutions=[]
            for i,solution in enumerate(pool.map(sanitize_solution,[(r,refs[r['id']]['entry_point']) for r in rows]),1):
                solutions.append(solution)
                if i%32==0:print('SANITIZED',arm,i,'/',len(rows),flush=True)
        sf.write_text(''.join(json.dumps(r)+'\n' for r in solutions))
        evaluate('humaneval',samples=str(sf),parallel=8,test_details=True)
        result=json.loads(sf.with_name('humaneval_solutions_eval_results.json').read_text())
        samples=[r for rr in result['eval'].values() for r in rr]
        assert len(samples)==328
        passed=sum(r['base_status']=='pass' and r['plus_status']=='pass' for r in samples)
        summary=dict(correct=passed,accuracy=passed/len(samples),metric='HumanEval+ pass@1 (avg2)')
    else:
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
        from lcb_runner.utils.extraction_utils import extract_code
        from lcb_runner.lm_styles import LMStyle
        problems=[CodeGenerationProblem(**json.loads(l)) for l in (H/'data/livecodebench_v6.jsonl').open()]
        byid={}
        for r in rows:byid.setdefault(r['id'],[]).append(r)
        generations=[[extract_code(r['response'].rsplit('</think>',1)[-1],LMStyle.OpenAIChat) for r in sorted(byid[str(p.question_id)],key=lambda r:r['sample'])] for p in problems]
        assert len(generations)==175 and all(len(x)==2 for x in generations)
        result=codegen_metrics([p.get_evaluation_sample() for p in problems],generations,k_list=[1],num_process_evaluate=8)
        (out/'lcbv6.official_results.json').write_text(json.dumps(result,indent=2))
        passed=sum(bool(tests) and all(v is True or (isinstance(v,(int,float)) and v>0) for v in tests) for samples in result[1].values() for tests in samples)
        summary=dict(correct=passed,accuracy=passed/350,metric='LiveCodeBench v6 pass@1 (avg2)',official_metrics=result[0])
    (out/f'{bench}.code_summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__':main()
