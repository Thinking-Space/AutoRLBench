import copy,importlib.util,json,os,shutil,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
INDEX=json.loads((ROOT/'benchmark-index.json').read_text())

def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]

class SubmissionTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.results=Path(self.tmp.name);self.logs=self.results/'experiment_logs';self.logs.mkdir();(self.logs/'fresh_eval').mkdir()
 def tearDown(self):self.tmp.cleanup()
 def fixture(self,task,reference=False):
  w=ROOT/'shared_baselines/task2_repaired_grpo/workspace' if reference else task/'workspace'
  shutil.copy(w/'outputs/metrics.jsonl',self.logs/'metrics.jsonl')
  source=w/'outputs/frozen_eval/150.jsonl'
  if not source.exists():source=w/'outputs/validation/150.jsonl'
  shutil.copy(source,self.logs/'fresh_eval/150.jsonl')
 def grader(self,task):
  tests=task/'evaluations/solution/verifier/tests';sys.path.insert(0,str(tests));module=load(tests/'grade_submission.py','grader_'+task.name);resources=tests.parent/'resources';return module,json.loads((resources/'grader_spec.json').read_text()),json.loads((resources/'eval_items.json').read_text())
 def test_all_seven_actual_candidates_fail_and_shared_reference_passes(self):
  for row in INDEX['tasks']:
   task=ROOT/row['path'];g,spec,items=self.grader(task)
   self.fixture(task);actual=g.evaluate(self.results,spec,items,replay=True);self.assertFalse(actual['passed'],(task,actual))
   self.assertEqual(actual['errors'],['Metric outside accepted interval'],(task,actual))
   self.fixture(task,reference=True);actual=g.evaluate(self.results,spec,items,replay=True);self.assertTrue(actual['passed'],(task,actual))
 def setup_reference(self):
  task=ROOT/'tasks/reward/verl_batch_reward_broadcast';self.fixture(task,reference=True);return self.grader(task)
 def test_missing_final_step_rejected(self):
  g,s,i=self.setup_reference();xs=rows(self.logs/'metrics.jsonl');(self.logs/'metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in xs[:-1]));self.assertFalse(g.evaluate(self.results,s,i,replay=True)['passed'])
 def test_duplicate_metrics_rejected(self):
  g,s,i=self.setup_reference();p=self.logs/'metrics.jsonl';p.write_text(p.read_text()+p.read_text().splitlines()[-1]+'\n');self.assertFalse(g.evaluate(self.results,s,i,replay=True)['passed'])
 def test_nonfinite_metric_rejected(self):
  g,s,i=self.setup_reference();xs=rows(self.logs/'metrics.jsonl');xs[-1]['data']['actor/grad_norm']=float('nan');(self.logs/'metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in xs));self.assertFalse(g.evaluate(self.results,s,i,replay=True)['passed'])
 def test_missing_final_answers_rejected(self):
  g,s,i=self.setup_reference();(self.logs/'fresh_eval/150.jsonl').unlink();self.assertFalse(g.evaluate(self.results,s,i,replay=True)['passed'])
 def test_unknown_question_rejected(self):
  g,s,i=self.setup_reference();p=self.logs/'fresh_eval/150.jsonl';xs=rows(p);xs[0]['problem_id']='invented';p.write_text('\n'.join(json.dumps(x) for x in xs));self.assertFalse(g.evaluate(self.results,s,i,replay=True)['passed'])
 def test_tampered_logged_accuracy_rejected(self):
  g,s,i=self.setup_reference();p=self.logs/'metrics.jsonl';xs=rows(p);xs[-1]['data']['val-core/aime24/acc/mean@32']=.99;p.write_text('\n'.join(json.dumps(x) for x in xs));self.assertFalse(g.evaluate(self.results,s,i,replay=True)['passed'])
 def test_missing_report_rejected_in_real_protocol(self):
  g,s,i=self.setup_reference();x=g.evaluate(self.results,s,i,workspace=self.results/'missing-workspace');self.assertFalse(x['passed']);self.assertIn('report.md',x['errors'][0])
 def test_real_protocol_cannot_be_enabled_by_historical_replay_flag(self):
  task=ROOT/'tasks/reward/verl_batch_reward_broadcast';text=(task/'evaluations/solution/verifier/tests/grade_submission.py').read_text();self.assertNotIn("add_argument('--replay",text)

if __name__=='__main__':unittest.main()
