"""Tiny synthetic rows test trusted resolution and sample identities."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import os
from unittest.mock import patch


def module(name):
    path = Path(__file__).with_name(name + '.py')
    assert path.exists(), 'Missing incident loader contract'
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


class LoaderContract(unittest.TestCase):
    def test_missing_original_path_fails_but_catalog_repair_reads_identical_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = {'id': 'hard-00000001', 'prompt': [{'role':'user','content':'Synthetic 1+1?'}],
                       'label':'2', 'metadata':{'difficulty':'hard'}}
            raw = json.dumps(payload).encode()
            (root/'moved').mkdir();(root/'moved/item.json').write_bytes(raw)
            record = {'id':payload['id'], 'path':'old/item.json', 'sha256':hashlib.sha256(raw).hexdigest(), 'difficulty':'hard'}
            with self.assertRaises(FileNotFoundError):
                module('loader_contract').read_resolved(root, record, lambda r,x:x['path'])
            actual,path = module('loader_contract').read_resolved(root,record,lambda r,x:'moved/item.json')
            self.assertEqual(actual,payload)
            self.assertEqual(path,'moved/item.json')

    def test_replacing_sample_with_another_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'other.json').write_text('{"id":"other","label":"2"}')
            record={'id':'original','path':'missing.json','sha256':'0'*64,'difficulty':'hard'}
            with self.assertRaises(ValueError):module('loader_contract').read_resolved(root,record,lambda r,x:'other.json')

    def test_expected_schedule_uses_cursor_not_difficulty_proportion(self):
        m=module('loader_contract')
        rows=[{'id':str(i),'difficulty':'easy' if i%3 else 'hard'} for i in range(7)]
        plan=m.expected_groups(rows,5,2,4,7)
        self.assertEqual(plan[4],['5','6'])
        self.assertEqual(plan[5],['0','1'])
        self.assertEqual(plan[7],['4','5'])

    def test_subprocess_retry_reads_same_id_and_reserves_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);record={'id':'one','difficulty':'easy','path':'one.json'}
            payload={'id':'one','difficulty':'easy','prompt':[{'role':'user','content':'Synthetic'}],
                     'label':'1','metadata':{'difficulty':'easy'}}
            (root/'one.json').write_text(json.dumps(payload));record['sha256']=hashlib.sha256((root/'one.json').read_bytes()).hexdigest()
            (root/'manifest.json').write_text(json.dumps({'rows':[record]}))
            loader=root/'loader.py';state=root/'resolver-count'
            loader.write_text('from pathlib import Path\n'
                'def resolve(root, record):\n'
                f'    state=Path({str(state)!r})\n'
                '    n=int(state.read_text()) if state.exists() else 0\n'
                '    state.write_text(str(n+1))\n'
                '    print("loader diagnostic")\n'
                '    if n<2:raise FileNotFoundError("temporary")\n'
                '    return record["path"]\n')
            import sys
            with patch.dict(os.environ,{'RELOCATION_LOADER':str(loader),'RELOCATION_OUTPUT':str(root/'out')}),patch.object(sys,'path',[str(Path(__file__).parent),*sys.path]):
                store=module('store').SceneStore(root)
                self.assertEqual(store.read(record,sample_index=64,step=4),payload)
            attempts=[json.loads(l) for l in (root/'out/read_attempts.jsonl').read_text().splitlines()]
            self.assertEqual([e['outcome'] for e in attempts],['missing','missing','ok'])
            self.assertEqual({e['id'] for e in attempts},{'one'})

    def test_resolver_program_error_is_not_reported_as_missing_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'manifest.json').write_text('{"rows":[]}')
            loader=root/'bad.py';loader.write_text('def resolve(root, record):\n    raise ValueError("ordinary bug")\n')
            import sys
            with patch.dict(os.environ,{'RELOCATION_LOADER':str(loader)}),patch.object(sys,'path',[str(Path(__file__).parent),*sys.path]):
                with self.assertRaisesRegex(RuntimeError,'ValueError'):
                    module('store').SceneStore(root).read({'id':'one','path':'old','sha256':'0'*64})


if __name__=='__main__':unittest.main()
