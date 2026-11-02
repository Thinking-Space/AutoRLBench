"""Refresh maintainer-only inventories for the two packaged review cases."""
import hashlib
import json
from pathlib import Path


def refresh(repo, task):
    path = task / 'provenance/source_manifest.json'
    current = json.loads(path.read_text()) if path.exists() else {'schema_version': 2}
    previous = {r['destination']: r for r in current.get('files', [])}
    rows = []
    for item in sorted(task.rglob('*')):
        if not item.is_file() or item == path or '__pycache__' in item.parts or item.suffix == '.pyc':
            continue
        name = item.relative_to(task).as_posix()
        prior = name.replace('provenance/', 'environment/', 1) if name.startswith('provenance/') else name
        row = dict(previous.get(name, previous.get(prior, {
            'source': None, 'before_sha256': None, 'scope': 'task_authored',
            'changes': ['Diagnosis-first task packaging or maintainer provenance.']})))
        row['destination'] = name
        row['after_sha256'] = hashlib.sha256(item.read_bytes()).hexdigest()
        if row.get('source'):
            row['before_sha256'] = hashlib.sha256((repo / row['source']).read_bytes()).hexdigest()
        if name == 'workspace/loader.py' and task.name.endswith('_control'):
            row.update(source='proposals/justrl2_data_relocation/incident/reference_loader.py',
                       before_sha256=hashlib.sha256((repo / 'proposals/justrl2_data_relocation/incident/reference_loader.py').read_bytes()).hexdigest(),
                       changes=['Replace maintainer-only docstring with neutral resolver documentation; executable AST unchanged.'])
        rows.append(row)
    current.update(files=rows, historical_manifest='provenance/history/from_base_source_manifest.json',
                   inventory_scope='All packaged files except this maintainer-only inventory.',
                   snapshot_sha256=hashlib.sha256((task / 'environment/snapshot.json').read_bytes()).hexdigest(),
                   protocol='assess_then_optionally_repair')
    path.write_text(json.dumps(current, indent=2) + '\n')
    return len(rows)


if __name__ == '__main__':
    repo = Path(__file__).resolve().parents[3]
    for name in ('justrl2_data_relocation', 'justrl2_data_review_control'):
        print(name, refresh(repo, repo / 'tasks/data' / name))
