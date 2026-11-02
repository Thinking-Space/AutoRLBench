"""Trusted bench-side data checks. The editable loader chooses only a path."""
import hashlib
import json
from pathlib import Path


def read_resolved(root, record, resolve):
    root = Path(root).resolve()
    relative = resolve(str(root), dict(record))
    if not isinstance(relative, str):
        raise ValueError('Loader must return a relative file path')
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Resolved payload must stay inside the scene dataset')
    path = root / rel
    if not path.resolve().is_relative_to(root):
        raise ValueError('Resolved payload is outside the scene')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('Resolved content differs from requested logical sample')
    payload = json.loads(raw)
    if payload.get('id') != record['id'] or payload.get('metadata',{}).get('difficulty') != record['difficulty']:
        raise ValueError('Resolved sample identity differs')
    if not isinstance(payload.get('prompt'),list) or 'label' not in payload:
        raise ValueError('Invalid original math payload')
    return payload, relative


def expected_groups(rows, cursor, batch_size, first_step, final_step):
    if not rows or cursor < 0 or batch_size <= 0 or final_step < first_step:
        raise ValueError('Invalid logical sampling plan')
    return {step:[rows[(cursor+(step-first_step)*batch_size+i)%len(rows)]['id']
                  for i in range(batch_size)] for step in range(first_step,final_step+1)}
