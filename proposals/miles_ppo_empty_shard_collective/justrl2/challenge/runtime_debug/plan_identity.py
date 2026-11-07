#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Maintainer check of candidate routing with distinct native communicators."""
import argparse
from datetime import timedelta
import importlib.util
import json
from pathlib import Path

import torch
import torch.distributed as dist


def check(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {'passed': False, 'scope': 'native CPU/Gloo group identity, same members and same payload'}
    original = dist.all_reduce
    try:
        spec = importlib.util.spec_from_file_location('candidate_plan', source)
        module = importlib.util.module_from_spec(spec)
        dist.init_process_group('gloo', init_method=(output / 'store').as_uri(),
                                world_size=1, rank=0, timeout=timedelta(seconds=5))
        groups = [dist.new_group([0]), dist.new_group([0])]
        observed, expected = [], []

        def recording(tensor, *args, **kwargs):
            group = kwargs.get('group', args[1] if len(args) > 1 else None)
            observed.append(next((i for i, g in enumerate(groups) if g is group), None))
            return original(tensor, *args, **kwargs)

        dist.all_reduce = recording
        # Observe aliases captured during module import as well as attribute calls.
        spec.loader.exec_module(module)
        module.clear()
        for shape in ((3,), (1, 3)):
            for i, group in enumerate(groups):
                value = torch.tensor([2.0, 4.0, 6.0]).reshape(shape)
                before = value.clone()
                module.reduce_sum(value, group=group)
                expected.append(i)
                if not torch.equal(value, before):
                    raise ValueError('Single-rank SUM changed values')
        result.update(passed=observed == expected, requested=expected, effective=observed)
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        dist.all_reduce = original
        if dist.is_initialized():
            dist.destroy_process_group()
    (output / 'identity.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='candidate collective_plan.py')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.source, args.output)
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)
