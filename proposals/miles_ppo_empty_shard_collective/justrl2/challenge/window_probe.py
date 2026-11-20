#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Private evaluator for the exact native CPU/Gloo candidate source."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load_runner(evaluator):
    evaluator = Path(evaluator).resolve()
    path = evaluator / 'run_window.py'
    sys.path.insert(0, str(evaluator))
    # Never reuse a solver-side module already imported by the host process.
    sys.modules.pop('runtime_support', None)
    spec = importlib.util.spec_from_file_location('private_window_executor', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def observe(runner, rank):
    trace = dict(rank=rank, rows=[], reports=[])
    original_load = runner.runtime._load_native
    current = None
    ordinal = 0

    def arrays(parts):
        return [part.detach().cpu().tolist() for part in parts]

    def load(source, namespace):
        nonlocal current, ordinal
        result = original_load(source, namespace)
        loss = sys.modules['miles.backends.training_utils.loss']
        native_compute = result['compute_advantages_and_returns']
        native_gae, native_whiten = loss.get_advantages_and_returns_batch, loss.distributed_masked_whiten
        native_policy, native_report = result['compute_policy_loss'], result['aggregate_train_losses']

        def compute(settings, batch):
            nonlocal current
            current = dict(update=len(trace['rows']), local_values=arrays(batch['values']),
                           whitening_called=False, policy_losses=[], clip_flags=[])
            trace['rows'].append(current)
            try:
                returned = native_compute(settings, batch)
                current.update(advantages=arrays(batch['advantages']), returns=arrays(batch['returns']))
                return returned
            except Exception as exc:
                current['error'] = dict(type=type(exc).__name__, message=str(exc))
                raise

        def gae(*args, **kwargs):
            returned = native_gae(*args, **kwargs)
            current.update(raw_advantages=arrays(returned[0]), raw_returns=arrays(returned[1]))
            return returned

        def whiten(values, mask, process_group=None, **kwargs):
            nonlocal ordinal
            dist = runner.torch.distributed
            original_all_reduce = dist.all_reduce
            participation = {'called': False}
            reducer = native_whiten.__globals__.get('_moments')
            origins = getattr(reducer, '_origins', {})
            identity = origins.get((values.device, runner.torch.float32)) if hasattr(origins, 'get') else None
            current['identity_before'] = identity.detach().cpu().tolist() if identity is not None else [0.0, 0.0, 0.0]

            def record_collective(tensor, *args, **call_kwargs):
                participation['called'] = True
                current['collective_input'] = tensor.detach().cpu().tolist()
                result = original_all_reduce(tensor, *args, **call_kwargs)
                current['collective_output'] = tensor.detach().cpu().tolist()
                return result

            dist.all_reduce = record_collective
            try:
                returned = native_whiten(values, mask, process_group=process_group, **kwargs)
            finally:
                dist.all_reduce = original_all_reduce
            if participation['called']:
                current.update(whitening_called=True, ordinal=ordinal,
                    whitening_members=dist.get_process_group_ranks(process_group),
                    whitening_values=values.detach().cpu().tolist(),
                    whitening_mask=mask.detach().cpu().tolist(),
                    whitening_output=returned.detach().cpu().tolist())
                origins = getattr(native_whiten.__globals__.get('_moments'), '_origins', {})
                identity = origins.get((values.device, runner.torch.float32)) if hasattr(origins, 'get') else None
                current['identity_after'] = identity.detach().cpu().tolist() if identity is not None else current['identity_before']
                ordinal += 1
            return returned

        def policy(*args, **kwargs):
            returned = native_policy(*args, **kwargs)
            current['policy_losses'].append(returned[0].detach().cpu().tolist())
            current['clip_flags'].append(returned[1].detach().cpu().tolist())
            return returned

        def report(*args, **kwargs):
            returned = native_report(*args, **kwargs)
            trace['reports'].append(returned)
            return returned

        result['compute_advantages_and_returns'] = compute
        loss.get_advantages_and_returns_batch = gae
        loss.distributed_masked_whiten = whiten
        result['compute_policy_loss'] = policy
        result['aggregate_train_losses'] = report
        return result

    runner.runtime._load_native = load
    return trace


def run(source, output, config=None, evaluator=None):
    evaluator = Path(evaluator or HERE / 'window_evaluator').resolve()
    runner = load_runner(evaluator)
    # The private launcher executes this file in each worker. It never imports
    # the contestant's runner or runtime loader.
    runner.__file__ = str(Path(__file__).resolve())
    fixture_root = evaluator / 'inputs'
    result = runner.run_updates(source, output, config, evaluator=evaluator, fixture_root=fixture_root)
    if result['status'] == 'completed':
        try:
            from window_reference import validate_trace
            fixture = json.loads((fixture_root / 'replay.json').read_text())
            config = runner.validate_config(dict(runner.default_config() if config is None else config))
            traces = {str(rank): json.loads((Path(output) / f'private-{rank}.json').read_text())
                      for rank in range(config['cp'] * config['dp'])}
            metrics = json.loads((Path(output) / 'metrics.json').read_text())
            audit = validate_trace(fixture, config, traces, metrics)
            result['independent_validation'] = 'passed' if audit['passed'] else 'failed'
            result['exit_code'] = int(result['independent_validation'] != 'passed')
            result['independent_errors'] = audit['errors']
        except Exception as exc:
            result.update(status='infrastructure_error', exit_code=2,
                          first_error=dict(operation='independent_validation', type=type(exc).__name__, message=str(exc)))
    (Path(output) / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--evaluator', type=Path, default=HERE / 'window_evaluator')
    parser.add_argument('--_worker', type=int)
    parser.add_argument('--_scratch', type=Path)
    parser.add_argument('--_evaluator', type=Path)
    parser.add_argument('--_fixture-root', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else None
    if args._worker is not None:
        evaluator = args._evaluator.resolve()
        runner = load_runner(evaluator)
        trace = observe(runner, args._worker)
        code = runner._worker(args.source, args.output, args._scratch, config,
                              args._worker, args._fixture_root)
        trace['code'] = code
        (args.output / f'private-{args._worker}.json').write_text(json.dumps(trace, allow_nan=False))
        sys.exit(code)
    result = run(args.source, args.output, config, args.evaluator)
    print(json.dumps(result))
    sys.exit(result['exit_code'])
