"""Bounded GUI lookahead with ordered commits and no cross-policy result reuse.

Whole groups reserve episode slots atomically. Individual episode completion
(including its cleanup and retries) releases slots and wakes admission. Results
may arrive out of order, but only the first qualifying task-unique prefix is
trained. The speculative task suffix is returned to the datasource cursor.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer

logger = logging.getLogger(__name__)


async def drain_engine_requests(args):
    """A cancelled HTTP waiter is not proof that SGLang stopped decoding.

    Called only after all this rollout's episode coroutines have stopped, and
    before returning to the async driver. Local HTTP is required so no remote
    POST actor can enqueue an old request after this barrier. Do not flush the
    prefix cache just to inspect load.
    """
    import sglang_router
    from packaging.version import parse

    router = f'http://{args.sglang_router_ip}:{args.sglang_router_port}'
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        path = '/list_workers' if parse(sglang_router.__version__) <= parse('0.2.1') else '/workers'
        response = await client.get(router + path)
        response.raise_for_status()
        registry = response.json()
        urls = registry['urls'] if path == '/list_workers' else [w['url'] for w in registry['workers']]
        expected = args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        if len(set(urls)) != expected or expected < 1:
            raise RuntimeError(f'Cannot verify GUI rollout engines: {len(set(urls))} != {expected}')

        async def drain_one(url):
            response = await client.post(url + '/abort_request', json={'abort_all': True})
            response.raise_for_status()
            idle = 0
            while idle < 2:
                response = await client.get(url + '/get_load')
                response.raise_for_status()
                loads = response.json()
                if not isinstance(loads, list) or not loads or any('num_reqs' not in x for x in loads):
                    raise RuntimeError(f'Cannot verify idle SGLang engine {url}')
                idle = idle + 1 if all(x['num_reqs'] == 0 for x in loads) else 0
                if idle < 2:
                    await asyncio.sleep(.1)

        tasks = [asyncio.create_task(drain_one(url)) for url in set(urls)]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def generate_lookahead(args, rollout_id, source):
    from slime.rollout.sglang_rollout import GenerateState, generate_and_rm, generate_and_rm_group

    state = GenerateState(args)
    if state.pendings or state.remaining_batch_size or state.aborted:
        raise RuntimeError('Stale generation state before GUI lookahead')
    owner = getattr(source, '__self__', None)
    if not all(callable(getattr(owner, key, None)) for key in ('sampling_cursor', 'restore_sampling_cursor')):
        raise ValueError('GUI lookahead needs a datasource with speculative cursor support')
    if args.partial_rollout or args.group_rm or args.rollout_all_samples_process_path or args.rollout_sample_filter_path:
        raise ValueError('GUI lookahead requires complete groups and no extra sample/group hooks')
    if getattr(args, 'use_distributed_post', False):
        raise ValueError('GUI lookahead requires local HTTP for a reliable cancellation barrier')
    target, size = args.rollout_batch_size, args.n_samples_per_prompt
    limit, extra = args.rollout_max_groups, args.rollout_lookahead_groups
    capacity = args.rollout_max_active_episodes
    if not (target > 0 and size > 1 and limit >= target and 0 < extra <= limit-target and capacity >= size):
        raise ValueError('Invalid GUI lookahead bounds')
    audit = args._gui_dapo_audit
    args._gui_dapo_boundary_errors = []
    dropped = MetricGatherer()
    loop = asyncio.get_running_loop()
    started = loop.time()
    changed = asyncio.Event()
    active = set()
    pending = {}  # task -> candidate ordinal
    ready = {}  # ordinal -> (complete group, qualification row)
    cursors = []
    selected = []
    submitted = prefix = peak_active = peak_window = 0
    ready_at = None
    cleanup_ok = False
    engine_drain_s = 0.
    original_cursor = owner.sampling_cursor()
    progress_at = started

    async def run_group(ordinal, group):
        reservations = {(ordinal, i) for i in range(size)}
        token_by_sample = {id(sample): (ordinal, i) for i, sample in enumerate(group)}

        async def episode_runner(a, sample, params, evaluation=False):
            try:
                return await generate_and_rm(a, sample, params, evaluation=evaluation)
            finally:
                active.discard(token_by_sample[id(sample)])
                changed.set()

        try:
            return await generate_and_rm_group(args, group, state.sampling_params.copy(),
                                              episode_runner=episode_runner)
        finally:
            # Handles cancellation before some children entered episode_runner.
            # generate_and_rm_group has already awaited child lease cleanup.
            active.difference_update(reservations)
            changed.set()

    def window_size():
        return len(selected) + len(pending) + sum(row['qualified'] for _, row in ready.values())

    def launch():
        nonlocal submitted, peak_active, peak_window
        while submitted < limit and window_size() < target + extra and len(active) + size <= capacity:
            cursor = owner.sampling_cursor()
            groups = source(1)
            if len(groups) != 1 or len(groups[0]) != size or len({id(s) for s in groups[0]}) != size:
                raise RuntimeError('GUI datasource did not return one complete independent group')
            ordinal = submitted
            cursors.append(cursor)
            submitted += 1
            active.update((ordinal, i) for i in range(size))
            task = asyncio.create_task(run_group(ordinal, groups[0]))
            task.add_done_callback(lambda _: changed.set())
            pending[task] = ordinal
            state.pendings.add(task)
            peak_active = max(peak_active, len(active))
            peak_window = max(peak_window, window_size())

    def reap():
        for task in list(pending):
            if not task.done():
                continue
            ordinal = pending.pop(task)
            state.pendings.discard(task)
            group = task.result()  # Unexpected engine errors must fail the batch.
            row = audit.qualify(args, group)
            row['candidate_ordinal'] = ordinal
            ready[ordinal] = (group if row['qualified'] else None), row

    try:
        async with asyncio.timeout(args.rollout_timeout_s):
            while len(selected) < target:
                # Clearing before inspection prevents losing completion signals.
                changed.clear()
                reap()
                if args._gui_dapo_boundary_errors:
                    raise RuntimeError(f'GUI lookahead boundary failed: {args._gui_dapo_boundary_errors}')
                while prefix in ready and len(selected) < target:
                    group, row = ready.pop(prefix)
                    prefix += 1
                    result = audit.assign(row)
                    if result.keep:
                        selected.append(group)
                    else:
                        dropped.on_dynamic_filter_drop(result.reason)
                if len(selected) == target:
                    ready_at = loop.time()
                    break
                launch()
                if not pending:
                    raise RuntimeError(f'DAPO exhausted {submitted} candidate groups: accepted '
                                       f'{len(selected)}/{target}; refusing a partial update')
                if loop.time() - progress_at >= 10:
                    logger.info('GUI DAPO lookahead rollout=%s accepted=%s/%s submitted=%s/%s '
                                'active_episodes=%s ready_out_of_order=%s', rollout_id, len(selected),
                                target, submitted, limit, len(active), len(ready))
                    progress_at = loop.time()
                try:
                    await asyncio.wait_for(changed.wait(), timeout=10)
                except TimeoutError:
                    pass
    finally:
        cleanup_started = loop.time()
        state.aborted = True
        tasks = list(pending)
        for task in tasks:
            if not task.done():
                task.cancel()
        try:
            # Shield: a cleanup deadline must NOT cancel lease cleanup a second
            # time. Poison the state and fail instead of returning train data.
            drain = asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.wait_for(asyncio.shield(drain), timeout=args.rollout_cleanup_timeout_s)
            for task in tasks:
                if task.cancelled():
                    # A task cancelled before its first instruction has no
                    # finally block to release the pre-reserved slots.
                    active.difference_update((pending[task], i) for i in range(size))
                elif task.exception() is not None:
                    raise task.exception()
                if not task.cancelled() and task.exception() is None:
                    ordinal = pending[task]
                    row = audit.qualify(args, task.result())
                    row['candidate_ordinal'] = ordinal
            if active:
                raise RuntimeError(f'GUI lookahead leaked {len(active)} episode reservations')
            if submitted > prefix or ready_at is None:
                engine_start = loop.time()
                remaining = args.rollout_cleanup_timeout_s - (engine_start - cleanup_started)
                async with asyncio.timeout(max(0, remaining)):
                    await drain_engine_requests(args)
                engine_drain_s = loop.time() - engine_start
            if args._gui_dapo_boundary_errors:
                raise RuntimeError(f'GUI lookahead boundary failed: {args._gui_dapo_boundary_errors}')
            cleanup_ok = True
        except TimeoutError as exc:
            raise RuntimeError('GUI lookahead cleanup timed out; refusing training/weight switch') from exc
        finally:
            for row in audit.rows:
                if row.get('candidate_ordinal', -1) >= prefix:
                    row['disposition'] = 'surplus_unused'
            audit.scheduler_metrics.update({
                'rollout/dapo/lookahead_groups': extra,
                'rollout/dapo/groups_submitted': submitted,
                'rollout/dapo/groups_prefix_consumed': prefix,
                'rollout/dapo/prefix_acceptance_rate': len(selected) / prefix if prefix else 0.,
                'rollout/dapo/submitted_utilization': len(selected) / submitted if submitted else 0.,
                'rollout/dapo/groups_surplus_requeued': submitted - prefix if ready_at is not None else submitted,
                'rollout/dapo/groups_surplus_completed': sum(r.get('candidate_ordinal', -1) >= prefix for r in audit.rows),
                'rollout/dapo/groups_cancelled': sum(t.cancelled() for t in tasks),
                'rollout/dapo/peak_active_episodes': peak_active,
                'rollout/dapo/peak_candidate_window': peak_window,
                'rollout/dapo/batch_ready_s': ready_at - started if ready_at is not None else -1,
                'rollout/dapo/cleanup_s': loop.time() - cleanup_started,
                'rollout/dapo/engine_drain_s': engine_drain_s,
                'rollout/dapo/total_s': loop.time() - started,
                'rollout/dapo/cleanup_ok': int(cleanup_ok),
            })
            if cleanup_ok:
                # Revisit every unconsumed speculative task with fresh samples
                # next batch. This also makes periodic cursor saves correct.
                cursor = (cursors[prefix] if prefix < submitted else owner.sampling_cursor()) if ready_at is not None else original_cursor
                owner.restore_sampling_cursor(cursor)
                state.reset()
    metrics = {**dropped.collect(), **audit.scheduler_metrics}
    logger.info('GUI DAPO lookahead rollout=%s complete: %s', rollout_id, metrics)
    assert len(selected) == target
    return RolloutFnTrainOutput(samples=selected, metrics=metrics)
