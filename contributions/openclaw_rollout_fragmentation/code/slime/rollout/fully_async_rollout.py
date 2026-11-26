"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency is sourced from ``args.sglang_server_concurrency`` and scaled by
the number of sglang engines to match the per-sample semaphore cap in
:mod:`slime.rollout.sglang_rollout`.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import logging
import queue
import threading
import time

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.misc import load_function
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")
RolloutGroup = list[Sample] | list[list[Sample]]


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _iter_group_samples(group):
    """Yield Sample leaves from plain or custom-generate fan-out groups."""
    for item in group:
        if isinstance(item, list):
            yield from _iter_group_samples(item)
        else:
            yield item


def _group_index(group) -> int:
    for sample in _iter_group_samples(group):
        if sample.index is not None:
            return int(sample.index)
    return 0


def _dynamic_filter_samples(args, group: RolloutGroup) -> list[Sample]:
    """Build one reward-bearing Sample per generated trajectory.

    OpenClaw may split one trajectory into several prefix fragments and divides
    its terminal reward across them. Dynamic sampling is prompt-level, so it
    must compare terminal trajectory rewards rather than fragment rewards.
    """
    samples: list[Sample] = []
    for trajectory in group:
        leaves = list(_iter_group_samples(trajectory if isinstance(trajectory, list) else [trajectory]))
        if not leaves:
            continue
        proxy = copy.copy(leaves[0])
        raw_reward = next(
            (
                leaf.metadata["raw_reward"]
                for leaf in leaves
                if leaf.metadata is not None and "raw_reward" in leaf.metadata
            ),
            None,
        )
        reward = raw_reward if raw_reward is not None else sum(leaf.get_reward_value(args) for leaf in leaves)
        if getattr(args, "reward_key", None):
            proxy.reward = dict(proxy.reward or {})
            proxy.reward[args.reward_key] = reward
        else:
            proxy.reward = reward
        samples.append(proxy)
    return samples


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            _global_worker = AsyncRolloutWorker(
                args, data_buffer, concurrency=args.sglang_server_concurrency * get_rollout_num_engines(args)
            )
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.output_queue: queue.Queue[tuple[int, RolloutGroup]] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self, max_items: int | None = None) -> list[tuple[int, RolloutGroup]]:
        completed: list[tuple[int, RolloutGroup]] = []
        while max_items is None or len(completed) < max_items:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        active_tasks: set[asyncio.Task] = set()
        max_concurrent = self.concurrency
        gid_counter = 0

        while self.running:
            try:
                # Reap done tasks
                if active_tasks:
                    done = {t for t in active_tasks if t.done()}
                    for t in done:
                        try:
                            t.result()  # results already handled in callback
                        except Exception as e:  # noqa: BLE001
                            logger.warning("fully-async task crashed: %r", e)
                    active_tasks -= done

                # Top up.
                while len(active_tasks) < max_concurrent and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        # Custom generators may mutate the source Sample into an
                        # ABORTED sentinel. Keep a clean prompt group for retry.
                        retry_group = copy.deepcopy(group)
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid, retry_group))
                        active_tasks.add(task)

                await asyncio.sleep(1)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(1)

        if active_tasks:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active_tasks),
            )
            try:
                await asyncio.wait(active_tasks, timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int, retry_group: list[Sample]):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected a sample group; dropping",
                    type(result).__name__,
                )
                return
            # Aborted group → requeue, don't ship to training.
            if any(sample.status == Sample.Status.ABORTED for sample in _iter_group_samples(result)):
                try:
                    self.data_buffer.add_samples([retry_group])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put((gid, result))

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None) is not None
        else None
    )
    metric_gatherer = MetricGatherer()

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        # Pull whatever's done.
        drained = 0
        for gid, group in worker.get_completed_groups(max_items=max(1, target - len(collected))):
            filter_output = call_dynamic_filter(dynamic_filter, args, _dynamic_filter_samples(args, group))
            if not filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
                continue
            collected[gid] = group
            drained += 1

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    out = sorted(collected.values(), key=_group_index)[:target]
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
    )
    return RolloutFnTrainOutput(samples=out, metrics=metric_gatherer.collect())


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
