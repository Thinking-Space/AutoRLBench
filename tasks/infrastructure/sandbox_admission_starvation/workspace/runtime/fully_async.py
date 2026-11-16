"""Public fully asynchronous fully asynchronous rollout producer.

Unlike the generic fully-async example, one public trajectory can fan out into
multiple training samples.  A completed logical prompt group therefore has the
shape ``list[Sample | list[Sample]]`` and must stay in that shape until the
RolloutManager recursively flattens the returned batch.
"""

from __future__ import annotations

import asyncio
import atexit
import copy
import inspect
import logging
import os
import queue
import threading
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import Any, Iterator

import aiohttp

from miles.rollout.base_types import RolloutFnTrainOutput
from miles.rollout.data_source import DataSource
from miles.rollout.engine_client.args import add_arguments as add_engine_client_arguments
from miles.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from miles.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from miles.utils.async_utils import run
from miles.utils.misc import load_function
from miles.utils.types import Sample

from .progress_watchdog import check_accepted_progress_timeout, optional_positive_env_float

logger = logging.getLogger(__name__)

Trajectory = Sample | list[Sample]
LogicalGroup = list[Trajectory]
SourceGroup = list[Sample]

_QUEUE_ENV = "PUBLIC_ASYNC_QUEUE_MAXSIZE"
_INFLIGHT_ENV = "PUBLIC_ASYNC_MAX_INFLIGHT_GROUPS"
_ZERO_ACCEPTED_TIMEOUT_ENV = "FULLY_ASYNC_ZERO_ACCEPTED_TIMEOUT_SECONDS"
_PROGRESS_LOG_INTERVAL_SEC = 30.0
_QUEUE_POLL_INTERVAL_SEC = 1.0
_PRODUCER_POLL_INTERVAL_SEC = 0.25
_SHUTDOWN_TASK_GRACE_SEC = 3.0
_THREAD_JOIN_TIMEOUT_SEC = 8.0
_GENERATED_METADATA_KEYS = ("abort_reason", "finish_reason", "reward")


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError("must be positive")
        return value
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default


def _iter_group_samples(group: LogicalGroup | SourceGroup) -> Iterator[Sample]:
    """Yield every branch without changing the nested group structure."""

    for item in group:
        if isinstance(item, Sample):
            yield item
        elif isinstance(item, list):
            for branch in item:
                if not isinstance(branch, Sample):
                    raise TypeError(f"Expected Sample branch, got {type(branch).__name__}")
                yield branch
        else:
            raise TypeError(f"Expected Sample or list[Sample] trajectory, got {type(item).__name__}")


def _trajectory_outcome(traj: Trajectory) -> Sample | None:
    """Select the terminal sample for the public workload."""
    if isinstance(traj, Sample):
        return traj
    return traj[-1] if traj else None


def _group_rewards(args: Namespace, group: LogicalGroup) -> list[float]:
    """One terminal reward per trajectory in the group.

    Per trajectory, not per sample: a branched trajectory is several Samples but
    one outcome, and counting its zero-reward segments would drag the mean down
    in proportion to how much that episode happened to branch.
    """
    out: list[float] = []
    for traj in group:
        sample = _trajectory_outcome(traj)
        if sample is None:
            continue
        try:
            out.append(float(sample.get_reward_value(args)))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def _branch_metrics(groups: list[LogicalGroup]) -> dict[str, float]:
    """How many training segments each trajectory was split into.

    A branching session does not produce one sample per episode: every branch
    that generated something becomes its own training segment, so a rollout's
    sample count is not its trajectory count. That ratio drives both the token
    budget and the per-trajectory advantage weighting (segments.py divides the
    broadcast advantage by ``num_segments``), and until now nothing reported it.

    Counted from the nesting rather than from ``num_segments`` metadata: a
    trajectory that never branched stays a bare Sample and carries no such key,
    so reading metadata alone would silently skip exactly the unbranched case
    and inflate the mean.
    """
    counts = [
        len(traj) if isinstance(traj, list) else 1
        for group in groups
        for traj in group
    ]
    if not counts:
        return {}
    branched = sum(1 for c in counts if c > 1)
    return {
        "rollout/branches/mean": sum(counts) / len(counts),
        "rollout/branches/max": max(counts),
        "rollout/branches/branched_ratio": branched / len(counts),
        "rollout/branches/total_segments": sum(counts),
        "rollout/branches/trajectories": len(counts),
    }


def _trajectory_sample(trajectory: Trajectory) -> Sample:
    if isinstance(trajectory, Sample):
        return trajectory
    if not trajectory:
        raise ValueError("A fan-out trajectory cannot be empty")
    if not all(isinstance(branch, Sample) for branch in trajectory):
        raise TypeError("A fan-out trajectory must contain only Sample objects")
    # fan_out_sample_segments keeps the original trajectory object as branch 0.
    return trajectory[0]


def _first_group_sample(group: LogicalGroup | SourceGroup) -> Sample:
    if not group:
        raise ValueError("A logical group cannot be empty")
    return _trajectory_sample(group[0])


def _validate_completed_group(group: Any, expected_trajectories: int) -> LogicalGroup:
    if not isinstance(group, list):
        raise TypeError(f"Completed group must be a list, got {type(group).__name__}")
    if len(group) != expected_trajectories:
        raise ValueError(
            f"Completed group has {len(group)} trajectories; expected {expected_trajectories}"
        )
    for trajectory in group:
        _trajectory_sample(trajectory)
    return group


def _validate_source_group(group: Any, expected_trajectories: int) -> SourceGroup:
    if not isinstance(group, list):
        raise TypeError(f"DataSource group must be a list, got {type(group).__name__}")
    if len(group) != expected_trajectories:
        raise ValueError(
            f"DataSource group has {len(group)} samples; expected {expected_trajectories}"
        )
    if not all(isinstance(sample, Sample) for sample in group):
        raise TypeError("DataSource must return flat list[Sample] prompt groups")
    return group


def group_oldest_weight_version(group: LogicalGroup) -> int | None:
    """Return the oldest version across every trajectory branch in ``group``."""

    versions = [
        version
        for sample in _iter_group_samples(group)
        if (version := sample.oldest_weight_version) is not None
    ]
    return min(versions) if versions else None


def _group_has_aborted_sample(group: LogicalGroup) -> bool:
    return any(sample.status == Sample.Status.ABORTED for sample in _iter_group_samples(group))


def _group_abort_reason(group: LogicalGroup) -> str:
    """Why this group is being discarded, read off the sample that aborted.

    The first sample is usually fine -- it is one bad sample out of four that
    discards the group -- so reading the reason off ``_first_group_sample`` gets
    the reason of a sample that did not abort, i.e. nothing. That is half of why
    974380's fatal message said ``abort_reason='unknown'``; the other half was
    that no code wrote the key at all.
    """
    reasons = [
        str((sample.metadata or {}).get("abort_reason") or "")
        for sample in _iter_group_samples(group)
        if sample.status == Sample.Status.ABORTED
    ]
    named = [reason for reason in reasons if reason]
    return named[0] if named else "unknown"


def _group_sort_key(group: LogicalGroup) -> tuple[bool, int]:
    index = _first_group_sample(group).index
    return index is None, int(index or 0)


def _should_retry_same_prompt(args: Namespace, reason: str | None, group: LogicalGroup) -> bool:
    """Nested-group equivalent of sglang_rollout._should_retry_same_prompt."""

    if reason not in {"solve_none", "solve_all", "all_zero", "all_one"}:
        return False
    # dev 基座删除了 --dynamic-sampling-resample-same-prompt-max-times (minicpm 独有),
    # getattr 恒 0 会让整条重试链静默死掉 —— 改从 env 取, 默认仍关。
    # 注意: dev 的 generate_and_rm_group 也删掉了按 metadata['dynamic_sampling_retry']
    # 偏移 sampling seed 的逻辑, 所以开 --sglang-enable-deterministic-inference 时
    # 同题重试会逐字节复现原输出、重试预算被白白烧光; 仅在随机采样 (正常温度) 下开启。
    max_retries = getattr(args, "dynamic_sampling_resample_same_prompt_max_times", 0)
    if not max_retries:
        max_retries = int(os.environ.get("PUBLIC_RESAMPLE_SAME_PROMPT_MAX_TIMES", "0") or 0)
    if max_retries <= 0:
        return False
    retry_count = int((_first_group_sample(group).metadata or {}).get("dynamic_sampling_retry", 0) or 0)
    return retry_count < max_retries


def _clean_retry_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = dict(metadata or {})
    for key in _GENERATED_METADATA_KEYS:
        cleaned.pop(key, None)
    return cleaned


def _clone_group_for_retry(group: LogicalGroup) -> SourceGroup:
    """Collapse fan-out trajectories to prompt samples and reset generated data."""

    retry_count = int((_first_group_sample(group).metadata or {}).get("dynamic_sampling_retry", 0) or 0) + 1
    retried_group: SourceGroup = []
    for trajectory in group:
        retried = copy.deepcopy(_trajectory_sample(trajectory))
        retried.reset_for_retry()
        if hasattr(retried, "rollout_routed_experts_weights"):
            retried.rollout_routed_experts_weights = None
        retried.status = Sample.Status.PENDING
        retried.session_id = None
        # dev 的路由亲和键是 routing_key 且 reset_for_retry 刻意保留 —— 但本重试
        # 的语义是"换个引擎重来" (原 minicpm 版靠清 session_id 达成), 显式清掉,
        # 让 routing-key 类策略给重试组发新亲和; cache_aware 下无影响。
        retried.routing_key = None
        retried.metadata = _clean_retry_metadata(retried.metadata)
        retried.metadata["dynamic_sampling_retry"] = retry_count
        retried_group.append(retried)
    return retried_group


def _reset_source_group_for_recycle(group: SourceGroup) -> SourceGroup:
    """Reset one source sample per trajectory before returning it to DataSource."""

    for sample in group:
        sample.reset_for_retry()
        sample.session_id = None
        sample.routing_key = None  # 同 _clone_group_for_retry: 回收重投不粘旧引擎
        sample.metadata = _clean_retry_metadata(sample.metadata)
    return group


class _CachedWeightVersion:
    """Throttled current-weight query against the SGLang router."""

    def __init__(self, ttl_sec: float = 1.0) -> None:
        self._ttl_sec = ttl_sec
        self._value: int | None = None
        self._last_attempt = 0.0
        self._last_warning = 0.0

    async def get(self, args: Namespace) -> int | None:
        now = time.monotonic()
        if now - self._last_attempt < self._ttl_sec:
            return self._value

        self._last_attempt = now
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/model_info"
        try:
            timeout = aiohttp.ClientTimeout(total=5.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    response.raise_for_status()
                    payload = await response.json()
            raw_version = payload["weight_version"]
            if isinstance(raw_version, bool):
                raise ValueError(f"invalid boolean weight version {raw_version!r}")
            self._value = int(str(raw_version))
        except Exception as exc:
            if now - self._last_warning >= _PROGRESS_LOG_INTERVAL_SEC:
                logger.warning(
                    "[public-fully-async] Cannot query numeric engine weight version from %s: %s",
                    url,
                    exc,
                )
                self._last_warning = now
        return self._value


_cached_weight_version = _CachedWeightVersion()


@dataclass
class _CompletedGroup:
    group_id: int
    source_group: SourceGroup
    group: LogicalGroup | None = None
    error: BaseException | None = None


class AsyncRolloutWorker:
    """Persistent producer running generation tasks on its own asyncio loop."""

    def __init__(self, args: Namespace, data_source: DataSource) -> None:
        self.args = args
        self.data_source = data_source

        rollout_batch_size = max(1, int(args.rollout_batch_size))
        default_inflight = max(
            1,
            min(
                rollout_batch_size,
                int(getattr(args, "sglang_server_concurrency", rollout_batch_size) or rollout_batch_size),
            ),
        )
        self.queue_maxsize = _positive_env_int(_QUEUE_ENV, rollout_batch_size)
        training_inflight = _positive_env_int("TRAINING_ASYNC_MAX_INFLIGHT_GROUPS", default_inflight)
        self.max_inflight_groups = _positive_env_int(_INFLIGHT_ENV, training_inflight)

        self.output_queue: queue.Queue[_CompletedGroup] = queue.Queue(maxsize=self.queue_maxsize)
        self._retry_queue: queue.Queue[SourceGroup] = queue.Queue()
        self._data_source_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._thread_done = threading.Event()
        self._fatal_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        self._stats_lock = threading.Lock()
        self._stats = {
            "fetched": 0,
            "submitted": 0,
            "completed": 0,
            "task_errors": 0,
            "recycled": 0,
            "retry_submitted": 0,
            "active": 0,
        }

        self._loop: asyncio.AbstractEventLoop | None = None
        self._main_task: asyncio.Task | None = None
        self._active_tasks: dict[asyncio.Task, tuple[int, SourceGroup]] = {}
        self._next_group_id = 0
        self.worker_thread = threading.Thread(
            target=self._thread_main,
            name="public-fully-async-rollout",
            daemon=True,
        )

    def start(self) -> None:
        if self.worker_thread.is_alive():
            return
        self.worker_thread.start()
        logger.info(
            "[public-fully-async] Started producer: queue_maxsize=%d max_inflight_groups=%d",
            self.queue_maxsize,
            self.max_inflight_groups,
        )

    def is_alive(self) -> bool:
        return self.worker_thread.is_alive()

    def _set_fatal_error(self, exc: BaseException) -> None:
        with self._fatal_lock:
            if self._fatal_error is None:
                self._fatal_error = exc
        self._stop_event.set()

    def raise_if_unhealthy(self) -> None:
        with self._fatal_lock:
            fatal_error = self._fatal_error
        if fatal_error is not None:
            raise RuntimeError("public fully-async producer failed") from fatal_error
        if self._thread_done.is_set() and self.output_queue.empty():
            raise RuntimeError("public fully-async producer stopped before the batch was complete")

    def status(self) -> dict[str, int | bool]:
        with self._stats_lock:
            stats = dict(self._stats)
        stats.update(
            {
                "queue_size": self.output_queue.qsize(),
                "retry_queue_size": self._retry_queue.qsize(),
                "alive": self.is_alive(),
            }
        )
        return stats

    def get_completed_group(self, timeout: float) -> _CompletedGroup:
        return self.output_queue.get(timeout=timeout)

    def submit_retry_group(self, group: SourceGroup) -> None:
        if self._stop_event.is_set():
            raise RuntimeError("Cannot retry a prompt after the async producer has stopped")
        _validate_source_group(group, self.args.n_samples_per_prompt)
        self._retry_queue.put(group)
        with self._stats_lock:
            self._stats["retry_submitted"] += 1
        self._wake_loop()

    def recycle_source_group(self, group: SourceGroup, *, reason: str) -> None:
        """Return an aborted/stale/error group without racing DataSource.get_samples."""

        _validate_source_group(group, self.args.n_samples_per_prompt)
        retry_group = _reset_source_group_for_recycle(group)
        try:
            with self._data_source_lock:
                self.data_source.add_samples([retry_group])
        except Exception as exc:
            # Read-only/custom data sources cannot accept recycling.  Keeping the
            # reset group in the producer-local retry queue avoids losing it.
            logger.warning(
                "[public-fully-async] DataSource rejected recycled group (%s): %s; "
                "retrying it from the producer-local queue",
                reason,
                exc,
            )
            self.submit_retry_group(retry_group)
        with self._stats_lock:
            self._stats["recycled"] += 1
        self._wake_loop()

    def _wake_loop(self) -> None:
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(lambda: None)
            except RuntimeError:
                pass

    def _cancel_from_loop(self) -> None:
        for task in tuple(self._active_tasks):
            task.cancel()
        if self._main_task is not None:
            self._main_task.cancel()

    def stop(self) -> None:
        self._stop_event.set()
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._cancel_from_loop)
            except RuntimeError:
                pass

        if self.worker_thread.is_alive() and threading.current_thread() is not self.worker_thread:
            self.worker_thread.join(timeout=_THREAD_JOIN_TIMEOUT_SEC)
            if self.worker_thread.is_alive():
                logger.error(
                    "[public-fully-async] Producer did not stop in %.1fs; forcing its event loop to stop",
                    _THREAD_JOIN_TIMEOUT_SEC,
                )
                loop = self._loop
                if loop is not None and loop.is_running():
                    try:
                        loop.call_soon_threadsafe(loop.stop)
                    except RuntimeError:
                        pass
                self.worker_thread.join(timeout=1.0)

        drained = 0
        while True:
            try:
                self.output_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        retry_drained = 0
        while True:
            try:
                self._retry_queue.get_nowait()
                retry_drained += 1
            except queue.Empty:
                break
        logger.info(
            "[public-fully-async] Producer stopped; drained %d completed and %d retry groups",
            drained,
            retry_drained,
        )

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._started_event.set()
        try:
            self._main_task = loop.create_task(self._continuous_worker_loop())
            loop.run_until_complete(self._main_task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            if not self._stop_event.is_set():
                self._set_fatal_error(exc)
                logger.exception("[public-fully-async] Producer thread crashed")
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                # Give cancellation/finally blocks a bounded grace period to run
                # async cleanup (e.g. DELETE /sandboxes over the network).
                # A single asyncio.sleep(0) gives only one loop turn — not enough
                # for an HTTP DELETE that takes hundreds of milliseconds (confirmed
                # by 899342: 8 sandboxes created, only 4 deleted — the other 4
                # leaked because their finally blocks couldn't complete).
                _SHUTDOWN_GRACE_SECONDS = float(
                    os.getenv("PUBLIC_ASYNC_SHUTDOWN_GRACE_SECONDS", "30")
                )
                try:
                    loop.run_until_complete(
                        asyncio.wait_for(
                            asyncio.gather(*pending, return_exceptions=True),
                            timeout=_SHUTDOWN_GRACE_SECONDS,
                        )
                    )
                except BaseException:
                    pass
            self._main_task = None
            self._active_tasks.clear()
            self._loop = None
            loop.close()
            self._thread_done.set()

    def _next_source_group(self) -> SourceGroup | None:
        try:
            return self._retry_queue.get_nowait()
        except queue.Empty:
            pass

        with self._data_source_lock:
            groups = self.data_source.get_samples(1)
        if not groups:
            return None
        if not isinstance(groups, list):
            raise TypeError(f"DataSource.get_samples must return a list, got {type(groups).__name__}")
        if len(groups) != 1:
            raise ValueError(f"DataSource.get_samples(1) returned {len(groups)} groups")

        group = _validate_source_group(groups[0], self.args.n_samples_per_prompt)
        with self._stats_lock:
            self._stats["fetched"] += 1
        return group

    async def _put_completed(self, completed: _CompletedGroup) -> bool:
        while not self._stop_event.is_set():
            try:
                self.output_queue.put_nowait(completed)
                with self._stats_lock:
                    self._stats["completed"] += 1
                    if completed.error is not None:
                        self._stats["task_errors"] += 1
                return True
            except queue.Full:
                # Waiting here is intentional: the bounded queue is the
                # producer/consumer backpressure boundary. Yield to keep active
                # public requests, timeout handlers, and cleanup coroutines
                # progressing while training is not draining the queue.
                await asyncio.sleep(_PRODUCER_POLL_INTERVAL_SEC)
        return False

    async def _continuous_worker_loop(self) -> None:
        state = GenerateState(self.args)
        state.reset()
        sampling_params = state.sampling_params
        logger.info("[public-fully-async] Producer event loop is ready")

        try:
            while not self._stop_event.is_set():
                while (
                    not self._stop_event.is_set()
                    and len(self._active_tasks) < self.max_inflight_groups
                    and not self.output_queue.full()
                ):
                    try:
                        source_group = self._next_source_group()
                    except BaseException as exc:
                        self._set_fatal_error(exc)
                        logger.exception("[public-fully-async] Failed to fetch a prompt group")
                        return

                    if source_group is None:
                        break

                    group_id = self._next_group_id
                    self._next_group_id += 1
                    task = asyncio.create_task(
                        generate_and_rm_group(
                            self.args,
                            source_group,
                            sampling_params=sampling_params.copy(),
                            evaluation=False,
                        ),
                        name=f"harbor-rollout-group-{group_id}",
                    )
                    self._active_tasks[task] = (group_id, source_group)
                    with self._stats_lock:
                        self._stats["submitted"] += 1
                        self._stats["active"] = len(self._active_tasks)

                if not self._active_tasks:
                    await asyncio.sleep(_PRODUCER_POLL_INTERVAL_SEC)
                    continue

                done, _ = await asyncio.wait(
                    tuple(self._active_tasks),
                    timeout=_PRODUCER_POLL_INTERVAL_SEC,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    group_id, source_group = self._active_tasks.pop(task)
                    with self._stats_lock:
                        self._stats["active"] = len(self._active_tasks)
                    try:
                        group = task.result()
                    except asyncio.CancelledError:
                        if self._stop_event.is_set():
                            continue
                        error = RuntimeError(f"Generation task {group_id} was unexpectedly cancelled")
                        logger.error("[public-fully-async] %s", error)
                        await self._put_completed(
                            _CompletedGroup(group_id=group_id, source_group=source_group, error=error)
                        )
                    except BaseException as exc:
                        logger.error(
                            "[public-fully-async] Generation task %d failed: %r",
                            group_id,
                            exc,
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )
                        await self._put_completed(
                            _CompletedGroup(group_id=group_id, source_group=source_group, error=exc)
                        )
                    else:
                        await self._put_completed(
                            _CompletedGroup(
                                group_id=group_id,
                                source_group=source_group,
                                group=group,
                            )
                        )
        finally:
            pending = tuple(self._active_tasks)
            for task in pending:
                task.cancel()
            if pending:
                done, still_pending = await asyncio.wait(pending, timeout=_SHUTDOWN_TASK_GRACE_SEC)
                for task in done:
                    try:
                        task.result()
                    except BaseException:
                        pass
                if still_pending:
                    logger.warning(
                        "[public-fully-async] Closing loop with %d generation tasks still cancelling",
                        len(still_pending),
                    )
            self._active_tasks.clear()
            with self._stats_lock:
                self._stats["active"] = 0
            state.reset()


_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def get_global_worker(args: Namespace, data_source: DataSource) -> AsyncRolloutWorker:
    """Get the process-global producer, creating it only for the first rollout."""

    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            if _global_worker.data_source is not data_source:
                raise RuntimeError("Global public rollout worker was created for a different DataSource")
            _global_worker.raise_if_unhealthy()
            return _global_worker

        worker = AsyncRolloutWorker(args, data_source)
        _global_worker = worker
        worker.start()
        return worker


def stop_global_worker() -> None:
    """Stop and detach the process-global public producer."""

    global _global_worker
    with _worker_lock:
        worker = _global_worker
        _global_worker = None
    if worker is not None:
        worker.stop()


def _is_last_rollout(args: Namespace, rollout_id: int) -> bool:
    # When --num-epoch is resolved by the driver after the Ray actor was
    # constructed, this actor's args.num_rollout can remain None.  Only stop
    # eagerly when the exclusive upper bound is actually available here.
    num_rollout = getattr(args, "num_rollout", None)
    return isinstance(num_rollout, int) and num_rollout > 0 and rollout_id >= num_rollout - 1


def _failure_limit(worker: AsyncRolloutWorker, target_size: int) -> int:
    # public's custom generate converts most external failures to ABORTED
    # samples rather than raising.  Bound consecutive failures so a dead public
    # service cannot turn into an infinite recycle loop.
    return max(8, target_size * 4, worker.max_inflight_groups * 4)


async def generate_rollout_async(
    args: Namespace,
    rollout_id: int,
    data_source: DataSource,
) -> RolloutFnTrainOutput:
    """Collect one training batch from the persistent public producer.

    Groups whose samples carry ``_tito_deferred`` hold only a trajectory
    summary until this collector decides their fate: materialize (keep) or
    discard (drop) happens here, through ``--deferred-rollout-group-finalize-function-path``.
    Without it, deferred samples reach training with an empty token list and
    their sessions leak on the Session Server.
    """

    assert args.rollout_global_dataset
    worker = get_global_worker(args, data_source)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path)
        if getattr(args, "dynamic_sampling_filter_path", None) is not None
        else None
    )
    deferred_group_finalizer = (
        load_function(args.deferred_rollout_group_finalize_function_path)
        if getattr(args, "deferred_rollout_group_finalize_function_path", None)
        else None
    )

    async def _finalize(group: LogicalGroup, *, keep: bool) -> None:
        if deferred_group_finalizer is None:
            return
        result = deferred_group_finalizer(args, group, keep=keep)
        if inspect.isawaitable(result):
            await result

    metric_gatherer = MetricGatherer()

    target_size = int(args.rollout_batch_size)
    if target_size <= 0:
        raise ValueError(f"rollout_batch_size must be positive, got {target_size}")
    valid_groups: list[LogicalGroup] = []
    # Rewards of every group the filter judged, kept or not. rollout/raw_reward
    # is computed on the training side, which only ever sees kept groups, so it
    # cannot answer "is the model improving": the zero-std filter removes the
    # all-correct groups along with the all-wrong ones, and a model that starts
    # solving whole groups has exactly those dropped. The unfiltered mean below
    # keeps that visible.
    prefilter_rewards: list[float] = []
    started_at = time.monotonic()
    last_accepted_at = started_at
    accepted_progress_timeout = optional_positive_env_float(
        _ZERO_ACCEPTED_TIMEOUT_ENV,
        # Backward-compatible Training launcher name.
        "TRAINING_ASYNC_ZERO_ACCEPTED_TIMEOUT_SECONDS",
    )
    next_progress_log = started_at + _PROGRESS_LOG_INTERVAL_SEC
    first_sample_logged = False
    consecutive_failures = 0
    failure_limit = _failure_limit(worker, target_size)
    task_errors = 0
    aborted_recycled = 0
    stale_recycled = 0
    filtered_dropped = 0
    staleness_values: list[int] = []
    max_weight_staleness = getattr(args, "max_weight_staleness", None)

    logger.info(
        "[public-fully-async] Rollout %d collecting %d logical groups; producer=%s",
        rollout_id,
        target_size,
        worker.status(),
    )

    while len(valid_groups) < target_size:
        _check_now = time.monotonic()
        check_accepted_progress_timeout(
            now=_check_now,
            last_accepted_at=last_accepted_at,
            timeout_seconds=accepted_progress_timeout,
            accepted=len(valid_groups),
            target=target_size,
            filtered_dropped=filtered_dropped,
        )
        try:
            completed = worker.get_completed_group(timeout=_QUEUE_POLL_INTERVAL_SEC)
        except queue.Empty:
            worker.raise_if_unhealthy()
            now = time.monotonic()
            if now >= next_progress_log:
                logger.info(
                    "[public-fully-async] Rollout %d waiting: valid=%d/%d elapsed=%.1fs producer=%s",
                    rollout_id,
                    len(valid_groups),
                    target_size,
                    now - started_at,
                    worker.status(),
                )
                next_progress_log = now + _PROGRESS_LOG_INTERVAL_SEC
            continue

        now = time.monotonic()
        if now >= next_progress_log:
            logger.info(
                "[public-fully-async] Rollout %d progress: valid=%d/%d elapsed=%.1fs producer=%s",
                rollout_id,
                len(valid_groups),
                target_size,
                now - started_at,
                worker.status(),
            )
            next_progress_log = now + _PROGRESS_LOG_INTERVAL_SEC

        if completed.error is not None:
            task_errors += 1
            consecutive_failures += 1
            logger.error(
                "[public-fully-async] Recycling group %d after generation error: %r",
                completed.group_id,
                completed.error,
            )
            worker.recycle_source_group(completed.source_group, reason="generation_error")
            if consecutive_failures >= failure_limit:
                raise RuntimeError(
                    f"public rollout saw {consecutive_failures} consecutive failed groups; "
                    "the producer or public service is not making usable progress"
                ) from completed.error
            continue

        try:
            group = _validate_completed_group(completed.group, args.n_samples_per_prompt)
        except Exception as exc:
            task_errors += 1
            consecutive_failures += 1
            logger.error(
                "[public-fully-async] Recycling malformed completed group %d: %s",
                completed.group_id,
                exc,
            )
            worker.recycle_source_group(completed.source_group, reason="malformed_result")
            if consecutive_failures >= failure_limit:
                raise RuntimeError(
                    f"public rollout saw {consecutive_failures} consecutive malformed/failed groups"
                ) from exc
            continue

        if _group_has_aborted_sample(group):
            consecutive_failures += 1
            aborted_recycled += 1
            metric_gatherer.on_dynamic_filter_drop(reason="group_has_aborted")
            abort_reason = _group_abort_reason(group)
            # Discard any deferred trajectory payload before the prompt group is
            # recycled: a recycle starts a fresh agent run under a new session,
            # and the old session's turns are not the retry's training data.
            try:
                await _finalize(group, keep=False)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[public-fully-async] Failed to discard deferred payloads of aborted group %d",
                    completed.group_id,
                )
            worker.recycle_source_group(completed.source_group, reason="aborted")
            logger.warning(
                "[public-fully-async] Recycled aborted group %d (%d/%d consecutive failures)"
                " abort_reason=%s",
                completed.group_id,
                consecutive_failures,
                failure_limit,
                abort_reason,
            )
            if consecutive_failures >= failure_limit:
                raise RuntimeError(
                    f"public rollout saw {consecutive_failures} consecutive aborted groups "
                    f"(latest abort_reason={abort_reason!r}); refusing to wait forever"
                )
            continue

        if max_weight_staleness is not None:
            oldest_version = group_oldest_weight_version(group)
            if oldest_version is not None:
                current_version = await _cached_weight_version.get(args)
            else:
                current_version = None
            if current_version is not None:
                staleness = current_version - oldest_version
                staleness_values.append(staleness)
                if staleness > max_weight_staleness:
                    stale_recycled += 1
                    metric_gatherer.on_dynamic_filter_drop(reason="weight_stale")
                    try:
                        await _finalize(group, keep=False)
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "[public-fully-async] Failed to discard deferred payloads of stale group %d",
                            completed.group_id,
                        )
                    worker.recycle_source_group(completed.source_group, reason="weight_stale")
                    logger.info(
                        "[public-fully-async] Recycled stale group %d "
                        "(oldest=%d current=%d staleness=%d max=%d)",
                        completed.group_id,
                        oldest_version,
                        current_version,
                        staleness,
                        max_weight_staleness,
                    )
                    continue

        if not first_sample_logged:
            sample = _first_group_sample(group)
            logger.info(
                "[public-fully-async] First rollout sample: prompt+response=%r label=%r reward=%r",
                str(sample.prompt) + sample.response,
                str(sample.label)[:100],
                sample.reward,
            )
            first_sample_logged = True

        prefilter_rewards.extend(_group_rewards(args, group))

        filter_output = call_dynamic_filter(dynamic_filter, args, group)
        if not filter_output.keep:
            filtered_dropped += 1
            metric_gatherer.on_dynamic_filter_drop(reason=filter_output.reason)
            # Zero-std and no-trainable groups are the filter working as
            # designed, not a failure of the producer: a base model that solves
            # nothing produces exactly these. Master never counted them toward
            # consecutive_failures, and the retry allow-list does not contain
            # this filter's reasons, so incrementing here could only kill the
            # whole job on a normal early-training plateau.
            try:
                await _finalize(group, keep=False)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "[public-fully-async] Failed to discard deferred payloads of dropped group %d",
                    completed.group_id,
                )
            if _should_retry_same_prompt(args, filter_output.reason, group):
                worker.submit_retry_group(_clone_group_for_retry(group))
                logger.info(
                    "[public-fully-async] Dynamic filter dropped group %d (%s); retrying same prompt",
                    completed.group_id,
                    filter_output.reason,
                )
            else:
                logger.warning(
                    "[public-fully-async] Dynamic filter dropped group %d (%s); no retry configured",
                    completed.group_id,
                    filter_output.reason,
                )
            continue

        if deferred_group_finalizer is not None:
            try:
                await _finalize(group, keep=True)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "[public-fully-async] Deferred materialization failed for group %d: %s",
                    completed.group_id,
                    exc,
                )
                metric_gatherer.on_dynamic_filter_drop(reason="materialize_failed")
                # Materialization failure loses the payload; do not train on a
                # summary-only sample. Unlike the filter path, though, the
                # produce-side work is not in the sample yet -- recycle the
                # source group so the prompt gets a fresh attempt (same as
                # master: data_buffer.add_samples after a failed finalize).
                try:
                    await _finalize(group, keep=False)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "[public-fully-async] Failed to discard deferred payloads after materialization failure, group %d",
                        completed.group_id,
                    )
                worker.recycle_source_group(completed.source_group, reason="materialize_failed")
                # This is a transient failure, not a producer failure; do not
                # increment consecutive_failures. The recycled group will
                # either succeed on retry or eventually trip the loop guard
                # through another path.
                continue

        consecutive_failures = 0
        valid_groups.append(group)
        last_accepted_at = time.monotonic()

    valid_groups.sort(key=_group_sort_key)
    last_sample = _first_group_sample(valid_groups[-1])
    elapsed = time.monotonic() - started_at
    logger.info(
        "[public-fully-async] Rollout %d complete in %.1fs: valid=%d last_prompt+response=%r "
        "last_reward=%r producer=%s",
        rollout_id,
        elapsed,
        len(valid_groups),
        str(last_sample.prompt) + last_sample.response,
        last_sample.reward,
        worker.status(),
    )

    metrics = metric_gatherer.collect()
    if task_errors:
        metrics["rollout/fully_async/task_error_groups"] = task_errors
    if aborted_recycled:
        metrics["rollout/fully_async/aborted_groups_recycled"] = aborted_recycled
    if stale_recycled:
        metrics["rollout/fully_async/stale_groups_recycled"] = stale_recycled
    if filtered_dropped:
        metrics["rollout/fully_async/dynamic_filter_groups_dropped"] = filtered_dropped
    if staleness_values:
        metrics["rollout/fully_async/weight_staleness_mean"] = sum(staleness_values) / len(staleness_values)
        metrics["rollout/fully_async/weight_staleness_max"] = max(staleness_values)
    metrics.update(_branch_metrics(valid_groups))
    if prefilter_rewards:
        n = len(prefilter_rewards)
        solved = sum(1 for r in prefilter_rewards if r > 0.5)
        metrics["rollout/unfiltered/raw_reward"] = sum(prefilter_rewards) / n
        metrics["rollout/unfiltered/solve_rate"] = solved / n
        metrics["rollout/unfiltered/trajectories"] = n

    return RolloutFnTrainOutput(samples=valid_groups, metrics=metrics)


def generate_rollout_fully_async(
    args: Namespace,
    rollout_id: int,
    data_source: DataSource,
    evaluation: bool = False,
) -> RolloutFnTrainOutput:
    """Legacy rollout-function entrypoint used by ``train_async.py``."""

    if evaluation:
        raise ValueError("public fully-async rollout does not support evaluation=True")

    is_last_rollout = _is_last_rollout(args, rollout_id)
    try:
        return run(generate_rollout_async(args, rollout_id, data_source))
    except Exception:
        # A failed consumer must not leave a producer silently running in the
        # RolloutManager process.
        stop_global_worker()
        raise
    finally:
        if is_last_rollout:
            logger.info("[public-fully-async] Rollout %d is the final rollout; stopping producer", rollout_id)
            stop_global_worker()


generate_rollout_fully_async.add_arguments = add_engine_client_arguments

atexit.register(stop_global_worker)
