"""Do not detach live generation HTTP requests at a GUI sampling boundary.

HTTP disconnect alone may leave image preprocessing or a routed request alive.
Wait for this request to finish (or fail once) before allowing episode cleanup
to return its capacity. This conservative barrier costs at most the pending
action in normal operation; the scheduler's cleanup watchdog fails closed if
serving cannot finish. It never converts a cancelled action into a reward.
"""
import asyncio


class InflightRequestUncertainError(RuntimeError):
    """The client cannot certify that serving finished the routed request."""


def record_boundary_failure(args, message):
    # A group already being cancelled may suppress child exceptions while it
    # awaits siblings. Preserve failure independently until the batch barrier.
    errors = getattr(args, '_gui_dapo_boundary_errors', None)
    if errors is not None:
        errors.append(message)


async def finish_inflight_on_cancel(awaitable):
    request = asyncio.create_task(awaitable)
    try:
        return await asyncio.shield(request)
    except asyncio.CancelledError:
        while not request.done():
            try:
                await asyncio.shield(request)
            except asyncio.CancelledError:
                # Repeated cancellation must not detach the serving request.
                continue
            except Exception:
                break
        if not request.cancelled():
            error = request.exception()
            if error is not None:
                raise InflightRequestUncertainError('Generation HTTP failed during cancellation') from error
        raise
    except Exception as exc:
        # A failed socket can leave router/tokenizer work alive. Stop this batch
        # instead of treating it as a normal rejected group and switching weights.
        raise InflightRequestUncertainError('Generation HTTP completion could not be verified') from exc
