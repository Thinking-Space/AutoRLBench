"""CLI arguments for the token-level partial-rollout gateway (Task 1+2 only).

Registered automatically via ``InferenceRolloutFn.add_arguments`` (see
``inference_rollout/inference_rollout_common.py``) when
``MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`` is set -- zero changes to
``miles/utils/arguments.py``. All defaults reproduce today's behavior
exactly (gateway transport local, partial mode off).
"""

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--rollout-gateway-transport",
        type=str,
        choices=["local", "remote"],
        default="local",
        help=(
            "local (default): generate calls go straight to the sglang router, byte-for-byte "
            "identical to pre-gateway behavior. remote: white-box generate calls "
            "(single_turn/multi_turn) route through the EngineClient gateway process at "
            "--rollout-gateway-url."
        ),
    )
    parser.add_argument(
        "--rollout-gateway-url",
        type=str,
        default=None,
        help="Base URL of the standalone gateway (session) server. Required when "
        "--rollout-gateway-transport=remote.",
    )
    parser.add_argument(
        "--rollout-partial-mode",
        type=str,
        choices=["off", "token"],
        default="off",
        help=(
            "off (default): EngineClient never resumes an aborted engine call; a single abort "
            "is returned as-is, same as today. token: EngineClient resumes an aborted call "
            "within the same logical turn (abort -> resend with accumulated input_ids) so the "
            "caller only ever sees a complete turn. For the white-box path (single_turn/"
            "multi_turn), this only takes effect when --rollout-gateway-transport=remote. For "
            "the black-box path (SessionCore.chat_completions), this takes effect unconditionally "
            "-- the shared EngineClient lives in-process inside the session server, so no "
            "transport flag is involved. Distinct from the existing --partial-rollout, which "
            "recycles whole samples across rollout steps."
        ),
    )
    parser.add_argument(
        "--engine-client-timeout",
        type=float,
        default=600.0,
        help="Per-request timeout (seconds) for EngineClient's calls to the inference engine "
        "backend. Defaults to the same value as --miles-router-timeout.",
    )
    return parser
