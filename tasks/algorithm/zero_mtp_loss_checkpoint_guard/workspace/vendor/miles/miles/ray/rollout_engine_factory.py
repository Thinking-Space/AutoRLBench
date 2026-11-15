"""Resolve an optional rollout-engine implementation without eager backend imports."""


def add_rollout_engine_arguments(parser):
    """Register the opt-in rollout-engine hook."""
    parser.add_argument(
        "--rollout-engine-class-path",
        type=str,
        default=None,
        help=(
            "Optional import path to a custom rollout engine class. "
            "When unset, Miles uses the default SGLangEngine implementation."
        ),
    )
    return parser


def resolve_rollout_engine_impl(args, default_engine_class):
    engine_path = getattr(args, "rollout_engine_class_path", None)
    if engine_path:
        from miles.utils.misc import load_function

        return load_function(engine_path)
    return default_engine_class
