"""Resolve an optional training-actor implementation."""


def add_train_actor_arguments(parser):
    """Register the opt-in training-actor hook."""
    parser.add_argument(
        "--train-actor-class-path",
        type=str,
        default=None,
        help=(
            "Optional import path to a custom Ray training actor class. "
            "When unset, Miles uses the actor selected by --train-backend."
        ),
    )
    return parser


def resolve_train_actor_impl(args, default_actor_class):
    actor_path = getattr(args, "train_actor_class_path", None)
    if actor_path:
        from miles.utils.misc import load_function

        return load_function(actor_path)
    return default_actor_class
