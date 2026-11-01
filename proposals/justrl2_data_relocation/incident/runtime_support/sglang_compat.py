# SPDX-License-Identifier: Apache-2.0
"""Map SGLang's renamed parser destinations before JustRL2 validation."""
ALIASES = {
    "sglang_dp_size": "sglang_data_parallel_size",
    "sglang_pp_size": "sglang_pipeline_parallel_size",
    "sglang_ep_size": "sglang_expert_parallel_size",
}


def validate_with_aliases(args, validator):
    for short, legacy in ALIASES.items():
        if not hasattr(args, short):
            raise ValueError("SGLang parser destination missing: " + short)
        value = getattr(args, short)
        if hasattr(args, legacy) and getattr(args, legacy) != value:
            raise ValueError("Conflicting SGLang parser destinations: " + legacy)
        setattr(args, legacy, value)
    return validator(args)


def install():
    import miles.utils.arguments as arguments
    original = arguments.sglang_validate_args

    def wrapped(args):
        return validate_with_aliases(args, original)

    arguments.sglang_validate_args = wrapped
