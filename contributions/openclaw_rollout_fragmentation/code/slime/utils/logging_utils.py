import logging

from . import swanlab_utils, wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if getattr(args, "use_swanlab", False):
        if primary:
            swanlab_utils.init_swanlab_primary(args, **kwargs)
        else:
            swanlab_utils.init_swanlab_secondary(args, **kwargs)
        return

    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def finish_tracking(args):
    try:
        if getattr(args, "use_swanlab", False):
            swanlab_utils.finish(args)
        else:
            wandb_utils.finish_wandb(args)
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish tracking run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if getattr(args, "use_swanlab", False):
        swanlab_utils.log(args, metrics, step_key)
    elif args.use_wandb:
        import wandb

        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
