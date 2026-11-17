import torch

from megatron.core import parallel_state


class StatesLoggingHelper:
    """Collect per-layer activation statistics for periodic logging."""

    tracker = {}
    curr_iteration = 0
    should_log_hidden_rms = False
    should_log_attention_max_logit = False

    @staticmethod
    def save_max_attention_logit_to_tracker(max_attention_logit: torch.Tensor, layer_number: int):
        if layer_number not in StatesLoggingHelper.tracker:
            StatesLoggingHelper.tracker[layer_number] = {}
        StatesLoggingHelper.tracker[layer_number]["max_attention_logit"] = max_attention_logit.detach()

    @staticmethod
    def save_attn_hidden_rms_to_tracker(attn_hidden_states: torch.Tensor, layer_number: int):
        hidden = attn_hidden_states.detach()
        hidden_rms = (hidden.norm(dim=-1) / (hidden.shape[-1] ** 0.5)).mean()
        if layer_number not in StatesLoggingHelper.tracker:
            StatesLoggingHelper.tracker[layer_number] = {}
        StatesLoggingHelper.tracker[layer_number]["attn_rms"] = hidden_rms
        StatesLoggingHelper.tracker[layer_number]["attn_max"] = hidden.max()

    @staticmethod
    def save_mlp_hidden_rms_to_tracker(mlp_hidden_states: torch.Tensor, layer_number: int):
        hidden = mlp_hidden_states.detach()
        hidden_rms = (hidden.norm(dim=-1) / (hidden.shape[-1] ** 0.5)).mean()
        if layer_number not in StatesLoggingHelper.tracker:
            StatesLoggingHelper.tracker[layer_number] = {}
        StatesLoggingHelper.tracker[layer_number]["mlp_rms"] = hidden_rms
        StatesLoggingHelper.tracker[layer_number]["mlp_max"] = hidden.max()

    @staticmethod
    def reduce_states_in_tracker():
        tracker = StatesLoggingHelper.tracker
        for layer_idx in tracker:
            layer_tracker = tracker[layer_idx]
            if "attn_rms" in layer_tracker:
                torch.distributed.all_reduce(
                    layer_tracker["attn_rms"],
                    op=torch.distributed.ReduceOp.AVG,
                    group=parallel_state.get_data_parallel_group(),
                )
                torch.distributed.all_reduce(
                    layer_tracker["mlp_rms"],
                    op=torch.distributed.ReduceOp.AVG,
                    group=parallel_state.get_data_parallel_group(),
                )
                torch.distributed.all_reduce(
                    layer_tracker["attn_max"],
                    op=torch.distributed.ReduceOp.MAX,
                    group=parallel_state.get_data_parallel_group(),
                )
                torch.distributed.all_reduce(
                    layer_tracker["mlp_max"],
                    op=torch.distributed.ReduceOp.MAX,
                    group=parallel_state.get_data_parallel_group(),
                )
            if "max_attention_logit" in layer_tracker:
                torch.distributed.all_reduce(
                    layer_tracker["max_attention_logit"],
                    op=torch.distributed.ReduceOp.AVG,
                    group=parallel_state.get_data_parallel_group(),
                )

    @staticmethod
    def clean_statistics_in_tracker():
        StatesLoggingHelper.tracker = {}
        StatesLoggingHelper.should_log_hidden_rms = False
        StatesLoggingHelper.should_log_attention_max_logit = False

    @staticmethod
    def track_states_metrics(
        iteration: int,
        writer,
        wandb_writer=None,
        log_hidden_rms_interval=None,
        log_max_attention_logit_interval=None,
    ):
        StatesLoggingHelper.curr_iteration = iteration
        should_emit = (
            log_hidden_rms_interval is not None
            and iteration % log_hidden_rms_interval == 0
        ) or (
            log_max_attention_logit_interval is not None
            and iteration % log_max_attention_logit_interval == 0
        )
        if should_emit and StatesLoggingHelper.tracker:
            StatesLoggingHelper.reduce_states_in_tracker()
            for layer_idx, layer_tracker in StatesLoggingHelper.tracker.items():
                if writer is not None:
                    if "attn_rms" in layer_tracker:
                        writer.add_scalar(
                            f"attention_output_rms/layer_{layer_idx}",
                            layer_tracker["attn_rms"],
                            iteration,
                        )
                        writer.add_scalar(
                            f"attention_output_max/layer_{layer_idx}",
                            layer_tracker["attn_max"],
                            iteration,
                        )
                        writer.add_scalar(
                            f"mlp_output_rms/layer_{layer_idx}",
                            layer_tracker["mlp_rms"],
                            iteration,
                        )
                        writer.add_scalar(
                            f"mlp_output_max/layer_{layer_idx}",
                            layer_tracker["mlp_max"],
                            iteration,
                        )
                    if "max_attention_logit" in layer_tracker:
                        writer.add_scalar(
                            f"max attention logit/layer_{layer_idx}",
                            layer_tracker["max_attention_logit"],
                            iteration,
                        )
                if wandb_writer is not None:
                    wandb_payload = {}
                    if "attn_rms" in layer_tracker:
                        wandb_payload.update(
                            {
                                f"attention_output_rms/layer_{layer_idx}": layer_tracker[
                                    "attn_rms"
                                ],
                                f"attention_output_max/layer_{layer_idx}": layer_tracker[
                                    "attn_max"
                                ],
                                f"mlp_output_rms/layer_{layer_idx}": layer_tracker["mlp_rms"],
                                f"mlp_output_max/layer_{layer_idx}": layer_tracker["mlp_max"],
                            }
                        )
                    if "max_attention_logit" in layer_tracker:
                        wandb_payload[
                            f"max attention logit/layer_{layer_idx}"
                        ] = layer_tracker["max_attention_logit"]
                    if wandb_payload:
                        wandb_writer.log(wandb_payload, iteration)
            StatesLoggingHelper.clean_statistics_in_tracker()

        if (
            log_hidden_rms_interval is not None
            and iteration % log_hidden_rms_interval == log_hidden_rms_interval - 1
        ):
            StatesLoggingHelper.should_log_hidden_rms = True

        if (
            log_max_attention_logit_interval is not None
            and iteration % log_max_attention_logit_interval == log_max_attention_logit_interval - 1
        ):
            StatesLoggingHelper.should_log_attention_max_logit = True
