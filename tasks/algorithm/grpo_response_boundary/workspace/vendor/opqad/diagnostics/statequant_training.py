"""First formal update evidence from real AdamW state, with no loss changes."""
import torch
from opqad.integrations.statequant_rollout import _record


def verify_optimizer_update(actor, grad_norm):
    stats = torch.zeros(5, dtype=torch.float64, device=grad_norm.device)
    for parameter, state in actor.actor_optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if value is None:
                continue
            value = value.to_local() if hasattr(value, "to_local") else value
            stats[0] += 1
            stats[1] += torch.count_nonzero(value)
            stats[2] += (~torch.isfinite(value)).sum()
            stats[3] += int(value.dtype != torch.float32)
        stats[4] += int(parameter.dtype != torch.float32)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(stats)
    report = dict(optimizer="AdamW", positive_learning_rate=any(g["lr"] > 0 for g in actor.actor_optimizer.param_groups), moment_tensors=int(stats[0]),
                  nonzero_moment_elements=int(stats[1]), nonfinite_moment_elements=int(stats[2]),
                  non_fp32_moment_tensors=int(stats[3]), non_fp32_master_tensors=int(stats[4]),
                  grad_norm=float(grad_norm), actual_learning_rates=[g["lr"] for g in actor.actor_optimizer.param_groups])
    if stats[0] <= 0 or stats[1] <= 0 or stats[2:].sum() != 0 or not torch.isfinite(grad_norm) or grad_norm <= 0:
        _record("first_optimizer_update", dict(status="failed", **report))
        raise RuntimeError(f"Formal StateQuant first update failed: {report}")
    _record("first_optimizer_update", dict(status="passed", **report))
