"""Compare one AdamW update with intact and missing moments on a fixed CPU batch."""
import io
import json

import torch


# Each batch supplies the same two actions and their centered rewards. The
# checkpoint cursor points at the first uncommitted batch.
BATCHES = ((-1.0, 1.0), (1.0, -1.0), (1.0, -1.0), (-1.0, 1.0))


def new_policy():
    logits = torch.nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float64, device="cpu"))
    optimizer = torch.optim.AdamW([logits], lr=0.01, weight_decay=0.0, foreach=False)
    return logits, optimizer


def update(logits, optimizer, rewards):
    optimizer.zero_grad()
    advantages = logits.new_tensor(rewards)
    loss = -(advantages * logits.log_softmax(dim=0)).mean()
    loss.backward()
    optimizer.step()


def resume(checkpoint, drop_moments):
    saved = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
    logits, optimizer = new_policy()
    with torch.no_grad():
        logits.copy_(saved["policy_logits"])
    optimizer.load_state_dict(saved["optimizer"])
    state = optimizer.state[logits]
    if drop_moments:
        state["exp_avg"].zero_()
        state["exp_avg_sq"].zero_()
    next_batch = saved["committed_batches"]
    assert int(state["step"].item()) == next_batch
    update(logits, optimizer, BATCHES[next_batch])
    return logits.detach()


def main():
    logits, optimizer = new_policy()
    cursor = len(BATCHES) - 1
    for rewards in BATCHES[:cursor]:
        update(logits, optimizer, rewards)

    buffer = io.BytesIO()
    saved = {
        "policy_logits": logits.detach(),
        "optimizer": optimizer.state_dict(),
        "committed_batches": cursor,
    }
    torch.save(saved, buffer)
    checkpoint = buffer.getvalue()
    update(logits, optimizer, BATCHES[cursor])

    full_resume = resume(checkpoint, drop_moments=False)
    missing_moments = resume(checkpoint, drop_moments=True)
    full_error = (logits.detach() - full_resume).abs().max().item()
    missing_error = (logits.detach() - missing_moments).abs().max().item()
    result = {
        "torch_version": str(torch.__version__),
        "committed_batches": cursor,
        "full_resume_max_abs_error": full_error,
        "missing_moments_max_abs_error": missing_error,
    }
    print(json.dumps(result, indent=2))
    assert full_error == 0.0 and missing_error > 0.0


if __name__ == "__main__":
    main()
