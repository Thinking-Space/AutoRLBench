"""Run a complete tool-routing GRPO experiment."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from environment_tool import Ticket, tickets
from policy import Policy
from reward_callback import post_process_rewards, terminal_reward


@dataclass
class Decision:
    stage: int
    prefix: list[int]
    action: int
    log_prob: float


@dataclass
class Fragment:
    group_index: int
    rollout_id: int
    decision: Decision | None
    status: str
    remove_sample: bool
    metadata: dict

    def get_reward_value(self, _args):
        return self.metadata["raw_reward"]


@dataclass
class Rollout:
    ticket: Ticket
    group: int
    slot: int
    index: int
    catalog_response: str = ""
    answer: str | None = None
    disposition: str = "completed"
    decisions: list[Decision] = field(default_factory=list)

    def reward(self):
        return terminal_reward(self.ticket, self.answer, self.catalog_response)

    def fragments(self):
        if self.answer is None:
            return [Fragment(self.group, self.index, None, "aborted", True, {"raw_reward": 0.0})]
        return [
            Fragment(self.group, self.index, decision, "completed", False, {"raw_reward": self.reward()})
            for decision in self.decisions
        ]

    def trace(self):
        return {
            "group": self.group, "rollout": self.index, "question": self.ticket.request(),
            "catalog_response": self.catalog_response or None,
            "actions": ["ABC"[d.action] for d in self.decisions],
            "outcome": self.disposition, "expected": self.ticket.target,
            "answer": self.answer, "reward": self.reward(),
        }


def choice_prompt(rollout, stage):
    request = rollout.ticket.request()
    if stage == 0:
        return request + "\nA: read the ticket catalog. B: answer directly. Select A or B.\nChoice:\n"
    result = rollout.catalog_response or "The ticket catalog was not consulted."
    return request + f"\n{result}\nSelect the correct team option A, B or C.\nChoice:\n"


@torch.no_grad()
def choose(policy, rollouts, stage, temperature, generator, greedy, batch_size):
    for start in range(0, len(rollouts), batch_size):
        batch = rollouts[start:start + batch_size]
        prefixes = [policy.encode(choice_prompt(rollout, stage)) for rollout in batch]
        log_probs = policy.log_probs(prefixes, 2 + stage, temperature)
        cpu_probs = log_probs.cpu()
        if greedy:
            actions = cpu_probs.argmax(dim=-1)
        else:
            # A per-dispatch CPU RNG makes sampling repeatable for identical policies.
            actions = torch.multinomial(cpu_probs.exp(), 1, generator=generator).squeeze(-1)
        for rollout, prefix, action, log_prob in zip(
            batch, prefixes, actions.tolist(), cpu_probs.gather(1, actions[:, None]).flatten().tolist(), strict=True,
        ):
            rollout.decisions.append(Decision(stage, prefix, action, log_prob))


def collect(policy, prompts, samples, step, args, greedy=False, evaluation=False):
    rollouts = [
        Rollout(ticket, group, slot, group * samples + slot)
        for group, ticket in enumerate(prompts)
        for slot in range(samples)
    ]
    generator = torch.Generator().manual_seed(args.seed + (400000 if evaluation else 1000 * step))
    choose(policy, rollouts, 0, args.temperature, generator, greedy, args.microbatch)
    ready = []
    for rollout in rollouts:
        lookup = rollout.decisions[0].action == 0
        # The adapter rejects some dispatched calls during periodic contention.
        # The schedule is independent of generated answers and shared across runs.
        slot_rng = random.Random(args.seed + step * 1009 + rollout.index * 65537)
        timeout_rate = 0.48 if step % 12 in (5, 6, 7, 8) else 0.13
        if lookup and not evaluation and slot_rng.random() < timeout_rate:
            rollout.disposition = "catalog_timeout"
            continue
        if lookup:
            rollout.catalog_response = rollout.ticket.lookup()
        ready.append(rollout)
    choose(policy, ready, 1, args.temperature, generator, greedy, args.microbatch)
    for rollout in ready:
        rollout.answer = "ABC"[rollout.decisions[-1].action]
    return rollouts


def summarize(rollouts):
    n = len(rollouts)
    routed = sum(item.answer is not None and item.ticket.is_correct(item.answer) for item in rollouts)
    grounded = sum(item.reward() == 1.0 for item in rollouts)
    requests = sum(item.decisions[0].action == 0 for item in rollouts)
    completed = sum(item.answer is not None for item in rollouts)
    return {
        "rollouts": n, "completed": completed, "catalog_requests": requests,
        "catalog_timeouts": n - completed, "correct_routes": routed, "verified_routes": grounded,
        "mean_reward": sum(item.reward() for item in rollouts) / n,
    }


def update(policy, optimizer, rollouts, args):
    records = [frag for rollout in rollouts for frag in rollout.fragments()]
    advantages = post_process_rewards(args, records)
    masked = [(frag.decision, adv) for frag, adv in zip(records, advantages, strict=True) if not frag.remove_sample]
    if not masked:
        return {"optimizer_steps": 0, "trainable_actions": 0, "policy_loss": 0.0, "grad_norm": 0.0}
    total_loss, grad_norms = 0.0, []
    for _ in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        for stage in range(2):
            turn = [(d, a) for d, a in masked if d.stage == stage]
            for start in range(0, len(turn), args.microbatch):
                batch = turn[start:start + args.microbatch]
                decisions, weights = zip(*batch, strict=True)
                log_probs = policy.log_probs([d.prefix for d in decisions], 2 + stage, args.temperature)
                actions = torch.tensor([d.action for d in decisions], device=policy.device)
                old = torch.tensor([d.log_prob for d in decisions], device=policy.device)
                adv = torch.tensor(weights, device=policy.device)
                ratio = (log_probs.gather(1, actions[:, None]).squeeze(-1) - old).exp()
                loss = -torch.minimum(ratio * adv, ratio.clamp(0.8, 1.2) * adv).sum() / len(masked)
                loss.backward()
                total_loss += float(loss.detach())
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        grad_norms.append(float(norm))
    return {
        "optimizer_steps": args.epochs, "trainable_actions": len(masked),
        "policy_loss": total_loss / args.epochs, "grad_norm": sum(grad_norms) / len(grad_norms),
    }


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    policy = Policy(args.model, torch.device(args.device))
    params = {name: param for name, param in policy.named_parameters() if param.requires_grad}
    assert params
    optimizer = torch.optim.AdamW(params.values(), lr=args.lr, weight_decay=0.0)
    args.out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    training_set = tickets("train", 2048)
    held_out = tickets("validation", args.validation)
    settings = {
        key: value for key, value in vars(args).items() if key not in {"model", "out"}
    }
    settings["torch"] = torch.__version__
    start_time = time.monotonic()
    total_updates = 0
    with (args.out / "metrics.jsonl").open("w") as log:
        def emit(record):
            line = json.dumps(record, allow_nan=False)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        emit({"event": "config", **settings, "train_tickets": len(training_set)})
        for step in range(args.steps + 1):
            if step > 0:
                prompts = rng.sample(training_set, args.prompts)
                rollouts = collect(policy, prompts, args.samples, step, args)
                metrics = update(policy, optimizer, rollouts, args)
                total_updates += metrics["optimizer_steps"]
                emit({"event": "train", "step": step, **summarize(rollouts), **metrics})
                if step in {1, (args.steps + 1) // 2, args.steps}:
                    path = args.out / f"rollouts-{step:03d}.jsonl"
                    path.write_text("".join(json.dumps(item.trace()) + "\n" for item in rollouts))
            if step % args.eval_interval == 0 or step == args.steps:
                rollouts = collect(policy, held_out, args.eval_samples, step, args, evaluation=True)
                emit({"event": "validation", "step": step, **summarize(rollouts)})
            if step > 0 and (step % args.eval_interval == 0 or step == args.steps):
                checkpoint = {
                    "step": step, "optimizer_steps": total_updates,
                    "adapter": {name: tensor.detach().cpu() for name, tensor in params.items()},
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, args.out / f"step-{step:03d}.pt")
        emit({"event": "complete", "step": args.steps, "optimizer_steps": total_updates,
              "elapsed_s": round(time.monotonic() - start_time, 1)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("assets/model"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps"))
    parser.add_argument("--steps", type=int, default=36)
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--eval-interval", type=int, default=6)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--eval-samples", type=int, default=3)
    parser.add_argument("--prompts", type=int, default=6)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--microbatch", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.35)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.set_defaults(advantage_estimator="grpo", rewards_normalization=True, grpo_std_normalization=True)
    run(parser.parse_args())
