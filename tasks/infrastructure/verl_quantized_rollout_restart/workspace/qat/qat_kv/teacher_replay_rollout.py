"""Replay cached teacher trajectories in place of student generation.

This turns the on-policy OPD run into its off-policy twin -- the LLM-QAT
structure -- by changing exactly one thing: where the response tokens come
from.  Everything downstream (student forward, teacher forward, reverse-KL
reward, advantage, policy loss, block-align, checkpointing) runs the same code
on the same shapes.

The seam is the one ``verl.utils.rollout_skip.RolloutSkip`` already uses:
monkeypatching ``actor_rollout_wg.generate_sequences``.  RolloutSkip itself is
not usable here -- it caches a single batch and replays it every step, which is
a debugging aid, not a dataset.

Two things are deliberately strict:

* A prompt with no cached trajectory raises.  Falling back to real generation
  would silently produce an on-policy arm wearing an off-policy label, and the
  run would look completely healthy.
* The tensors are built with verl's own ``pad_2d_list_to_length`` and
  ``get_response_mask``, and the log-prob padding value is -1, matching
  ``vllm_rollout_spmd.py`` exactly.  A different pad value would change the
  masked mean of every downstream metric.

The teacher log-probs are published under *both* ``rollout_log_probs`` and
``teacher_rollout_log_probs``.  The first is the one verl already acts on:
in bypass mode ``maybe_apply_rollout_correction`` assigns
``old_log_probs = rollout_log_probs`` so that PPO clips
``pi_current / pi_rollout``.  Putting the teacher there makes that ratio the
importance weight pi_student/pi_teacher through the stock path, exactly
parallel to the on-policy arm, which puts the student's rollout log-probs in
the same slot.  The second name is only read by the trainer's replay branch,
which logs the student-versus-teacher ratio before the substitution happens.

One consequence to be aware of when reading dashboards: ``rollout_corr/*``
compares ``rollout_log_probs`` against ``old_log_probs``, which are the same
tensor after the substitution, so those metrics are identically zero here --
just as they already are in the on-policy arm.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length


ENV_VAR = "QAT_TEACHER_REPLAY"


class TeacherReplayRollout:
    """Serves pre-generated teacher trajectories keyed by prompt token ids."""

    print_mark = "[TeacherReplay]"

    def __init__(
        self,
        trajectory_path: str,
        config,
        rollout_wg,
        eos_token_id=None,
        pad_token_id=None,
    ):
        self.path = Path(trajectory_path)
        self.rollout_config = config.actor_rollout_ref.rollout
        self._rollout_wg = rollout_wg
        if eos_token_id is None or pad_token_id is None:
            resolved_eos, resolved_pad = self._resolve_special_tokens(config)
            eos_token_id = resolved_eos if eos_token_id is None else eos_token_id
            pad_token_id = resolved_pad if pad_token_id is None else pad_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        n = int(self.rollout_config.get("n", 1))
        if n != 1:
            raise ValueError(
                f"teacher replay assumes one cached response per prompt, but rollout.n={n}"
            )
        self.response_length = int(self.rollout_config.response_length)

        import pyarrow.parquet as pq

        table = pq.read_table(self.path)
        cols = table.to_pydict()
        if "response_logprobs" not in cols:
            raise ValueError(
                f"{self.path} has no 'response_logprobs' column; it was produced before "
                "scripts/rollout_teacher_traj.py started saving the sampled-token "
                "log-probs, and without them old_log_probs cannot be set to the "
                "behavior policy"
            )

        self._by_prompt: dict[tuple[int, ...], tuple[list[int], list[float]]] = {}
        for prompt_ids, resp_ids, resp_lp in zip(
            cols["prompt_token_ids"], cols["response_token_ids"], cols["response_logprobs"]
        ):
            if len(resp_ids) != len(resp_lp):
                raise ValueError(
                    f"cached trajectory has {len(resp_ids)} tokens but {len(resp_lp)} log-probs"
                )
            self._by_prompt[tuple(prompt_ids)] = (list(resp_ids), list(resp_lp))

        if len(self._by_prompt) != table.num_rows:
            raise ValueError(
                f"{table.num_rows} cached trajectories collapsed to {len(self._by_prompt)} "
                "distinct prompts; duplicate prompts make the replay ambiguous"
            )

        print(
            f"{self.print_mark} loaded {len(self._by_prompt)} teacher trajectories "
            f"from {self.path}",
            flush=True,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_special_tokens(config):
        """Resolve eos/pad without relying on ``prompts.meta_info``.

        The patch sits on the worker *group*, but the training path only gains
        ``meta_info["eos_token_id"]`` inside the worker
        (``fsdp_workers.py:1050``), so at this level the key is absent -- job
        8347 died on exactly that.  (The validation path does set it, in
        ``ray_trainer.py:1238``, which is why only training was affected.)

        The precedence below mirrors the worker: generation_config first,
        tokenizer as the fallback.  Getting this wrong would shift where
        get_response_mask stops and silently change every masked mean.
        """
        from transformers import AutoTokenizer

        from verl.utils.model import get_generation_config

        path = config.actor_rollout_ref.model.path
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        generation_config = get_generation_config(path, trust_remote_code=True)
        if generation_config is not None:
            eos = generation_config.eos_token_id
            pad = generation_config.pad_token_id
        else:
            eos = tokenizer.eos_token_id
            pad = tokenizer.pad_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        if pad is None:
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
        if eos is None:
            raise ValueError(
                f"could not resolve an eos token id for {path}; get_response_mask "
                "would then never terminate a response"
            )
        print(
            f"{TeacherReplayRollout.print_mark} resolved eos_token_id={eos} "
            f"pad_token_id={pad} from {path}",
            flush=True,
        )
        return eos, pad

    def wrap_generate_sequences(self) -> None:
        original = self._rollout_wg.generate_sequences

        def replay_fn(prompts: DataProto, **kwargs):
            # Validation prompts are a different dataset and are not cached;
            # they must still go through the real engine.
            if prompts.meta_info.get("validate", False):
                return original(prompts, **kwargs)
            return self.replay(prompts)

        self._rollout_wg.generate_sequences = replay_fn
        print(f"{self.print_mark} patched actor_rollout_wg.generate_sequences()", flush=True)

    # ------------------------------------------------------------------

    @staticmethod
    def _raw_prompt_ids(prompts: DataProto, row: int) -> tuple[int, ...]:
        """Recover the unpadded prompt for one row.

        ``raw_prompt_ids`` is what the dataset carried in; when it is absent the
        left padding is stripped with the attention mask rather than by
        comparing against pad_token_id, because a prompt may legitimately
        contain the pad token.
        """
        raw = prompts.non_tensor_batch.get("raw_prompt_ids")
        if raw is not None:
            return tuple(int(t) for t in raw[row])
        ids = prompts.batch["input_ids"][row]
        mask = prompts.batch["attention_mask"][row]
        return tuple(int(t) for t in ids[mask.bool()].tolist())

    def replay(self, prompts: DataProto) -> DataProto:
        started = time.perf_counter()
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.size(0)
        # meta_info wins when present (validation fills it in); otherwise fall
        # back to what was resolved from the model directory at construction.
        eos_token_id = prompts.meta_info.get("eos_token_id", self.eos_token_id)
        pad_token_id = prompts.meta_info.get("pad_token_id", self.pad_token_id)
        if pad_token_id is None:
            pad_token_id = eos_token_id
        if isinstance(pad_token_id, list):
            # get_response_mask takes a list of eos ids, but padding is a
            # single token.
            pad_token_id = pad_token_id[0]

        responses: list[list[int]] = []
        logprobs: list[list[float]] = []
        missing = 0
        for row in range(batch_size):
            key = self._raw_prompt_ids(prompts, row)
            hit = self._by_prompt.get(key)
            if hit is None:
                missing += 1
                if missing == 1:
                    print(
                        f"{self.print_mark} no cached trajectory for a prompt of "
                        f"{len(key)} tokens starting {key[:12]}",
                        flush=True,
                    )
                responses.append([])
                logprobs.append([])
                continue
            resp, lp = hit
            responses.append(resp[: self.response_length])
            logprobs.append(lp[: self.response_length])

        if missing:
            raise KeyError(
                f"{missing}/{batch_size} prompts in this batch have no cached teacher "
                f"trajectory in {self.path}.  Point data.train_files at the prompt subset "
                "the rollout actually covered; generating the misses instead would turn "
                "this into an on-policy arm without any visible sign."
            )

        response = pad_2d_list_to_length(
            responses, pad_token_id, max_length=self.response_length
        ).to(idx.device)
        # -1 is the pad value vllm_rollout_spmd.py uses for rollout_log_probs.
        teacher_log_probs = (
            pad_2d_list_to_length(logprobs, -1, max_length=self.response_length)
            .to(idx.device)
            .to(torch.float32)
        )

        seq = torch.cat([idx, response], dim=-1)

        delta_position_id = torch.arange(1, self.response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(
                batch_size, position_ids.size(1), -1
            )
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                # Both names carry the same tensor, for two different consumers.
                #
                # rollout_corr_helper.maybe_apply_rollout_correction() runs in
                # bypass mode here and does
                #     batch["old_log_probs"] = batch["rollout_log_probs"]
                # so that "PPO clips ratio = pi_current / pi_rollout".  Feeding
                # the teacher's log-probs into that slot is exactly what makes
                # the ratio the importance weight pi_student/pi_teacher, via
                # the stock path -- the on-policy arm puts the student's
                # rollout log-probs in the same slot.  Omitting the key raises
                # "bypass_old_logprob_for_rollout=True requires
                # rollout_log_probs in batch" (job 8407).
                "rollout_log_probs": teacher_log_probs,
                # The explicit name is what the trainer's replay branch reads
                # to log the student-vs-teacher ratio before the substitution.
                "teacher_rollout_log_probs": teacher_log_probs,
            },
            batch_size=batch_size,
        )

        # vllm_rollout_spmd.py:671 pops raw_prompt_ids before returning, and the
        # trainer then does batch.union(gen_batch_output); leaving the key in
        # collides with the copy the full batch still carries.
        non_tensor_batch = {
            k: v for k, v in prompts.non_tensor_batch.items() if k != "raw_prompt_ids"
        }
        out = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

        # ray_trainer.py:1913 does timing_raw.update(meta_info["timing"]).  The
        # worker fills this in (fsdp_workers.py:1092) *after* the layer this
        # patch replaces, so it has to be supplied here -- job 8398 died on the
        # KeyError.  The numbers are the real replay cost, which is the point
        # of the arm: this is what the on-policy run spends ~96 s/step on.
        elapsed = time.perf_counter() - started
        out.meta_info["timing"] = {
            "generate_sequences": elapsed,
            "generation_timing/max": elapsed,
            "generation_timing/min": elapsed,
            "generation_timing/topk_ratio": 0.0,
        }
        out.meta_info["behavior_policy"] = "teacher_replay"
        # The worker returns on CPU (fsdp_workers.py:1093).
        return out.to("cpu")


def maybe_install(config, rollout_wg) -> TeacherReplayRollout | None:
    """Install the replay if QAT_TEACHER_REPLAY points at a trajectory file.

    Environment-controlled on purpose: adding a hydra key here has bitten this
    project before (a duplicated override aborted job 8280 with
    ConfigCompositionException), and an unset variable cannot alter an ordinary
    production run.
    """
    path = os.environ.get(ENV_VAR, "").strip()
    if not path:
        return None
    replay = TeacherReplayRollout(path, config, rollout_wg)
    replay.wrap_generate_sequences()
    return replay
