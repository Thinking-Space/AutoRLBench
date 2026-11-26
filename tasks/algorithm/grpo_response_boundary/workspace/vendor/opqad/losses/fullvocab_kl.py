"""Analytic full-vocabulary KL between a quantized student and a frozen teacher.

Why this exists
---------------
The top-k OPD objective scores the union of the student's and teacher's top-512
logits.  On the 1.5B OSCAR INT2 arm the student's probability mass *left* that
set while the objective improved:

    step   student_p_sum_intersection   max_student_p_intersection   log_ppl
       1            0.890                       0.322                 3.436
      37            0.754                       0.106                 5.889

A top-k objective cannot see that.  The other 151424 tokens are not in the sum,
so nothing penalises mass draining into them, and ``val-topk/overlap_ratio``
rose 0.590 -> 0.724 throughout the collapse because set membership is blind to
where the mass is.  This term closes that hole by summing over the whole
vocabulary.

Direction: REVERSE KL, deliberately
-----------------------------------
    reverse:  sum_v p_s(v) [log p_s(v) - log p_t(v)]     (mode-seeking)
    forward:  sum_v p_t(v) [log p_t(v) - log p_s(v)]     (mode-covering)

Reverse is the one aimed at the observed failure.  Where the student puts mass
and the teacher has none, ``log p_t`` is very negative and the term blows up, so
the drain is punished directly.  Forward KL weights by ``p_t``, so it barely
notices what the student does outside the teacher's support -- and its
mode-covering pull is a flattening pressure, which is the disease.  ``forward``
is implemented anyway so the comparison is one argument rather than a rewrite.

It is also a DIRECT differentiable loss, not a detached reward.  That removes
the policy-gradient estimator entirely, and with it the mechanism that
``val-topk/adv_at_max_student_intersection`` kept pointing at: it was negative
at every step (-0.475, -0.242, -0.118, -0.079), i.e. the objective pushed down
on whichever token the student was most confident about, and a flat
distribution is the fixed point of that dynamic.

Memory: why hidden states, and why chunked
------------------------------------------
Teacher logits are never materialised anywhere.  At the 1.5B/16k/mb=4 shape:

    teacher logits [4, 16384, 151936] bf16   = 19.9 GB   <- never built
    teacher hidden [4, 16384,   1536] bf16   =  0.20 GB  <- what is shipped
    teacher lm_head      [151936, 1536] bf16 =  0.47 GB  <- one resident copy

``block_align_qat``'s docstring rejects shipping teacher hidden states at
"28 layers x [4, 18432, 3584] ~= 14.8 GB per micro-batch".  That is 28 layers.
This needs exactly ONE -- the final post-norm hidden state that ``lm_head``
consumes -- so it is 1/28 of that figure and it is viable.

The KL is then evaluated in chunks along the sequence axis, each chunk wrapped
in ``torch.utils.checkpoint`` so the fp32 log-softmaxes do not all survive until
backward.  Without checkpointing the retained intermediates are
``n_chunks * 2 * [b, C, V] fp32``, which at C=512 is ~80 GB -- the chunking
alone does not save anything, the recompute is the point.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional, Tuple

import torch

__all__ = [
    "FullVocabKLConfig",
    "full_vocab_kl",
    "load_teacher_lm_head",
]


class FullVocabKLConfig:
    """Knobs for the full-vocabulary KL term.

    Args:
        coef: linear scale on the returned loss.  The returned loss is already
            multiplied by it, matching ``compute_quantization_block_alignment``
            (``block_align_qat.py:239``), so a logged ``actor/fullvocab_kl``
            is directly comparable to ``actor/pg_loss``.
        direction: ``"reverse"`` (default), ``"forward"``,
            ``"teacher_top1_ratio"``, or ``"head_hinge"``.  The
            ``teacher_top1_ratio`` mode uses
            ``w_f = min(exp(log p_t(y*) - log p_s(y*)), w_max)`` and returns
            ``reverse + w_f * forward_clipped`` where ``y* = argmax p_t``.
            The positive per-token log-ratio in ``forward_clipped`` is capped
            at ``log(w_max)``; there is deliberately no lower clamp, so a
            student that is sharper on the teacher top-1 receives ``w_f < 1``.
        chunk: sequence positions per checkpointed chunk.  Transient cost is
            about ``b * chunk * V * 10`` bytes (bf16 logits + two fp32
            log-softmaxes + fp32 probs).  256 at b=4, V=151936 is ~1.6 GB.
        student_temperature / teacher_temperature: applied to the respective
            logits before the softmax, matching the OPD path's handling.
    """

    __slots__ = (
        "coef", "direction", "chunk", "student_temperature",
        "teacher_temperature", "w_max", "forward_topk", "rho", "lam",
        "rho_start", "rho_end", "rho_anneal_steps", "recovery_power", "gate_schedule",
    )

    def __init__(
        self,
        coef: float = 1.0,
        direction: str = "reverse",
        chunk: int = 256,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        w_max: float = 100.0,
        forward_topk: int = 0,
        rho: float = 0.0,
        lam: float = 1.0,
        rho_end: float = 0.0,
        rho_anneal_steps: int = 0,
        recovery_power: int = 2,
        gate_schedule: Optional[list] = None,
    ) -> None:
        if direction not in ("reverse", "forward", "teacher_top1_ratio", "head_hinge"):
            raise ValueError(
                "direction must be 'reverse', 'forward', 'teacher_top1_ratio' or "
                f"'head_hinge'; got {direction!r}"
            )
        if chunk <= 0:
            raise ValueError(f"chunk must be positive; got {chunk}")
        if student_temperature <= 0 or teacher_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if w_max <= 0:
            raise ValueError("w_max must be positive")
        if direction == "head_hinge":
            if not rho > 1.0:
                raise ValueError(
                    "direction=head_hinge needs QAT_FULLKL_RHO > 1 -- it is the "
                    "slack the hinge grants (the student may sit up to rho times "
                    "under the teacher on a head token before the term fires). "
                    f"got {rho!r}.  rho <= 1 would penalise everything and make "
                    "the term a plain forward KL with extra steps."
                )
            if lam < 0:
                raise ValueError(f"lam must be >= 0; got {lam}")
        if forward_topk < 0:
            raise ValueError(f"forward_topk must be >= 0; got {forward_topk}")
        if forward_topk and direction not in ("teacher_top1_ratio", "head_hinge"):
            raise ValueError(
                "forward_topk only applies to direction=teacher_top1_ratio or head_hinge; "
                f"got {direction!r}"
            )
        self.coef = float(coef)
        self.direction = direction
        self.chunk = int(chunk)
        self.student_temperature = float(student_temperature)
        self.teacher_temperature = float(teacher_temperature)
        self.w_max = float(w_max)
        self.forward_topk = int(forward_topk)
        self.rho = float(rho)
        self.lam = float(lam)
        if recovery_power not in (1, 2):
            raise ValueError(f"recovery_power must be 1 or 2; got {recovery_power}")
        self.recovery_power = int(recovery_power)
        # rho annealing.  `rho` is the LIVE value the loss reads; `rho_start`
        # keeps the step-0 value so `set_rho_for_step` can interpolate without
        # drifting when it is called more than once per step (the actor's
        # micro-batch loop does exactly that).
        self.rho_start = float(rho)
        self.rho_end = float(rho_end)
        self.rho_anneal_steps = int(rho_anneal_steps)
        self.gate_schedule = None
        if gate_schedule is not None:
            from opqad.losses.gate_schedule import validate_gate_schedule
            if direction != "head_hinge" or self.rho_anneal_steps:
                raise ValueError("gate_schedule requires head_hinge without rho annealing")
            self.gate_schedule = validate_gate_schedule(gate_schedule)
            first = self.gate_schedule[0]
            if (self.forward_topk, self.rho) != (first["top_k"], first["rho"]):
                raise ValueError("Initial top_k/rho must match the first gate_schedule stage")
        if self.rho_anneal_steps:
            if direction != "head_hinge":
                raise ValueError(
                    "QAT_FULLKL_RHO_ANNEAL_STEPS only applies to "
                    f"direction=head_hinge; got {direction!r}"
                )
            if not self.rho_end > 1.0:
                raise ValueError(
                    "QAT_FULLKL_RHO_END must be > 1 for the same reason rho is: "
                    "it is the slack the hinge grants, and <= 1 makes the term "
                    f"a plain forward KL.  got {self.rho_end!r}"
                )

    @classmethod
    def from_env(cls, coef: Optional[float] = None) -> "FullVocabKLConfig":
        """Build from QAT_FULLKL_* with an explicit coef override.

        The coefficient comes from the trainer config rather than the
        environment so it lands in the hydra line and the run directory name,
        the way ``block_align_coef`` does.

        Everything else goes through ``opqad.utils.envreg``: direction and the two
        temperatures are REQUIRED there, so an arm that turns this term on has
        to say which divergence it is running.  Defaulting them silently is the
        failure class this repository keeps paying for -- and ``direction``
        in particular flips the objective from mode-seeking to mode-covering,
        which is the difference between the fix and the disease.
        """
        from opqad.utils.envreg import env_float, env_int, env_str

        return cls(
            coef=env_float("QAT_FULLKL_COEF") if coef is None else float(coef),
            direction=env_str("QAT_FULLKL_DIRECTION").strip().lower(),
            chunk=env_int("QAT_FULLKL_CHUNK"),
            student_temperature=env_float("QAT_FULLKL_STUDENT_TEMPERATURE"),
            teacher_temperature=env_float("QAT_FULLKL_TEACHER_TEMPERATURE"),
            w_max=env_float("QAT_FULLKL_W_MAX"),
            forward_topk=env_int("QAT_FULLKL_TOPK"),
            rho=env_float("QAT_FULLKL_RHO"),
            lam=env_float("QAT_FULLKL_LAMBDA"),
            rho_end=env_float("QAT_FULLKL_RHO_END"),
            rho_anneal_steps=env_int("QAT_FULLKL_RHO_ANNEAL_STEPS"),
            recovery_power=env_int("QAT_FULLKL_RECOVERY_POWER"),
            gate_schedule=(json.loads(env_str("QAT_FULLKL_GATE_SCHEDULE"))
                           if env_str("QAT_FULLKL_GATE_SCHEDULE") else None),
        )

    def set_rho_for_step(self, global_step: Optional[int]) -> float:
        """Set the joint top-k/rho gate or legacy rho anneal for this update.

        A piecewise gate chooses both top-k and rho from inclusive one-based
        update ranges. Its global_step is mandatory, including after resume.
        Without a piecewise gate, the legacy anneal below is unchanged.

        Linear in rho itself, from `rho_start` at step 0 to `rho_end` at
        `rho_anneal_steps`, then held.  Linear in rho and not in log rho
        because the quantity the operator reasons about is "how many times
        under the teacher is the student allowed to sit", and that is rho.

        Why anneal at all: at rho=10 the hinge is nearly silent.  Measured
        `actor/fullvocab_hinge_active` is 0.026-0.035 on the k2v3 arms and
        0.006 on 4B OSCAR, against the 0.10-0.20 band the launcher banner
        names as the target -- so the band is so loose that the term
        contributes almost nothing and the objective is effectively
        KL + block_align.  Tightening rho over training raises the fraction
        of teacher-head tokens the student is held to.

        With neither gate_schedule nor rho_anneal_steps, return the fixed rho.
        """
        # The actor calls this once before all micro-batches. A resumed
        # worker resolves both values from checkpoint global_steps, independent
        # of prior calls; model, optimizer and dataloader state stay in place.
        if self.gate_schedule is not None:
            from opqad.losses.gate_schedule import gate_for_step
            self.forward_topk, self.rho = gate_for_step(self.gate_schedule, global_step)
            return self.rho
        if not self.rho_anneal_steps:
            return self.rho
        s = 0 if global_step is None or global_step < 0 else int(global_step)
        t = min(1.0, s / float(self.rho_anneal_steps))
        self.rho = self.rho_start + (self.rho_end - self.rho_start) * t
        return self.rho


def load_teacher_lm_head(
    model_path: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Read the frozen teacher's ``lm_head.weight`` [V, H] without the model.

    Loading the whole ``AutoModelForCausalLM`` to reach one matrix would put a
    second copy of the teacher in the actor process.  Only this tensor is
    needed: the teacher's contribution is ``post_norm_hidden @ lm_head.T`` and
    nothing else.

    Raises rather than falling back when the checkpoint ties its embeddings.
    A tied model has no ``lm_head.weight`` and the right tensor is
    ``model.embed_tokens.weight``; picking one silently would produce a
    plausible-looking teacher built from the wrong matrix.
    """
    import glob
    import json

    from safetensors.torch import safe_open

    shards = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"no *.safetensors under {model_path}")

    cfg_path = os.path.join(model_path, "config.json")
    tied = False
    if os.path.isfile(cfg_path):
        with open(cfg_path) as fh:
            tied = bool(json.load(fh).get("tie_word_embeddings", False))

    want = "model.embed_tokens.weight" if tied else "lm_head.weight"
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as fh:
            if want in fh.keys():
                w = fh.get_tensor(want)
                return w.to(device=device, dtype=dtype).contiguous()

    raise KeyError(
        f"{want!r} not found in any shard under {model_path} "
        f"(tie_word_embeddings={tied}); refusing to guess which matrix is the head"
    )


def _kl_chunk(
    student_logits: torch.Tensor,   # [b, c, V]  requires grad
    teacher_hidden: torch.Tensor,   # [b, c, H]
    lm_head: torch.Tensor,          # [V, H]
    s_temp: float,
    t_temp: float,
    reverse: bool,
    teacher_top1_ratio: bool,
    w_max: float,
    forward_topk: int = 0,
    head_hinge: bool = False,
    rho: float = 0.0,
    lam: float = 1.0,
    recovery_power: int = 2,
) -> torch.Tensor:
    """Per-token KL for one sequence chunk.  Returns [8, b, c] in fp32.

    Channel 0 is the per-token loss and is the only differentiable one.
    Channels 1-7 are detached diagnostics: the forward weight w_f, whether it
    saturated at ``w_max``, reverse and forward KL, teacher head mass, hinge
    value, and hinge activation rate.  On plain reverse/forward branches w_f
    is constant and unused channels are zero.

    ``forward_topk`` > 0 restricts the FORWARD sum to the teacher's top-k
    tokens.  The reverse KL stays over the full vocabulary in every case: it is
    the term that punishes student mass where the teacher has none, and there
    is no "top-k" version of that question.

    Everything here is transient by design -- the caller wraps this in
    ``torch.utils.checkpoint`` so these intermediates are dropped after the
    forward and rebuilt during backward.
    """
    s = student_logits.float()
    if s_temp != 1.0:
        s = s / s_temp
    logp_s = torch.log_softmax(s, dim=-1)
    del s

    # The teacher head is applied here, inside the chunk, so the full
    # [b, T, V] teacher logit tensor never exists.
    t = torch.nn.functional.linear(teacher_hidden, lm_head).float()
    if t_temp != 1.0:
        t = t / t_temp
    logp_t = torch.log_softmax(t, dim=-1)
    del t

    if head_hinge:
        # L = KL(p_s||p_t) + (lam / 2 m_S) sum_{v in S} p_t(v) [log(p_t/p_s) - log rho]_+^2
        #
        # S is the TEACHER's top-k and m_S = sum_{v in S} p_t(v), so the second
        # term is a weighted MEAN over the teacher's head: it does not change
        # meaning when k changes or when the head is more/less concentrated.
        # Measured, k=16 covers 96.3% of the teacher's mass, so S is the head
        # and the 151920-token tail -- the part that carries the mode-covering
        # pull -- is not in the sum at all.
        #
        # The hinge is the point.  Below rho the gradient is EXACTLY zero: the
        # student is allowed to sit up to rho times under the teacher on a head
        # token and is not pushed.  Every previous form had a single fixed
        # point at p_s = p_t and dragged the whole head toward it, which is how
        # j112003 lowered the student's probability ON the teacher's argmax
        # (0.0636 -> 0.0532) while its own weight metric said the mechanism was
        # working.  Here the fixed point is a BAND, and inside the band the
        # term is silent.
        #
        # Squared, so the gradient is proportional to the excess: a token the
        # student is 1e30x under pulls far harder than one it is 1000x under.
        # The clamped log-ratio of the earlier version destroyed exactly that
        # distinction.
        #
        # Nothing here is clipped, deliberately, and the term is therefore
        # unbounded: p_s underflows and [.]_+^2 is quadratic in the log-ratio.
        # The global gradient clip at norm 1.0 is what bounds the update; what
        # this file owes is visibility, so the activation rate and the worst
        # token are both logged.
        d = logp_t - logp_s
        d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
        reverse_kl = (logp_s.exp() * (-d)).sum(dim=-1)
        k = forward_topk if forward_topk > 0 else 16
        vals, idx = logp_t.topk(k, dim=-1)
        d_head = vals - logp_s.gather(-1, idx)              # exact, not clipped
        pt_head = vals.exp()
        m_s = pt_head.sum(dim=-1)
        excess = (d_head - math.log(rho)).clamp_min(0.0)
        hinge = (pt_head * excess.pow(recovery_power)).sum(dim=-1) / (2.0 * m_s.clamp_min(1e-8))
        per_token = reverse_kl + lam * hinge
        active = (excess > 0).to(per_token.dtype).mean(dim=-1)
        zero = torch.zeros_like(per_token)
        return torch.stack(
            [
                per_token,
                torch.ones_like(per_token),     # no weight in this form
                zero,                           # nothing saturates
                reverse_kl.detach(),
                zero,                           # no forward KL term
                m_s.detach(),
                hinge.detach(),
                active.detach(),
            ],
            dim=0,
        )

    if teacher_top1_ratio:
        # Compare the teacher's top-1 token in BOTH distributions.  The
        # denominator is not the student's own top-1 probability: using that
        # would ignore precisely the teacher token we want to recover.
        #
        # THE CLAMP IS ON w_f ONLY.  Nothing inside either KL is clamped.
        #
        # An earlier version clamped the per-token log-ratio and then built the
        # forward sum out of the clamped values.  That bounds the term, but it
        # destroys the thing the term is for: every saturated position then
        # contributes exactly log(w_max), so "the student is 1000x under the
        # teacher here" and "the student is 1e30x under" produce the same
        # value AND the same gradient.  The sum can no longer tell a bad
        # position from a catastrophic one, which is precisely the distinction
        # the ratio weighting exists to make.
        #
        # The price is that the term is no longer bounded: p_s has no lower
        # bound, so a single position can put tens of nats into forward_kl and
        # w_f multiplies it (measured: p_t(y*)=0.888, p_s(y*)=5.8e-32 gives
        # forward=63.5).  That is visible rather than silent --
        # fullvocab_forward and fullvocab_fwd_max are both logged -- and the
        # global gradient clip at norm 1.0 already renormalises every update.
        #
        # No lower clamp on w_f, which is deliberate: where the student is
        # SHARPER than the teacher on y*, the forward term is down-weighted.
        # The reverse KL is the unclamped anchor, as before.
        log_w = math.log(w_max)
        d = logp_t - logp_s
        d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
        # reverse: full vocabulary, unclamped, always.  It is the anchor.
        reverse_kl = (logp_s.exp() * (-d)).sum(dim=-1)
        if forward_topk > 0:
            # Restrict the FORWARD sum to the teacher's top-k.
            #
            # Why: w is one scalar per position and it multiplies a sum over
            # the WHOLE vocabulary, so "the student lost the teacher's best
            # token" is used to amplify every token the teacher gives mass to,
            # including a 151928-long tail of near-zeros.  That is
            # mode-covering, and mode-covering IS the flattening this objective
            # exists to stop -- measured on j111727: w_mean fell 14.6 -> 12.5
            # (the mechanism working) while ta_opd_p_tail rose 3.6% -> 6.7% and
            # log_ppl rose 3.44 -> 4.29 (the disease winning).
            #
            # With k tokens the sum covers the teacher's head and nothing else.
            # At k=1 it is a weighted cross-entropy on the teacher's argmax.
            vals, idx = logp_t.topk(forward_topk, dim=-1)
            d_head = vals - logp_s.gather(-1, idx)          # exact, not clamped
            pt_head = vals.exp()
            forward_kl = (pt_head * d_head).sum(dim=-1)
            # topk returns descending, so column 0 IS the teacher argmax.  The
            # Clamped in LOG space so an underflowed p_s cannot make an inf on
            # the way to the cap, and again after exp because exp(log(w_max))
            # comes back as 100.00000762939453 in fp32 and w_f <= w_max is a
            # contract, not an approximation.
            w_f = d_head[..., 0].clamp(max=log_w).exp().clamp(max=w_max).detach()
            head_mass = pt_head.sum(dim=-1).detach()
        else:
            t1 = logp_t.argmax(dim=-1, keepdim=True)
            w_f = d.gather(-1, t1).squeeze(-1).clamp(max=log_w).exp().clamp(max=w_max).detach()
            forward_kl = (logp_t.exp() * d).sum(dim=-1)     # exact, not clamped
            head_mass = torch.ones_like(w_f)
        # Eight channels, not one.  w_f is the whole mechanism -- it is supposed
        # to start large, then fall back toward 1 as the student recovers the
        # teacher's token, and a run where it does NOT fall is a run where the
        # mechanism is not working.  reverse and forward are reported apart
        # because the loss is their sum and the balance between them is the
        # thing under test: at w_mean ~13 the forward term carries ~97% of the
        # value, which is a different objective from the one on the tin.  The
        # extra channels are detached and cost seven [b, c] tensors.
        per_token = reverse_kl + w_f * forward_kl
        return torch.stack(
            [
                per_token,
                w_f,
                (w_f >= w_max).to(w_f.dtype),
                reverse_kl.detach(),
                forward_kl.detach(),
                head_mass,
                torch.zeros_like(per_token),    # no hinge in this form
                torch.zeros_like(per_token),
            ],
            dim=0,
        )
    if reverse:
        # sum_v p_s (log p_s - log p_t)
        kl = (logp_s.exp() * (logp_s - logp_t)).sum(dim=-1)
    else:
        # sum_v p_t (log p_t - log p_s)
        kl = (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1)
    # Same [8, b, c] shape on every branch so the caller has one code path; the
    # diagnostic channels are constant where there is no weight.
    ones, zeros = torch.ones_like(kl), torch.zeros_like(kl)
    return torch.stack(
        [
            kl,
            ones,
            zeros,
            kl.detach() if reverse else zeros,
            zeros if reverse else kl.detach(),
            ones,
            zeros,
            zeros,
        ],
        dim=0,
    )


def full_vocab_kl(
    student_logits: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head: torch.Tensor,
    response_mask: torch.Tensor,
    cfg: FullVocabKLConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Token-mean full-vocabulary KL, already scaled by ``cfg.coef``.

    Args:
        student_logits: [b, T, V], differentiable.
        teacher_hidden: [b, T, H], the teacher's FINAL POST-NORM hidden state --
            the tensor ``lm_head`` consumes.  Passing a pre-norm hidden state
            produces a well-formed number that is not the teacher.
        lm_head: [V, H] frozen teacher head.
        response_mask: [b, T], 1 on scored positions.
        cfg: see :class:`FullVocabKLConfig`.

    Returns:
        ``(loss, metrics)``.  ``loss`` is ``coef * (sum_masked KL) / n_masked``
        which is verl's ``token-mean`` aggregation, so it is on the same
        footing as ``actor/pg_loss``.
    """
    if student_logits.dim() != 3:
        raise ValueError(f"student_logits must be [b, T, V]; got {tuple(student_logits.shape)}")
    if teacher_hidden.dim() != 3:
        raise ValueError(f"teacher_hidden must be [b, T, H]; got {tuple(teacher_hidden.shape)}")
    if lm_head.dim() != 2:
        raise ValueError(f"lm_head must be [V, H]; got {tuple(lm_head.shape)}")
    b, t_len, vocab = student_logits.shape
    if teacher_hidden.shape[:2] != (b, t_len):
        raise ValueError(
            f"teacher_hidden {tuple(teacher_hidden.shape)} does not align with "
            f"student_logits {tuple(student_logits.shape)} on [b, T]"
        )
    if lm_head.shape[0] != vocab:
        raise ValueError(
            f"lm_head vocab {lm_head.shape[0]} != student logits vocab {vocab}"
        )
    if lm_head.shape[1] != teacher_hidden.shape[2]:
        raise ValueError(
            f"lm_head hidden {lm_head.shape[1]} != teacher_hidden hidden {teacher_hidden.shape[2]}"
        )
    if response_mask.shape != (b, t_len):
        raise ValueError(
            f"response_mask must be [b, T] = {(b, t_len)}; got {tuple(response_mask.shape)}"
        )

    from torch.utils.checkpoint import checkpoint

    mask = response_mask.to(torch.float32)
    denom = mask.sum()
    reverse = cfg.direction == "reverse"
    teacher_top1_ratio = cfg.direction == "teacher_top1_ratio"
    head_hinge = cfg.direction == "head_hinge"
    # A micro-batch whose every position is masked out contributes nothing.
    # Returning a detached zero here would break the backward graph, so build
    # the zero FROM the logits instead.
    if float(denom) == 0.0:
        zero = student_logits.sum() * 0.0
        # Every key the non-empty path returns.  A short metrics dict on the
        # masked-out micro-batch is how ta_opd_loss raised a KeyError in the
        # actor on 2026-09-20.
        return zero, {
            "fullvocab_kl": 0.0,
            "fullvocab_kl_tokens": 0.0,
            "fullvocab_w_mean": 0.0,
            "fullvocab_w_saturated": 0.0,
            "fullvocab_reverse": 0.0,
            "fullvocab_forward": 0.0,
            "fullvocab_pt_head": 0.0,
            "fullvocab_fwd_max": 0.0,
            "fullvocab_hinge": 0.0,
            "fullvocab_hinge_active": 0.0,
            "fullvocab_hinge_max": 0.0,
        }

    total = student_logits.new_zeros((), dtype=torch.float32)
    chan = {
        k: student_logits.new_zeros((), dtype=torch.float32)
        for k in ("w", "sat", "rev", "fwd", "head", "hinge", "hinge_active")
    }
    # The hinge is unbounded by design, so the worst single token is reported
    # rather than left to be inferred from a mean.
    hinge_max = student_logits.new_full((), 0.0, dtype=torch.float32)
    # The forward term is no longer bounded (only w_f is capped), so the worst
    # single token is reported rather than left to be inferred from a mean.
    fwd_max = student_logits.new_full((), float("-inf"), dtype=torch.float32)
    grad_on = torch.is_grad_enabled() and student_logits.requires_grad
    for start in range(0, t_len, cfg.chunk):
        stop = min(start + cfg.chunk, t_len)
        m = mask[:, start:stop]
        # Whole chunk masked out (common with right-padded responses): skip the
        # matmul and the two log-softmaxes entirely.
        if float(m.sum()) == 0.0:
            continue
        args = (
            student_logits[:, start:stop, :],
            teacher_hidden[:, start:stop, :],
            lm_head,
            cfg.student_temperature,
            cfg.teacher_temperature,
            reverse,
            teacher_top1_ratio,
            cfg.w_max,
            cfg.forward_topk,
            head_hinge,
            cfg.rho,
            cfg.lam,
            cfg.recovery_power,
        )
        if grad_on:
            out = checkpoint(_kl_chunk, *args, use_reentrant=False)
        else:
            out = _kl_chunk(*args)
        total = total + (out[0] * m).sum()
        with torch.no_grad():
            for _i, _k in enumerate(
                ("w", "sat", "rev", "fwd", "head", "hinge", "hinge_active"), start=1
            ):
                chan[_k] = chan[_k] + (out[_i].detach() * m).sum()
            hinge_max = torch.maximum(hinge_max, (out[6].detach() * m).max())
            masked_fwd = torch.where(m > 0, out[4].detach(), torch.full_like(out[4], float("-inf")))
            fwd_max = torch.maximum(fwd_max, masked_fwd.max())

    loss = total / denom
    metrics = {
        "fullvocab_kl": float(loss.detach()),
        "fullvocab_kl_tokens": float(denom.detach()),
        # The mechanism, not the loss.  w_mean is supposed to fall toward 1 as
        # the student recovers the teacher's top-1; w_saturated is the fraction
        # of positions where the clamp bound it, i.e. where the amplification
        # is being truncated rather than measured.
        "fullvocab_w_mean": float(chan["w"] / denom.detach()),
        "fullvocab_w_saturated": float(chan["sat"] / denom.detach()),
        # reverse and forward apart, because the loss is their sum and the
        # balance is the thing under test.
        "fullvocab_reverse": float(chan["rev"] / denom.detach()),
        "fullvocab_forward": float(chan["fwd"] / denom.detach()),
        # Teacher mass inside the forward sum: 1.0 over the full vocabulary,
        # and at top-k the answer to "is k big enough to be the teacher's head".
        "fullvocab_pt_head": float(chan["head"] / denom.detach()),
        # Worst single token.  With the clamp off the inside of the sum, this
        # is the number that says whether one position is carrying the update.
        "fullvocab_fwd_max": float(fwd_max),
        # head_hinge only.  hinge_active is the fraction of (position, head
        # token) pairs where the student sits more than rho under the teacher,
        # i.e. how much of the head the term is actually touching: as it
        # approaches 1 the hinge has stopped being selective and is a plain
        # penalty on the whole head again.  hinge_max is the worst single
        # position, because the term is quadratic in an unbounded log-ratio.
        "fullvocab_hinge": float(chan["hinge"] / denom.detach()),
        "fullvocab_hinge_active": float(chan["hinge_active"] / denom.detach()),
        "fullvocab_hinge_max": float(hinge_max),
    }
    return loss * cfg.coef, metrics
