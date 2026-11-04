# Oracle diagnosis

Expected verdict: **unhealthy**.

## The component

The **critic's value head** — `LinearForLastLayer` in
`slime/backends/megatron_utils/model_provider.py` (bundled in the workspace at
`slime/slime/backends/megatron_utils/model_provider.py`), at the line

```python
self.weight.data.normal_(mean=0.0, std=init_method_std)
```

in `LinearForLastLayer.__init__`.

That class is what slime installs as the critic's output layer. In
`get_model_provider_func` the critic branch overwrites whatever `output_layer`
the model provider built:

```python
if args.role == "critic":
    model.output_layer = LinearForLastLayer(
        input_size=args.hidden_size, output_size=1, config=config
    )
```

It does so **unconditionally**, after the backbone is constructed, so this
constructor is the only thing that decides the value head's initial weights.
`init_method_std` is the backbone's weight-init scale (0.02 for Qwen3), which
is a sensible default for a `[vocab, hidden]` LM head and the wrong idea
entirely for a `[1, hidden]` value head.

## Why ~5-6 and not some other number

`V(s)` is a single linear projection of the backbone's final hidden state, so a
random weight makes it a random projection rather than a prediction:

- `w ~ N(0, 0.02²)` over 2048 inputs gives `||w||₂ ≈ 0.02 · √2048 ≈ 0.91`.
- Qwen3-1.7B's final-norm hidden state has per-component RMS ≈ 2.55.
- Therefore `std(V) ≈ ||w||₂ · 2.55 ≈ 2.3`, i.e. `E[V²] ≈ 5.3`.

Now compare against what the value loss measures. slime's
`value_loss_function` is the PPO clipped value loss

```python
loss = torch.max((values_clipped - returns)**2, (values - returns)**2)
```

with no 0.5 factor. On the critic's **first** optimizer step after a reload,
`values == old_values`, so the clipping is inactive and this is exactly
`mean((V − R)²)`. With `--gamma 1.0 --lambd 1.0` the GAE return for every
response token collapses to the episode return `R ∈ {0, 1}`, and `V` is
independent of it, so

```
mean((V − R)²) = E[V²] − 2·E[V]·E[R] + E[R²] ≈ E[V²] + pass_rate ≈ 5.3 + ~0.1
```

which is the 5-6 that gets reported. A critic that is genuinely uninformed
should instead report `E[R²] = pass_rate ≤ 1`: predicting 0 everywhere is the
best a critic can do before it has seen a single reward, and its loss is then
bounded by the reward scale. So the inflated number is not "the critic hasn't
learned yet" — it is the cost of unlearning a prediction nobody asked for.

## Why the reload exposes it

This is the part worth being precise about. The stale predictions do **not**
come out of the checkpoint — they are created on the way back in.

A saved actor checkpoint contains `output_layer.weight` with shape
`[vocab_size, hidden_size]`. It contains no `[1, hidden_size]` value head,
because the actor never had one. So when the critic is rebuilt on reload:

1. `model_provider` constructs the backbone and then replaces `output_layer`
   with a fresh `LinearForLastLayer(hidden, 1)` — randomly initialised, right
   here, at construction time.
2. `load_checkpoint` restores the backbone from the checkpoint. It has nothing
   to restore into the value head, so the random weight survives untouched.

The critic therefore opens every reload with a different arbitrary value
function. Each restart looks like "the critic forgot everything and is
relearning", which reads as a checkpoint or optimizer-state problem and sends
you looking in the wrong place — the actual defect is one line in a
constructor, and it is equally present on a from-scratch run (which is why the
from-scratch comparison showed the same thing and was not reassuring).

## The fix

Zero-initialise the value head:

```python
self.weight.data.zero_()
if bias:
    self.bias.data.zero_()
```

`LinearForLastLayer` is only ever instantiated for the critic with
`output_size=1`, so this is both minimal and unconditionally correct. A fresh
critic then predicts exactly 0, and its first reported value loss is
`mean(R²) = pass_rate ≤ 1` — bounded by the reward scale, as it should be.

For reference, the production incident was closed with four layers of defence:
zero at construction (this fix), zero again after `load_checkpoint`,
`optimizer.reload_model_params()` to resync the fp32 master weights, and an
offline checker asserting a fresh critic predicts 0.

In this scenario the first layer alone is sufficient, but that is worth stating
precisely, because slime v0.3.2 randomises the value head in **two** places:

| | where | when it runs |
|---|---|---|
| (a) | `LinearForLastLayer.__init__` | always, at construction |
| (b) | `model.py:_reinitialize_critic_output_layer` | only if `_critic_output_layer_needs_reinit` |

Path (b) requires the critic's load path to be a *Megatron* checkpoint — it
looks for a `.metadata` file — whose `output_layer` shape does not match
`[1, hidden_size]`. Stage B loads the critic from the **HuggingFace** weights,
so `needs_reinit` is false, (b) never runs, and (a) is the only thing that sets
the head. Had the critic been pointed at a Megatron checkpoint instead, (b)
would re-randomise the head right after load and zeroing (a) alone would not
help — which is exactly what layer 2 of the production fix is for.

An answer that notices (b) as well is strictly better, not wrong.

Two details of the HF load path are worth knowing, because they are why the
other two layers are not needed here: `_load_checkpoint_hf` already calls
`optimizer.reload_model_params()` itself (layer 3), and the constructor runs
before the optimizer exists at all, so there is no stale fp32 master copy of
the head to resync.

## Acceptable variations

Full credit requires naming the value head / `output_layer` /
`LinearForLastLayer` as the culprit, identifying random (non-zero)
initialisation as the defect, and locating it at construction time in
`model_provider.py`. Also acceptable:

- Saying "`init_method_std` is wrong for a 1-output head" rather than quoting
  `normal_` directly.
- Deriving `E[V²]` loosely ("order of a few units, set by the hidden-state
  norm") instead of computing 5.3.
- Proposing a fix that zeroes the head after load, or that zeroes it in both
  places, instead of only at init.

Not acceptable:

- Attributing it to the checkpoint containing a stale value head, or to the
  optimizer state — the checkpoint has no value head at all.
- Blaming the reward scale, the advantage estimator, GAE, or `--value-clip`.
- Calling it normal warmup behaviour that needs more warmup steps.
