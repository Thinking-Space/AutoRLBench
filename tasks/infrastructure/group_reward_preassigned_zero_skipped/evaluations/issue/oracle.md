# Canonical issue oracle

Reconciled from the frozen log excerpt, the offline interface regression, and
the sanitized adapter source in the workspace.

Expected verdict: **unhealthy**.

## Root cause

- `runtime/group_adapter.py::mark_generator_zero`
  - kind: `symbol`
  - observed: writes `reward = 0.0` and records
    `behavior_zero = True`, but does not write `reward_record = {"mode":
    "binary", "value": 0.0}`.
- `runtime/group_adapter.py::normalize_group`
  - kind: `symbol`
  - observed: skips a sample whenever `reward is not None`, so the
    generator-assigned zero is never normalized into the record consumed by
    the filter.
- `runtime/group_adapter.py::accept_valid_group`
  - kind: `symbol`
  - observed: requires an explicit binary reward record on every retained
    sample and returns `non_binary_reward_mode` for groups containing the
    unnormalized behavioral zeros.

These two writes and the skip compose into one root-cause path: a valid
generator label bypasses the group adapter's representation boundary. The
model's outputs and the dynamic-sampling all-pass/all-fail policy are not the
cause.

## Observable impact

- The sanitized representative group is valid on all eight samples: three
  submitted answers plus five budget-exhausted behavioral zeros.
- The three submitted answers have binary records after group scoring; the
  five zeros do not.
- `accept_valid_group` rejects the group with `non_binary_reward_mode`.
- The recorded log snapshot contains 881 unique groups dropped for this
  reason. The 605 `solve_all` and 42 `solve_none` drops are expected filter
  behavior and must not be attributed to this defect.
- Application-level training supplied optimizer samples during the snapshot;
  this is a lost-group/throughput defect, not proof that the policy never
  learned.

## Trigger conditions

- Dynamic group sampling is enabled.
- At least one trajectory exhausts its turn budget without submitting a final
  answer.
- The generator converts that outcome to the valid value `0.0` before the
  group adapter runs.
- The group otherwise has both binary outcomes and sufficient valid labels.

## Documentation reconciliation

The repeated `non_binary_reward_mode` log line is the symptom. The root cause
is the mismatch between the label producer and the group adapter, not the
filter, the final score distribution, or an environment failure. The safe
repair must preserve the zero and reject arbitrary or environment-generated
preassigned rewards; it must not convert every existing zero into a valid
binary label.
