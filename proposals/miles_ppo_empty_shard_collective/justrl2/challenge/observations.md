# Recorded observation view

These files are a limited projection of a recorded actor/critic training
failure, not a new execution. Python callsite frames and some instrumentation
fields are omitted. No token ID, reward, mask, group membership or retained
backend exception text has been changed.

- `replay.json` contains both serialized input samples in dispatch order.
- `mesh.json` lists the CP and DP process groups independently for actor and
  critic. Ranks are local to each role's four-worker world. The launch script
  sets TP1/PP1, CP2, GBS2/MBS1 and disables balancing, so replica 0 gets sample 0
  and replica 1 gets sample 1.
- `progress.json` records reachability of two sampling checkpoints within the
  update-input processing interval. Checkpoint numbers are ordinal labels, not
  source locations or collective sequence numbers. A worker recorded at 1
  reached the later checkpoint; one recorded only at 0 did not record it. This
  does not establish the ordering of events between different workers, or prove
  that a particular collective was entered.
- `failure.txt` is the retained terminal backend exception. Its `[0]` and `[1]`
  labels are communicator-local and must not be treated as global worker IDs.

`benchmark/train.sh` records the actual training settings. Model and converted
checkpoint paths are supplied externally through its required environment
variables; weights are not included. No optimizer update or saved checkpoint
from this attempt is claimed by these observations.
