# Third-party notices

`corpus/repos/` holds unmodified snapshots of six permissively licensed Python repositories,
taken from the public SWE-smith mirrors of those projects. Each snapshot keeps the project's own
license file, which governs that snapshot; the list below records what it is.

| Snapshot | Upstream project | License (file in the snapshot) |
| --- | --- | --- |
| `alecthomas_1776_voluptuous.a7a55f83` | alecthomas/voluptuous | BSD 3-Clause (`COPYING`) |
| `borntyping_1776_python-colorlog.dfa10f59` | borntyping/python-colorlog | MIT (`LICENSE`) |
| `lincolnloop_1776_python-qrcode.456b01d4` | lincolnloop/python-qrcode | BSD 3-Clause (`LICENSE`) |
| `mewwts_1776_addict.75284f95` | mewwts/addict | MIT (`LICENSE`) |
| `mozilla_1776_bleach.73871d76` | mozilla/bleach | Apache-2.0 (`LICENSE`) |
| `pyasn1_1776_pyasn1.0f07d724` | pyasn1/pyasn1 | BSD 2-Clause (`LICENSE.rst`) |

The injected bugs in `corpus/patches/` and the task instances they belong to come from the
public SWE-smith dataset (`SWE-bench/SWE-smith`, MIT). The rollout patches in
`corpus/submissions/` were produced against these snapshots -- 11 recorded from a training
run, the rest staged for this task -- and are diffs of the snapshots' own files, carrying the
same licenses.

Everything else in this workspace -- `rlgrade/`, `scripts/`, `tests/`, `logs/` -- is part of the
task and is covered by the task's Apache-2.0 license.
