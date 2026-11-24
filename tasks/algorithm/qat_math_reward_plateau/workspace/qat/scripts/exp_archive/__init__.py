"""Build a dated, self-describing archive of every experiment we have run.

The experiments themselves live under ``$QAT_ROOT/runs`` in a layout that grew
by accretion: evaluation results sit in a dozen differently-named directories,
training runs carry their config in ``effective_settings.json``, and the
per-step metrics only exist inside slurm ``.out`` logs.  Nothing there is
organised by *when* it happened, which is the one axis a reader coming back to
this project actually wants.

This package reads all of that and writes ``QAT/experiments/`` -- one directory
per calendar day, flat inside: a README narrating the day's work, a generated
FACTS.md it rests on, that day's configs, evaluations, curves and figures.
Individual jobs do *not* get directories; a day of work is the unit a reader
cares about, and smoke tests get a line in a table rather than a folder.  Bulk
artefacts (raw generations, checkpoints, swanlab media) are not copied; each
day records absolute pointers to them in ``POINTERS.md``.

Deliberately no fallbacks: a record whose shape we do not recognise raises
rather than being silently dropped or half-parsed.  An archive that quietly
omits an experiment is worse than no archive, because it reads as complete.
"""

from .harvest import (
    EvalRecord,
    TrainRecord,
    harvest,
)
from .render import render_archive

__all__ = ["EvalRecord", "TrainRecord", "harvest", "render_archive"]
