"""Reward pipeline for repository-repair RL.

`grade_record` is the entry point the trainer's reward worker calls once per rollout; the
modules below are the stages it goes through (command construction, execution, log parsing,
node-id matching, test restoration, reward aggregation).
"""
from .pipeline import grade_record, materialize  # noqa: F401
from .types import Grade, Instance, RunOutput, Submission  # noqa: F401

__all__ = ["grade_record", "materialize", "Grade", "Instance", "RunOutput", "Submission"]
