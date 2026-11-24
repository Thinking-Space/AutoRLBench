"""Repository-side evaluation and launch helpers.

The pinned Open-MOPD checkout also contains a top-level ``scripts`` package.
Making this directory an explicit package prevents that unrelated package
from shadowing QAT helpers when both trees are present on ``PYTHONPATH``.
The trainer is launched with its own checkout as ``sys.path[0]``, so this does
not alter imports inside the pinned VeRL process.
"""

