"""Project-only lazy hook bootstrap for Ray and spawned vLLM workers."""
import os

if os.environ.get("TRAINING_RUNTIME_HOOK") == "1":
    import runtime_lowmem
    runtime_lowmem.install()
