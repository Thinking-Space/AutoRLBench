# Canonical diagnosis

Expected verdict: unhealthy training pipeline.

case/sync.py finalize_weights reloads initial safetensors after Actor weights have been synchronized into vLLM; the hook is also on the async update path. Actor gradients need not be zero, but rollout generation repeatedly uses initial weights. Remove only the extra initial checkpoint load. The repaired hook is telemetry-only.

Root cause must be tied to source/configuration and recorded evidence; curve shape alone is insufficient.
