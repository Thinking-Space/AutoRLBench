# Probe: does vLLM 0.11 sleep(level=2) + wake_up + weight reload lose the fp8 KV-cache scales?
import os, glob, json
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from safetensors.torch import load_file
from vllm import LLM, SamplingParams

def main():
    model_dir = os.environ["MODEL"]
    llm = LLM(model=model_dir, kv_cache_dtype="fp8", enable_sleep_mode=True, max_model_len=4096,
              gpu_memory_utilization=0.5, seed=0)
    model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model
    attn = model.model.layers[0].self_attn.attn
    def scales(tag):
        print(tag, "k_scale", attn._k_scale.item(), "v_scale", attn._v_scale.item(),
              "k_scale_float", attn._k_scale_float, "v_scale_float", attn._v_scale_float, flush=True)
    sp = SamplingParams(temperature=0, max_tokens=32)
    msgs = [[{"role": "user", "content": "What is 12*13? Answer briefly."}]]
    scales("before_sleep")
    print("gen_before:", repr(llm.chat(msgs, sp)[0].outputs[0].text), flush=True)
    llm.sleep(level=2)
    llm.wake_up()
    weights = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        weights.update(load_file(f, device="cpu"))
    model.load_weights(weights.items())          # what a trainer weight sync does
    llm.reset_prefix_cache()
    scales("after_wake_reload")
    print("gen_after:", repr(llm.chat(msgs, sp)[0].outputs[0].text), flush=True)

if __name__ == "__main__":
    main()
