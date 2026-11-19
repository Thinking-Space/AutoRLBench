# Ground-truth diagnosis

## Verdict

Unhealthy. The official-image reproduction reports `response_length/clip_ratio:1.0` in steps 1–3. Each has 64 responses and a minimum, mean, and maximum length of 16,384 tokens. The printed response is whitespace. The same failure appears in the original training log and starts before a learning-rate-driven policy change.

## Root cause

The live NVFP4 resynchronization code in `workspace/qat/qat_kv/nvfp4_rollout.py` assumes that the derived scales stay intact after checkpoint loading. This is a promise about the values, not their storage. The active training bridge is `workspace/qat/vendor/verl/verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`. Its release operation calls vLLM sleep level 2, which discards weight storage and backs up named buffers. The dense ModelOpt NVFP4 module holds `input_global_scale`, `weight_global_scale`, `alpha`, and `input_global_scale_inv` as frozen parameters. They do not belong to the saved-buffer set. Remapping the weight storage does not restore their contents.

Before inference resumes, the custom synchronization repacks the incoming actor matrices using `weight_global_scale`. It writes packed weights and block scales but does not reconstruct the four calibration parameters that the forward path needs. They are original post-load values, not new actor master weights sent with those matrices. The order matters because the pack step already reads the damaged scale. A normal count of synchronized tensors and a stable allocation address can both coexist with corrupted values.

## Observable impact and evidence

- The official rerun records release at level 2 and reports that a worker requantized 112 layers, consumed 196 of 731 input tensors, and forwarded 535 to the regular loader. The first batch fails despite that successful dispatch and a zero learning rate.
- The earlier public training record (job 25966) ran with free_cache_engine=False. At step 3, zero of 64 responses reached the limit. It bypasses weight release between rollout and training. The author-side standalone check used the default level 1 and reached 16/16 normal stops; its CPU backup preserves weight pages. The standalone check is supplemental and is outside the solver workspace.
- The other training logs vary compilation and cache residency. A failed startup under compilation is a separate event. Decode graph capture in both successful and failed runs, a zero training learning rate, and sampling with EOS enabled rule out simple explanations from failed graph creation, optimizer drift, or disabled EOS.
- Credit explanations that connect the loss of frozen scale bytes to disabled or corrupted NVFP4 projections and repetitive decoding. The strongest explanation notes that nearly zero residual branches can reduce the model to an embedding-to-next-token mapping, at which point a newline can feed itself. There is no full historical token-ID array in the workspace. Do not require a claim that every token ID in every sequence was captured.

## Grading

Full root-cause credit requires both the native level-2 storage transition and the incomplete NVFP4 refresh, including why the earlier healthy training run avoids the storage transition. Give partial credit for locating the correct transition or the missing calibration restoration without closing the chain. Generic cache corruption, compilation flags, reward shaping, entropy, token budget, or the zero learning rate alone are insufficient.
