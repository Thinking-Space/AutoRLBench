# Repair the quantized rollout worker

Repair the code in `/workspace` so the vLLM worker can generate normal math responses after it releases and reacquires its cache and weight storage. Keep NVFP4, the existing level-2 sleep and wake cycle, and the live refresh from the training actor. Preserve the original data, model, launch recipe, token budget and metric logging.

Write `/workspace/report.md` explaining the cause, your code change, and how it preserves the sleep and refresh behavior. Do not launch training. The harness will run `bash train.sh` with `LOG_DIR` set to its fresh `experiment_logs` directory. It runs the original three steps on two GPUs, with 64 prompts per step and a 16,384-token response budget. Each batch must have at least 48 responses shorter than the limit. The verifier checks level-2 memory release and the final checkpoint.
