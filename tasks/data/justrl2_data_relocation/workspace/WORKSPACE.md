# Experiment workspace

`logs/recorded/` contains the historical run. `run.sh` is the foreground launch
command; the evaluation harness supplies `LOG_DIR` and runs it after the report
is submitted. Inspect the actual configuration in the log and the launch files.

`runtime/` is the task-specific data-loading and training integration. The public
framework source under `framework/` is a pinned reference for inspecting native
rollout, grading and optimization; runtime staging rebuilds it from the pinned
archive with the adapter in `runtime/stage_runtime.py`. Changes to the reference
copy alone are not applied by the launch command.

`prepared/dataset` and `prepared/hf-safetensors` are generated from pinned public
assets during prepare. Reruns create a separate copy of the data; they do not
mutate the prepared assets. Keep the run budget, dataset identities and event
schedule specified by the evaluation instruction.
