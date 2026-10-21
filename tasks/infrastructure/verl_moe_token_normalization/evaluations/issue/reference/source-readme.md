# training_case_m7q2

Status: final_materials. Per-token loss callback omits the token count, so native auxiliary z-loss gradients skip final token normalization. Full run and metrics are retained; no collapse claim is inferred.

Source: https://github.com/verl-project/verl/issues/6609 ; framework commit `source-snapshot`. Complete training framework implementation is included with the agent-visible exclusions documented below; the optional, unused recipe extension remains declared in `.gitmodules`. No Git history, model weights or data files are bundled.

Real full model Qwen1.5-MoE-A2.7B-Chat (14,315,784,192 global parameters), genuine GSM8K 7473 training rows; GRPO, 4 prompts x 4 responses, prompt-unit PPO mini-batch4, CP2/EP2/TP1, 150 training steps (user-requested cap; not a full epoch). Reference policy and validation are retained. Two tasks share the image problem2 fix, source problem3 padding mask and computation compatibility hooks. The after runtime additionally finishes W&B explicitly at the last step; this changes logging lifecycle only. The original before run has150 real console steps, checkpoint150 and validation150, but only148 raw W&B records. The user accepted this disclosed logging gap; no before rerun or fabricated records are used. No benchmark agent evaluation was requested or executed.

Build environment: `docker build -t verl-runtime-runtime environment/`. Base is immutable digest in Dockerfile; only the declared MCore router backport differs. requirements.txt records the effective268-distribution closure; custom CUDA binaries are supplied by the base image and should not be replaced by generic PyPI wheels. This legacy Dockerfile task does not claim an existing unified conda environment.

Inside that image, mount workspace writable and set PTDBENCH_WORKSPACE. Run `bash environment/prepare.sh` in the only networked stage; it verifies every fixed resource SHA and size before preprocessing. Then disconnect networking and run `bash workspace/scripts/train.sh` with the correct mounted paths and two allocated GPUs. Container HOME/tmp/caches must be mounted under the chosen workspace; scripts do not allocate GPUs or start schedulers. Keep at least56GiB free/card at admission and sufficient host memory; shared GPU capacity can change mid-run.

Offline W&B raw files are produced under each new run. Curated training stdout, original offline W&B scalar history and resolved config are added after the completion audit. Before W&B has1-148 only; separate console-history records1-150 are included and labeled, not substituted into W&B. After original W&B records are preserved as produced; the user also accepts an after W&B trailing gap when real150-step console/checkpoint/validation evidence is complete. Any missing steps are explicitly listed, never fabricated or used to trigger a rerun. Bulky original image/checkpoint/Ray/offline-W&B archives remain in the experiment evidence archive outside the <=50MiB task. No originals are removed. Issue judge is deliberately unconfigured (exit2/not_evaluated); solution native-gradient verifier is supplied, but not executed as an agent evaluation. It checks actual nonzero z-loss behavior under both flag settings; this small diagnostic is not full RL. No agent_trace.json/report.json is fabricated.

Licenses: framework Apache-2.0; GSM8K MIT (fixed source README); model Tongyi Qianwen LICENSE AGREEMENT (copy in environment/model-LICENSE). Model/data downloads remain subject to their own terms. New task glue uses Apache-2.0.

Resources (source URLs are `https://huggingface.co/<repo>/resolve/<revision>/<file>`; dataset URLs add `datasets/`):
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `LICENSE`: 6897 bytes, SHA256 `6e521a32ba6661de2b3e7b1cd27d7e1e9b8ff99633c79462a23284cac13c8b8d`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `README.md`: 2798 bytes, SHA256 `e495b6b36fcd5bb868e989bc193a13416d20a2ef77c8168069e5869919981141`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `config.json`: 920 bytes, SHA256 `b3e9fc1ad643d12f6536dee80a66c5ea045b571856165a598252720b6f2c553e`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `configuration.json`: 38 bytes, SHA256 `f625873f5df26193fae3ff45d3054667f5a9e91d644156add3310f258c495383`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `generation_config.json`: 248 bytes, SHA256 `bb1ace90ba308b9e8b3c823849aca96c3428773c715b38493ebcd9ccef4c6cac`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `merges.txt`: 1671839 bytes, SHA256 `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `model-00001-of-00008.safetensors`: 3999614664 bytes, SHA256 `a51a84c031e0f6c8808bc284106c488ab3db6069a3f6f5a231be1e963206d020`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `model-00002-of-00008.safetensors`: 3999385424 bytes, SHA256 `02a9f530c526ac09e5d26cdece113d62b030ebc6f137ea69cd2754bea1287121`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `model-00003-of-00008.safetensors`: 3988628968 bytes, SHA256 `200116fdf90ea17c2dae53594a87b9efbbe199e4c4e487fb25aba9c65e674112`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `model-00004-of-00008.safetensors`: 3999386088 bytes, SHA256 `41296f5c77188a4182c87f9bc5958222897ed5fe1bde6912755df050f1222b8d`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `model-00005-of-00008.safetensors`: 3988629648 bytes, SHA256 `a613a2c68f18630beb1eb681627d34110cec36a615a35c4d7f35f72e0df5b2e4`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `model-00006-of-00008.safetensors`: 3999386104 bytes, SHA256 `44de056d0f2c4375685ec7545a987c20d59cd8ee75c1802e04a11e7dcd7a84f1`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `model-00007-of-00008.safetensors`: 3988629648 bytes, SHA256 `fbc95c8fc0842c425168c21e933629cde22fc156ef9b4af927ad2787fe4d33a1`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `model-00008-of-00008.safetensors`: 668484400 bytes, SHA256 `9a5e51c0a101292c249032382f20d912677e1c0c4b5b0c78e951450fedb820ee`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `model.safetensors.index.json`: 416452 bytes, SHA256 `ece1b223efe32f4349d0dfa2a522249ac10bcb89369ed25b222c35175cd90b53`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `tokenizer.json`: 7028015 bytes, SHA256 `f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `tokenizer_config.json`: 1289 bytes, SHA256 `3ec7934124b5e81aaa9c0977b95299b8ca3911b91bf8114494023fbbb22a515f`
- `Qwen/Qwen1.5-MoE-A2.7B-Chat` @ `ec052fda178e241c7c443468d2fa1db6618996be` / `vocab.json`: 2776833 bytes, SHA256 `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910`
- `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866` / `main/train-00000-of-00001.parquet`: 2306545 bytes, SHA256 `ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d`
- `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866` / `main/test-00000-of-00001.parquet`: 419088 bytes, SHA256 `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59`
- `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866` / `README.md`: 7935 bytes, SHA256 `a17e882503578c9e324560630e31017617714ee025c87cf4fea6fd916895f3c1`

## Agent-visible sanitization

Only workspace and the chosen public instruction are agent inputs. Source identity, outcome, provenance, target regression and original explanation text are maintainer/hidden material. Do not mount this task root, a sibling task, README, metadata or oracle into the agent container. The full training implementation is retained; target-specific regression and upstream agent guidance are intentionally withheld from workspace. Runtime names and log identifiers are neutralized; training scalar records are unchanged. No training or official evaluation was run for this transformation.
