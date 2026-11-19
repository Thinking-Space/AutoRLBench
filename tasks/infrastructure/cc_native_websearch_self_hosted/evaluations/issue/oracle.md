# Canonical issue oracle

Historical snapshot, no repaired control. The contributor supplied the cause before the run: native CC search shares the configured model base URL, whereas the self-hosted inference backend does not provide Anthropic's server-side search service. The recorded source, original requests, backend converter and real RL run substantiate that cause.

Expected verdict: **unhealthy** for the requested search-enabled RL experiment.

## Root cause and causal chain

1. In `code/cc_search_rollout.py`, `generate` sets `ANTHROPIC_BASE_URL` to the recording adapter and enables the real `WebSearch` tool. Ordinary model turns use the original `AnthropicAdapter`, generating real actor tokens through SGLang.
2. `RecordingAdapter.dispatch` detects declarations whose types start with `web_search_` or `web_fetch_`. It sends the request to an actual SGLang worker's `/v1/messages`, replacing the model alias. This path has no implementation or routing to an actual Anthropic-compatible server-tool search provider. Changing the client model base URL alone cannot supply that capability.
3. The native CC 2.1.300 request is `POST /v1/messages?beta=true`, with `tools: [{type: web_search_20250305, name: web_search, max_uses: 8}]` and `tool_choice: {type: tool, name: web_search}`. Do not require a nonexistent `/search` endpoint as the cause.
4. In the installed `sglang/srt/entrypoints/anthropic/serving.py`, `handle_messages` invokes `_convert_to_chat_completion_request`; the converter filters Anthropic server built-ins because its OpenAI-compatible backend cannot invoke them. A forced tool choice with no remaining custom tools raises `ValueError`, returned as HTTP 400. This is an unsupported provider capability, not a sampling or optimizer parameter error.

Expected behavior, not a tested control: a compatible configured tool provider must execute the native search request and return valid search results, or the integration must explicitly route/translate it to a real search-capable service. No fixed workspace or measured post-fix score exists. Do not grade a specific repair implementation.

## Observable impact

- `outputs/native-search.jsonl`: 891 original/forwarded request and actual worker response records; all statuses 400; original paths `/v1/messages?beta=true`. The response explicitly identifies unsupported server-side Anthropic built-ins.
- `outputs/cc-samples/`: first, middle and last selected training sample transcripts, input, token-turn records and outcomes. Tool failures are visible to real CC.
- `outputs/outcomes.jsonl` and `experiment.json`: 300 outcomes, 299 with trainable tokens, 895 outer calls. Background production differs from the 96 samples consumed for training; do not equate these counts.
- `outputs/rollouts/rollout_{0,1,2}.json`: three readable batches of 32 samples, including masks/rewards/logprobs. `actor-token-verification.json` checks all 96 selected samples and 25,190 effective tokens against their original SGLang outputs.
- `outputs/training-metrics.json` and full `run.log`: step 0 gradient norm 0; step 1 2.405400311945096; step 2 1.7096229973075858. Raw reward means are 0, 0.025 and 0.06875. The latter batches have 3 and 2 prompt groups with reward variation.
- Save log reports successful final iteration 2. Later NCCL errors and the summary KeyError are real additional runtime/reporting issues, not explanations for the hundreds of earlier identical native-tool 400s.

## Plausible wrong diagnoses

- The client needs `/search`: contradicted by the actual Messages request path and converter error.
- Bad weights or no model inference: normal `/generate` succeeds and the selected actor tokens/logprobs are original.
- RL never starts or every gradient is zero: two finite nonzero gradient norms and a final save are recorded.
- A rising reward means search worked: exact answers may come from model knowledge and reward includes a format component; every recorded search still fails.
- NCCL or the summary KeyError caused the native search failure: these appear during/after finalization, after many identical tool errors.
- Add a dummy endpoint or invent search results: does not explain/provide the missing native server-tool capability.

There is one primary causal incompatibility. The client routing boundary and backend conversion are linked parts of that cause; mentioning only one without explaining the other merits partial root-cause credit. A correct diagnosis need not explain the incidental finalization problems in full, but must not attribute the native-search failure to them or conceal them as a clean successful run.
