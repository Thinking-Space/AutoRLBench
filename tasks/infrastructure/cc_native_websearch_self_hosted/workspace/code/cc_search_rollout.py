"""Search-R1 QA in real Claude Code; train only real actor-sampled tokens.

The normal Messages path uses slime's existing token-capturing adapter. Native
server-tool requests are forwarded unchanged (except model alias) to the actual
SGLang worker. No search error or model response is scripted.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import secrets
import string
import time

import aiohttp
from aiohttp import web
from slime.agent.adapters import AnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer

LOG = logging.getLogger(__name__)
SYSTEM = ('You answer factual questions with the WebSearch tool. Use WebSearch at least once '
          'before giving your final answer. If the tool fails, do not invent search results; '
          'do not retry a failed search. Answer from your own knowledge, or state that you do not know. '
          'Put only the short final answer inside '
          '<answer> and </answer>. Do not use XML search actions; call the actual WebSearch tool.')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


class RecordingAdapter(AnthropicAdapter):
    def __init__(self, *, evidence_root, **kwargs):
        self.evidence_root = evidence_root
        self.counts = {}
        self.native_url = None
        super().__init__(debug_callback=self.record_tokens, **kwargs)

    def _register_routes(self, app):
        app.router.add_post('/v1/messages', self.dispatch)
        # Token counting is local and does not execute a model/tool.
        from slime.agent.adapters.anthropic import _count_tokens
        app.router.add_post('/v1/messages/count_tokens', _count_tokens)

    def record_tokens(self, sid, translated, tools, response, turn):
        target = self.evidence_root / sid / 'token-turns.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('a') as stream:
            stream.write(json.dumps({'prompt_ids': turn.prompt_ids, 'output_ids': turn.output_ids,
                                     'output_log_probs': turn.output_log_probs, 'finish_reason': turn.finish_reason,
                                     'assistant': response}, ensure_ascii=False) + '\n')

    async def worker_url(self):
        if self.native_url:
            return self.native_url
        async with aiohttp.ClientSession(trust_env=False) as client:
            for path in ('/workers', '/v1/workers'):
                try:
                    async with client.get(self.sglang_url + path, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status != 200:
                            continue
                        data = await response.json(content_type=None)
                    entries = data if isinstance(data, list) else (data.get('urls') or data.get('workers') or data.get('data') or [])
                    for entry in entries:
                        url = entry if isinstance(entry, str) else (entry.get('url') or entry.get('worker_url'))
                        if url and url.startswith('http'):
                            self.native_url = url.rstrip('/')
                            return self.native_url
                except (aiohttp.ClientError, ValueError, asyncio.TimeoutError):
                    continue
        raise RuntimeError('Cannot discover actual SGLang worker; refusing to substitute a search error')

    async def dispatch(self, request):
        body = await request.json()
        sid = self._session_id(request, body)
        index = self.counts.get(sid, 0)
        self.counts[sid] = index + 1
        evidence = {'method': request.method, 'path': request.path_qs, 'body': body}
        path = self.evidence_root / sid / f'request-{index:03d}.json'
        is_server_tool = any(str(tool.get('type', '')).startswith(('web_search_', 'web_fetch_'))
                             for tool in body.get('tools', []))
        if is_server_tool:
            upstream = await self.worker_url()
            forwarded = dict(body, model=os.environ.get('CC_ACTOR_MODEL', 'cc-search-r1-actor'))
            async with aiohttp.ClientSession(trust_env=False) as client:
                async with client.post(upstream + '/v1/messages', json=forwarded,
                                       timeout=aiohttp.ClientTimeout(total=90)) as response:
                    raw = await response.read()
                    status = response.status
                    content_type = response.headers.get('Content-Type', 'application/json')
            evidence.update(upstream=upstream + '/v1/messages', forwarded_body=forwarded,
                            status=status, response_body=raw.decode(errors='replace'),
                            origin='actual_sglang_worker')
            dump(path, evidence)
            LOG.info('[native_websearch] sid=%s status=%s upstream=%s', sid, status, upstream)
            return web.Response(status=status, body=raw, headers={'Content-Type': content_type})
        dump(path, evidence)
        response = await super()._run_turn(request)
        evidence.update(status=response.status, origin='token_capturing_actor_adapter')
        dump(path, evidence)
        return response


class State(metaclass=SingletonMeta):
    def __init__(self, args):
        self.args = args
        self.root = Path(os.environ['CC_SEARCH_RUN_ROOT'])
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=False)
        self.adapter = RecordingAdapter(
            tokenizer=self.tokenizer,
            sglang_url=f'http://{args.sglang_router_ip}:{args.sglang_router_port}',
            tool_parser='qwen3_coder', reasoning_parser='qwen3',
            max_turns_per_sid=12,
            evidence_root=self.root / 'cc-trajectories',
        )
        self.handle = run_app_in_thread(self.adapter.app, host='127.0.0.1', port=0,
                                       runner_kwargs={'handler_cancellation': True, 'access_log_class': FilteredAccessLogger})
        self.base_url = f'http://127.0.0.1:{self.handle.port}'
        self.sem = asyncio.Semaphore(4)


def normalize(text):
    # Same normalization as the Search-R1 example's qa_em_format.em_check.
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_answer(text, targets):
    answers = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    answer = answers[-1].strip() if answers else ''
    if isinstance(targets, str):
        targets = [targets]
    correct = bool(answer) and any(normalize(answer) == normalize(str(target)) for target in targets)
    return (1.0 if correct else (0.2 if answer else 0.0)), answer, correct


async def generate(args, base_sample, sampling_params, evaluation=False):
    state = State(args)
    sid = f'ccqa-{base_sample.group_index}-{base_sample.index}-{secrets.token_hex(5)}'
    base_sample.session_id = sid
    state.adapter.open_session(sid, sampling_defaults=sampling_params, max_context_tokens=8192)
    folder = state.root / 'cc-trajectories' / sid
    folder.mkdir(parents=True, exist_ok=True)
    question = (base_sample.metadata or {}).get('question') or str(base_sample.prompt)
    prompt = f'Question: {question}\nUse the native WebSearch tool first, then answer inside <answer>...</answer>.'
    command = [os.environ['CC_BIN'], '-p', prompt, '--model', 'claude-sonnet-4-6',
               '--system-prompt', SYSTEM, '--tools', 'WebSearch', '--allowedTools', 'WebSearch',
               '--permission-mode', 'bypassPermissions', '--setting-sources', '',
               '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
               '--disable-slash-commands', '--no-session-persistence', '--max-turns', '3',
               '--output-format', 'stream-json', '--verbose']
    env = {name: os.environ[name] for name in ('PATH', 'HOME', 'LANG', 'LD_LIBRARY_PATH') if name in os.environ}
    env.update(ANTHROPIC_API_KEY=sid, ANTHROPIC_BASE_URL=state.base_url,
               CLAUDE_CONFIG_DIR=str(folder / 'cc-config'), IS_SANDBOX='1',
               CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', DISABLE_AUTOUPDATER='1',
               NO_PROXY='127.0.0.1,localhost')
    dump(folder / 'input.json', {'question': question, 'targets': base_sample.label,
                               'source': base_sample.metadata, 'command': command})
    started = time.monotonic()
    process = None
    try:
        async with state.sem:
            process = await asyncio.create_subprocess_exec(*command, cwd=folder, env=env,
                                                          stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=180)
            except asyncio.TimeoutError:
                process.kill()
                stdout, stderr = await process.communicate()
        (folder / 'cc.stdout.jsonl').write_bytes(stdout)
        (folder / 'cc.stderr.log').write_bytes(stderr)
        messages, final, tool_errors, tool_calls = [], '', 0, 0
        for line in stdout.decode(errors='replace').splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('type') == 'assistant':
                text = []
                for block in event.get('message', {}).get('content', []):
                    if block.get('type') == 'text':
                        text.append(block.get('text', ''))
                    if block.get('type') == 'tool_use' and block.get('name') == 'WebSearch':
                        tool_calls += 1
                if text:
                    final = '\n'.join(text)
            if event.get('type') == 'user':
                tool_errors += sum(bool(block.get('is_error')) for block in event.get('message', {}).get('content', [])
                                   if block.get('type') == 'tool_result')
        reward, answer, correct = score_answer(final, base_sample.label)
        details = {'question': question, 'final_text': final, 'answer': answer, 'exact_match': correct,
                   'reward': reward, 'native_websearch_calls': tool_calls, 'tool_errors': tool_errors,
                   'cc_exit_code': process.returncode, 'elapsed_s': time.monotonic() - started,
                   'sampling_parameters': sampling_params, 'model': 'Qwen/Qwen3.5-9B'}
        samples = await state.adapter.finish_session(sid, base_sample=base_sample, reward=reward,
                                                     extra_metadata=details, wait_timeout=30)
        if not samples:
            dump(folder / 'outcome.json', details)
            raise RuntimeError(f'CC produced no trainable actor tokens; sid={sid}, exit={process.returncode}')
        for sample in samples:
            assert sample.response_length == len(sample.loss_mask) == len(sample.rollout_log_probs)
            assert sample.response_length <= len(sample.tokens)
            assert sum(sample.loss_mask) > 0
        details.update(segments=len(samples), trainable_tokens=sum(sum(s.loss_mask) for s in samples))
        dump(folder / 'outcome.json', details)
        LOG.info('[cc_search_r1] %s reward=%.2f EM=%s searches=%d errors=%d segments=%d tokens=%d elapsed=%.1f',
                 sid, reward, correct, tool_calls, tool_errors, len(samples), details['trainable_tokens'], details['elapsed_s'])
        return samples
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        await state.adapter.drop_session(sid, wait_timeout=30)


async def reward_func(args, sample, **kwargs):
    # Slime calls custom reward functions with a batch, including an empty
    # batch when finish_session has already attached every outcome reward.
    def value(item):
        if item.reward is None:
            raise ValueError('CC rollout did not assign an outcome reward')
        return float(item.reward)

    return [value(item) for item in sample] if isinstance(sample, list) else value(sample)
