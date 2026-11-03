import json
import re
from pathlib import Path

SRC = Path('outputs/awm_qwen38_27b_rollouts_20pct_20260909/baseline_rollouts.jsonl')
DST = Path('outputs/awm_qwen38_27b_rollouts_20pct_20260909/sft_rollouts.jsonl')


def clean_assistant(text: str) -> str:
    # Remove every hidden reasoning block, wherever it appears.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Some traces contain a closing tag without the opening tag.
    if '</think>' in text:
        text = text.split('</think>', 1)[1]
    return text.strip()


def compact_tool_result(content: str) -> str:
    # list_tools is environment boilerplate and can be omitted from training.
    if content.startswith('Available MCP Tools'):
        return ''
    try:
        obj = json.loads(content)
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    except Exception:
        return content.strip()


def convert(d):
    traj = d.get('trajectory')
    if not traj:
        return None
    source_messages = d.get('messages') or []
    system = next((m.get('content', '') for m in source_messages if m.get('role') == 'system'), '')
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': d.get('task') or d.get('task_description', '')})
    for step in traj:
        assistant = clean_assistant(step.get('content', ''))
        calls = step.get('tool_calls') or []
        # Remove the initial list_tools scaffolding (call and its giant response).
        real_calls = [c for c in calls if c.get('name') != 'list_tools']
        if real_calls:
            # Keep the model-facing XML convention used by the source traces.
            call_text = []
            for c in real_calls:
                args = c.get('arguments', {})
                call_text.append('<tool_call>\n' + json.dumps({'name': c.get('name'), 'arguments': args}, ensure_ascii=False) + '\n</tool_call>')
            messages.append({'role': 'assistant', 'content': '\n'.join(call_text)})
        elif assistant and '<tool_call>' not in assistant:
            messages.append({'role': 'assistant', 'content': assistant})
        response = step.get('tool_response') or {}
        result = compact_tool_result(response.get('content', ''))
        if result:
            messages.append({'role': 'tool', 'content': result})
    # Avoid records with no actual assistant supervision.
    if len(messages) < 2:
        return None
    # Keep the row to the standard chat-SFT schema; provenance stays in the
    # source file and is intentionally omitted for trainer compatibility.
    return {'messages': messages}


def main():
    DST.parent.mkdir(parents=True, exist_ok=True)
    total = kept = success = 0
    with SRC.open() as fi, DST.open('w') as fo:
        for line in fi:
            total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # SFT should learn completed, verified behavior; discard failed rollouts.
            if not d.get('success', False):
                continue
            out = convert(d)
            if out is None:
                continue
            kept += 1
            success += 1
            fo.write(json.dumps(out, ensure_ascii=False) + '\n')
    print(f'total={total} kept={kept} success={success} output={DST}')


if __name__ == '__main__':
    main()
