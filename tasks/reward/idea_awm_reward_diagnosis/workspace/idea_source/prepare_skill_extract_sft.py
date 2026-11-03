import json, random
from pathlib import Path
import pandas as pd

src = Path('/workspace/idea_source/outputs/awm_qwen35_rollouts_8_per_task/skill_extract.jsonl')
out = Path('/workspace/idea_source/outputs/awm_qwen35_rollouts_8_per_task/sft_parquet')
out.mkdir(parents=True, exist_ok=True)
rows = []
with src.open() as f:
    for line in f:
        if not line.strip():
            continue
        x = json.loads(line)
        if not x.get('parse_ok', False):
            continue
        task = str(x.get('task_description', '')).strip()
        skill = str(x.get('episode_skill', '')).strip()
        summary = str(x.get('episode_summary', '')).strip()
        if task and skill:
            rows.append({'prompt': 'Analyze the task below and return a concise reusable policy skill as JSON with episode_summary and episode_skill.\n\nTask: ' + task,
                         'response': json.dumps({'episode_summary': summary, 'episode_skill': skill}, ensure_ascii=False)})
random.Random(2026).shuffle(rows)
val = max(1, round(len(rows) * 0.1))
pd.DataFrame(rows[val:]).to_parquet(out / 'train.parquet', index=False)
pd.DataFrame(rows[:val]).to_parquet(out / 'val.parquet', index=False)
print(f'total={len(rows)} train={len(rows)-val} val={val} out={out}')
