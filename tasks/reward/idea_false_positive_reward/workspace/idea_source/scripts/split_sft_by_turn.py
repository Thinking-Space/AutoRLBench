import json
from pathlib import Path


SRC = Path('outputs/awm_qwen38_27b_rollouts_20pct_20260909/sft_rollouts.jsonl')
DST = Path('outputs/awm_qwen38_27b_rollouts_20pct_20260909/sft_rollouts_turn_level.jsonl')


def main():
    total_trajectories = 0
    total_samples = 0
    with SRC.open() as source, DST.open('w') as target:
        for line in source:
            row = json.loads(line)
            messages = row.get('messages', [])
            total_trajectories += 1

            # Each assistant turn becomes the last message and therefore the
            # supervised target for one independent SFT example.
            for index, message in enumerate(messages):
                if message.get('role') != 'assistant':
                    continue
                example = {'messages': messages[: index + 1]}
                target.write(json.dumps(example, ensure_ascii=False) + '\n')
                total_samples += 1

    print(
        f'trajectories={total_trajectories} samples={total_samples} '
        f'output={DST}'
    )


if __name__ == '__main__':
    main()
