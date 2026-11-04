"""Reviewer analysis of raw traces; never used as an agent-editable success flag."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def trl(root):
    counts = defaultdict(int)
    trajectory_info = {}
    for record in records(root / "request_traces.jsonl"):
        if "server_prompt_ids" not in record:
            counts["incomplete_requests"] += 1
            continue
        expected = record["expected_prompt_ids"]
        actual = [prompt for prompt in record["server_prompt_ids"] for _ in range(record["client_n"])][:len(expected)]
        counts["server_client_token_disagreements"] += sum(a != b for a, b in zip(record["server_prompt_ids"], record["client_prompt_ids"], strict=True))
        counts[record["phase"] + "_rows"] += len(expected)
        counts[record["phase"] + "_mismatches"] += sum(a != b for a, b in zip(expected, actual, strict=True))
        groups = defaultdict(list)
        for row, wanted, got in zip(record["rows"], expected, actual, strict=True):
            groups[row // 4].append((wanted, got))
        if record["phase"] == "continuation":
            for group in groups.values():
                counts["continuation_groups"] += 1
                counts["affected_continuation_groups"] += any(a != b for a, b in group)
                if len({tuple(a) for a, _ in group}) > 1:
                    counts["divergent_groups"] += 1
                    counts["rows_in_divergent_groups"] += len(group)
                    counts["mismatches_in_divergent_groups"] += sum(a != b for a, b in group)
        for trajectory_id in record["trajectory_ids"]:
            trajectory_info[trajectory_id] = (record["mode"], record["step"])
    rewards = defaultdict(lambda: {"trajectories": 0, "successes": 0, "bugs": set(), "repaired_bugs": set()})
    for record in records(root / "tool_traces.jsonl"):
        if record["event"] != "grade":
            continue
        mode, step = trajectory_info.get(record["trajectory_id"], ("unknown", -1))
        values = rewards[f"{mode}_step_{step}"]
        values["trajectories"] += 1
        values["successes"] += int(record["reward"] == 1)
        values["bugs"].add(record["case_id"])
        if record["reward"] == 1:
            values["repaired_bugs"].add(record["case_id"])
    return {"correspondence": dict(counts), "rewards": {
        key: {**value, "bugs": sorted(value["bugs"]), "repaired_bugs": sorted(value["repaired_bugs"])}
        for key, value in rewards.items()}}


def sql(root):
    counts = defaultdict(int)
    examples = []
    for call in records(root / "credit_traces.jsonl"):
        for row in call["rows"]:
            counts["traced_rows"] += 1
            if sum(row["rewards"]) <= 0:
                continue
            mask = row["mask"]
            active = [i for i, value in enumerate(mask) if value]
            observations = [i for i in range(active[0], active[-1]) if not mask[i]] if active else []
            if not observations:
                continue
            counts["rewarded_tool_rows"] += 1
            early = [i for i in active if i < max(observations)]
            counts["early_action_tokens"] += len(early)
            counts["early_action_zero_returns"] += sum(row["returns"][i] == 0 for i in early)
            if len(examples) < 6:
                examples.append({"call_index": call["call_index"], "row": row["row"],
                    "terminal_reward": sum(row["rewards"]), "early_action_tokens": len(early),
                    "early_return_min": min(row["returns"][i] for i in early),
                    "early_return_max": max(row["returns"][i] for i in early),
                    "early_advantage_min": min(row["advantages"][i] for i in early),
                    "early_advantage_max": max(row["advantages"][i] for i in early)})
    return {"credit": dict(counts), "examples": examples, "metrics": records(root / "metrics.jsonl")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["sql", "trl"])
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    report = sql(args.run) if args.task == "sql" else trl(args.run)
    (args.run / "reviewer_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "metrics"}, indent=2))
