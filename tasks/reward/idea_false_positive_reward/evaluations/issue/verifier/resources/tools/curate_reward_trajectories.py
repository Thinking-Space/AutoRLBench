#!/usr/bin/env python3
"""Replay the current 8 audited cases; never select or rewrite historical samples."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import re
import sqlite3
from trajectory_utils import digest_bytes, digest_file, episode_records, transcript, open_original, evidence_paths

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / "task.toml").is_file())
RULES = {"dashboard": ("task_management_16", 8)}

def records(path):
    grouped = defaultdict(list)
    with open_original(path, "rb") as stream:
        for number, raw in enumerate(stream, 1):
            row = json.loads(raw)
            grouped[row["traj_uid"]].append((number, raw, row))
    return grouped

def json_records(path):
    with open_original(path) as stream:
        for number, line in enumerate(stream, 1):
            yield number, json.loads(line)

def readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)

def parse_tool_calls(messages):
    names, invalid = [], []
    for message in messages:
        if not message["label"].startswith("Assistant"):
            continue
        for raw in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", message["text"], re.S):
            try:
                value = json.loads(raw)
                args = value.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                names.append(args.get("tool_name", value["name"]))
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                invalid.append({"message": message["label"], "error": str(error), "source_preserved": True})
    return names, invalid

def materialize(path, tables):
    with sqlite3.connect(path) as conn:
        for table in tables:
            conn.execute(table["ddl"])
            if table["rows"]:
                placeholders = ",".join("?" for _ in table["rows"][0])
                conn.executemany('INSERT INTO "' + table["table"] + '" VALUES (' + placeholders + ')', table["rows"])

def semantic_observations(kind, path, row, calls):
    with readonly(path) as conn:
        if kind == "dashboard":
            ids = [x[0] for x in conn.execute("SELECT id FROM dashboards WHERE created_by=1 AND name='Marketing Campaign Overview'")]
            counts = [[i, conn.execute("SELECT count(*) FROM widgets WHERE dashboard_id=?", (i,)).fetchone()[0], conn.execute("SELECT count(*) FROM dashboard_boards WHERE dashboard_id=?", (i,)).fetchone()[0]] for i in ids]
            assert all(x[1:] == [0, 0] for x in counts)
            assert not any(x.endswith(("create_widget", "attach_board_to_dashboard")) for x in calls)
            return {"current_user_dashboard_widget_link_counts": counts, "current_user_id": 1, "seed_dashboard": conn.execute("SELECT id,name,created_by FROM dashboards WHERE id=1").fetchall()}
        raise ValueError('Unsupported archived-DB case: ' + kind)

def artifact(project, path):
    with open_original(path, 'rb') as stream:
        data = stream.read()
    return {"path": path.relative_to(project).as_posix(), "sha256": digest_bytes(data), "bytes": len(data)}


def replay(project=PROJECT):
    manifest = json.loads((project / "evaluations/issue/verifier/resources/curated_trajectory_manifest.json").read_text())
    if manifest["schema_version"] != 2:
        raise ValueError("Expected the current consolidated v2 evidence manifest")
    from reward_replay import replay_v2
    return replay_v2(project, manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Replay immutable evidence (the default)")
    parser.parse_args()
    print(json.dumps(replay(), indent=2))
