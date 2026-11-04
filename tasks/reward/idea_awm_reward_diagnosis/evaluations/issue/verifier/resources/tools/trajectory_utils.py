"""Shared readers and hashes for immutable historical evidence."""
import hashlib
import json
import re

def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()

def digest_file(path):
    result = hashlib.sha256()
    with open_original(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def open_original(path, mode='rt'):
    """Read an existing original file directly; no extraction or fallback storage."""
    return path.open(mode)


def evidence_paths(directory):
    """Directly readable JSONL files in the evidence folder."""
    return sorted(path for path in directory.glob('*.jsonl') if path.is_file())

def episode_records(project, case):
    """Resolve one episode from the shared step log, without a duplicate file."""
    path = project / case["artifacts"][0]["path"]
    rows = []
    with open_original(path, "rb") as stream:
        for number, raw in enumerate(stream, 1):
            row = json.loads(raw)
            if row.get("traj_uid") == case["traj_uid"]:
                rows.append((number, raw, row))
    if not rows:
        raise ValueError("Episode missing from shared rollout: " + case["traj_uid"])
    expected = case.get("episode_records_sha256")
    if expected and digest_bytes(b"".join(row[1] for row in rows)) != expected:
        raise ValueError("Episode content changed: " + case["traj_uid"])
    return rows


def transcript(record):
    text = record["obs_text_base"]
    task, history = text.split("User task:\n", 1)[1].split("\nInteraction history:\n", 1)
    chunks = re.split(r"(?:^|\n)(Assistant \d+:|Tool \d+:)\n", history)
    messages = []
    for index in range(1, len(chunks), 2):
        messages.append({"label": chunks[index], "text": chunks[index + 1].strip()})
    messages.append({"label": "Final answer:", "text": record["output"]})
    return task.strip(), messages
