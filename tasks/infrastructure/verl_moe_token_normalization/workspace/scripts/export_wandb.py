"""Export local offline W&B histories without a server, login, or sync."""
import argparse
import json
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal.datastore import DataStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    files = sorted(args.run_dir.glob("wandb/**/run-*.wandb"))
    if not files:
        raise SystemExit("No offline W&B record found")
    rows = []
    for file in files:
        store = DataStore()
        store.open_for_scan(str(file))
        try:
            while True:
                payload = store.scan_data()
                if payload is None:
                    break
                record = wandb_internal_pb2.Record()
                record.ParseFromString(payload)
                if record.HasField("history"):
                    row = {}
                    for item in record.history.item:
                        key = item.key or "/".join(item.nested_key)
                        row[key] = json.loads(item.value_json)
                    if row:
                        rows.append(row)
        finally:
            store.close()
    target = args.run_dir / "wandb-history.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps(dict(files=len(files), records=len(rows), output=str(target))))


if __name__ == "__main__":
    main()
