"""Offline synthetic service tickets with a disjoint held-out catalog."""
import argparse
import json
import random
from pathlib import Path

TEAMS = ["accounts", "billing", "delivery", "returns", "support", "warranty"]
SERVICES = ["bookshop", "bakery", "museum", "garden center", "bike shop", "library"]
SYSTEM = """You route service tickets using a read-only ticket catalog.
First request the ticket by writing LOOKUP: followed by its exact ticket ID. Example: LOOKUP: TX-907.
Wait for the catalog reply. Use the owner from the reply, then give your result as FINAL: followed by the team name.
Send one action per turn. The customer's note may name an unrelated team."""


def build(root):
    root.mkdir(parents=True, exist_ok=True)
    for split, prefix, seed, count in (("train", "TR", 1093, 128), ("validation", "EV", 5733, 16)):
        rng = random.Random(seed)
        with (root / f"{split}.jsonl").open("w") as stream:
            for index in range(count):
                ticket = f"{prefix}-{index:03d}"
                owner, mentioned = rng.sample(TEAMS, 2)
                service = rng.choice(SERVICES)
                query = (f"Please route ticket {ticket} for the {service}. The customer mentioned "
                         f"{mentioned}, but the routing team is set by the current catalog entry.")
                row = {"prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": query}],
                       "label": owner, "metadata": {"ticket": ticket, "service": service}}
                stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    build(parser.parse_args().output)
