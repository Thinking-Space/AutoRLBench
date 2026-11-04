"""Deterministic SQLite episodes and a query/result evaluator, with no ML dependencies."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import random
import sqlite3
from typing import Any

SCHEMA = """
CREATE TABLE orders(order_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL,
 warehouse_id INTEGER NOT NULL, ordered_at TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('complete','pending','cancelled')));
CREATE TABLE order_items(item_id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL
 REFERENCES orders(order_id), quantity INTEGER NOT NULL, unit_price_cents INTEGER NOT NULL);
CREATE TABLE refunds(refund_id INTEGER PRIMARY KEY, item_id INTEGER NOT NULL
 REFERENCES order_items(item_id), amount_cents INTEGER NOT NULL);
CREATE TABLE shipments(shipment_id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL
 REFERENCES orders(order_id), quantity INTEGER NOT NULL, promised_day INTEGER NOT NULL);
CREATE TABLE shipment_events(event_id INTEGER PRIMARY KEY, shipment_id INTEGER NOT NULL
 REFERENCES shipments(shipment_id), event_kind TEXT NOT NULL, event_day INTEGER NOT NULL);
"""
FAMILIES = ("net_revenue", "late_delivery", "partial_fulfillment")
SYSTEM = (
    "You are an operations analyst. Use the supplied schema and the warehouse query tool to answer. "
    "Finish with one JSON object containing sql (a read-only SQLite query) and rows "
    "(the complete query result as a list of lists). Integer monetary amounts are cents. "
    "Do not hardcode result values into SQL; the query must also work on other valid databases. "
    "You may make at most five tool calls. After obtaining the correct rows, finish your answer; "
    "do not repeat a successful tool call. Check these SQL correctness rules: excluding cancelled "
    "orders includes pending orders unless the question explicitly requires complete orders; "
    "pre-aggregate one-to-many child tables before joining so values are not multiplied; "
    "missing refunds count as zero; use floating-point division before rounding a rate. "
    "The schema is:\n" + SCHEMA
)


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    family: str
    template: int
    seed: int
    question: str
    reference_sql: str


def database(seed: int, n_orders: int = 48) -> sqlite3.Connection:
    """Generate valid records, including multiple items/refunds/shipments and repeated events."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.executescript("PRAGMA foreign_keys=ON;" + SCHEMA)
    item_id = refund_id = shipment_id = event_id = 0
    for order_id in range(1, n_orders + 1):
        status = rng.choices(["complete", "pending", "cancelled"], [6, 3, 1])[0]
        day = rng.randrange(59)
        conn.execute("INSERT INTO orders VALUES (?,?,?,?,?)", (
            order_id, rng.randint(1, 8), rng.randint(1, 3),
            (date(2025, 1, 1) + timedelta(days=day)).isoformat(), status))
        quantity = 0
        for _ in range(rng.randint(1, 3)):
            item_id += 1
            qty, price = rng.randint(1, 4), rng.randint(1, 50) * 100
            quantity += qty
            conn.execute("INSERT INTO order_items VALUES (?,?,?,?)", (item_id, order_id, qty, price))
            remaining = qty * price
            for _ in range(rng.choices([0, 1, 2], [6, 2, 2])[0]):
                refund_id += 1
                amount = rng.randint(0, remaining // 100) * 100
                remaining -= amount
                conn.execute("INSERT INTO refunds VALUES (?,?,?)", (refund_id, item_id, amount))
        remaining_qty = quantity if status == "complete" else rng.randrange(quantity)
        while remaining_qty:
            shipment_id += 1
            qty = rng.randint(1, remaining_qty)
            remaining_qty -= qty
            promised = day + rng.randint(2, 7)
            conn.execute("INSERT INTO shipments VALUES (?,?,?,?)", (shipment_id, order_id, qty, promised))
            # Every generated shipment is delivered; pending orders may have unshipped units.
            delivered = promised + rng.choice([-1, 0, 0, 1, 3])
            for kind, event_day in [("dispatched", day), ("delivered", delivered)]:
                for _ in range(rng.randint(1, 2)):
                    event_id += 1
                    conn.execute("INSERT INTO shipment_events VALUES (?,?,?,?)", (
                        event_id, shipment_id, kind, event_day))
    conn.commit()
    return conn


def make_case(split: str, index: int) -> Case:
    if split not in ("train", "dev", "test") or index < 0:
        raise ValueError("Expected train/dev/test and a nonnegative index")
    family = FAMILIES[index % len(FAMILIES)]
    template = (index // 3) % 2 if split == "train" else (2 if split == "dev" else 3)
    seed = {"train": 100_000, "dev": 200_000, "test": 300_000}[split] + index
    # Templates partition grouping structures, rather than paraphrases of one query.
    dimensions = ["o.warehouse_id", "o.warehouse_id, substr(o.ordered_at,1,7)",
                  "o.customer_id", "o.customer_id, substr(o.ordered_at,1,7)"]
    labels = ["warehouse_id", "warehouse_id and order month (YYYY-MM)",
              "customer_id", "customer_id and order month (YYYY-MM)"]
    group, label = dimensions[template], labels[template]
    common = "Exclude cancelled orders. Include only groups with at least one qualifying order. "
    if family == "net_revenue":
        question = (f"For each {label}, return net revenue in integer cents: sum(quantity * "
                    "unit_price_cents) minus all refunds for those items. " + common +
                    "Count each item once even when it has multiple refunds. Output the group columns then net_cents.")
        query = f"""WITH refunded AS (
 SELECT item_id, SUM(amount_cents) amount FROM refunds GROUP BY item_id
), order_value AS (
 SELECT i.order_id, SUM(i.quantity*i.unit_price_cents-COALESCE(r.amount,0)) net
 FROM order_items i LEFT JOIN refunded r ON r.item_id=i.item_id GROUP BY i.order_id
)
SELECT {group}, SUM(v.net) AS net_cents FROM orders o JOIN order_value v USING(order_id)
WHERE o.status != 'cancelled' GROUP BY {group}"""
    elif family == "late_delivery":
        question = (f"For each {label}, return the late-delivery rate of complete orders, in integer "
                    "basis points (10000 means 100%, round to nearest integer). An order is late if any "
                    "shipment's last delivered event_day exceeds promised_day. Repeated events and split "
                    "shipments count only once per order. Only status='complete' qualifies. " + common +
                    "Output the group columns then late_basis_points.")
        query = f"""WITH delivered AS (
 SELECT shipment_id, MAX(event_day) day FROM shipment_events
 WHERE event_kind='delivered' GROUP BY shipment_id
), late AS (
 SELECT s.order_id, MAX(CASE WHEN d.day>s.promised_day THEN 1 ELSE 0 END) flag
 FROM shipments s JOIN delivered d USING(shipment_id) GROUP BY s.order_id
)
SELECT {group}, CAST(ROUND(10000.0*SUM(l.flag)/COUNT(*)) AS INTEGER) late_basis_points
FROM orders o JOIN late l USING(order_id) WHERE o.status='complete' GROUP BY {group}"""
    else:
        question = (f"For each {label}, return the number of partially fulfilled orders and total "
                    "remaining units in those orders. Partially fulfilled means shipped quantity is "
                    "strictly greater than zero and strictly less than ordered quantity. Sum shipment "
                    "quantities once per shipment, regardless of event count. " + common +
                    "Output group columns, partial_orders, remaining_units.")
        query = f"""WITH quantities AS (
 SELECT order_id, SUM(quantity) ordered FROM order_items GROUP BY order_id
), sent AS (
 SELECT order_id, SUM(quantity) shipped FROM shipments GROUP BY order_id
)
SELECT {group}, COUNT(*) partial_orders, SUM(q.ordered-s.shipped) remaining_units
FROM orders o JOIN quantities q USING(order_id) JOIN sent s USING(order_id)
WHERE o.status!='cancelled' AND s.shipped>0 AND s.shipped<q.ordered GROUP BY {group}"""
    return Case(f"{split}-{index:05d}", split, family, template, seed, question, query)


def execute(conn: sqlite3.Connection, query: str, max_rows: int = 1000) -> list[list[Any]]:
    """Bounded, read-only execution; timeout is measured in SQLite VM operations."""
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
               sqlite3.SQLITE_RECURSIVE}
    def authorize(action, arg1, arg2, _db, _trigger):
        if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in {"load_extension", "readfile", "writefile"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
    ticks = 0
    def progress():
        nonlocal ticks
        ticks += 1
        return int(ticks > 1000)
    conn.set_authorizer(authorize)
    conn.set_progress_handler(progress, 1000)
    try:
        cursor = conn.execute(query)
        if cursor.description is None:
            raise ValueError("A SELECT result is required")
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError("Result exceeds row limit")
        if any(isinstance(v, (bytes, bytearray)) for row in rows for v in row):
            raise ValueError("Binary values are not supported")
        return [list(row) for row in rows]
    finally:
        conn.set_authorizer(None)
        conn.set_progress_handler(None, 0)


def same_rows(left: list, right: list) -> bool:
    # These tasks specify integer outputs; multiset equality preserves duplicate rows.
    def normalize(rows):
        if not isinstance(rows, list) or any(not isinstance(r, (list, tuple)) for r in rows):
            raise ValueError("Expected rows")
        normalized = []
        for row in rows:
            converted = []
            for value in row:
                if isinstance(value, bool) or not isinstance(value, (int, float, str, type(None))):
                    raise ValueError("Unsupported cell")
                if isinstance(value, float):
                    if not value.is_integer():
                        raise ValueError("Expected integer-valued numeric result")
                    value = int(value)
                converted.append(value)
            normalized.append(tuple(converted))
        return Counter(normalized)
    try:
        return normalize(left) == normalize(right)
    except (ValueError, TypeError, OverflowError):
        return False


def parse_answer(text: str) -> dict:
    decoder = json.JSONDecoder()
    answers = []
    for pos, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[pos:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("sql"), str) and isinstance(obj.get("rows"), list):
            answers.append(obj)
    if not answers:
        raise ValueError("Missing final JSON with sql and rows")
    return answers[-1]


def grade(case: Case, answer: str | dict, perturbations: int = 3) -> dict:
    try:
        obj = parse_answer(answer) if isinstance(answer, str) else answer
        if not isinstance(obj.get("sql"), str) or not isinstance(obj.get("rows"), list):
            raise ValueError("Expected sql and rows")
        for j in range(perturbations + 1):
            conn = database(case.seed + 1_000_003 * j)
            try:
                expected = execute(conn, case.reference_sql)
                actual = execute(conn, obj["sql"])
            finally:
                conn.close()
            if not same_rows(actual, expected):
                return {"score": 0.0, "reason": "query_result_mismatch"}
            if j == 0 and not same_rows(obj["rows"], expected):
                return {"score": 0.0, "reason": "answer_mismatch"}
        return {"score": 1.0, "reason": "correct"}
    except (sqlite3.Error, ValueError, TypeError, AttributeError, KeyError) as exc:
        return {"score": 0.0, "reason": "invalid_answer", "error": str(exc)[:200]}


class Warehouse:
    def __init__(self, case: Case):
        self.case = case
        self.connection = database(case.seed)
        self.calls = 0

    def call(self, action: str, query: str = "") -> dict:
        self.calls += 1
        if self.calls > 5:
            return {"error": "Tool-call budget exhausted"}
        if action == "schema":
            return {"schema": SCHEMA}
        if action != "query":
            return {"error": "action must be schema or query"}
        try:
            return {"rows": execute(self.connection, query, max_rows=80)}
        except (sqlite3.Error, ValueError) as exc:
            return {"error": str(exc)[:200]}

    def close(self):
        self.connection.close()


def write_dataset(destination: Path, counts=(512, 96, 128)) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "generator": "sql_credit.workload", "splits": {}}
    for split, count in zip(("train", "dev", "test"), counts, strict=True):
        cases = [make_case(split, i) for i in range(count)]
        content = "".join(json.dumps(asdict(case), sort_keys=True) + "\n" for case in cases)
        path = destination / f"{split}.jsonl"
        path.write_text(content)
        manifest["splits"][split] = {"count": count, "sha256": hashlib.sha256(content.encode()).hexdigest(),
                                     "bytes": len(content.encode()), "templates": sorted({c.template for c in cases})}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.output), indent=2))
