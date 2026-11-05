import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workspace"))

import json
import sqlite3

import pytest

from reproductions.sql_credit.workload import Warehouse, database, execute, grade, make_case, same_rows, write_dataset


@pytest.mark.parametrize("split", ["train", "dev", "test"])
@pytest.mark.parametrize("index", range(6))
def test_reference_passes_original_and_perturbed_databases(split, index):
    case = make_case(split, index)
    conn = database(case.seed)
    try:
        rows = execute(conn, case.reference_sql)
    finally:
        conn.close()
    assert grade(case, {"sql": case.reference_sql, "rows": rows})["score"] == 1


def test_hardcoded_correct_answer_fails_perturbation():
    case = make_case("train", 0)
    conn = database(case.seed)
    rows = execute(conn, case.reference_sql)
    conn.close()
    sql = " UNION ALL ".join("SELECT " + ",".join(map(str, row)) for row in rows)
    assert grade(case, {"sql": sql, "rows": rows}, perturbations=0)["score"] == 1
    assert grade(case, {"sql": sql, "rows": rows})["score"] == 0


def test_join_fanout_is_rejected():
    case = make_case("train", 0)
    sql = """SELECT o.warehouse_id, SUM(i.quantity*i.unit_price_cents-COALESCE(r.amount_cents,0))
    FROM orders o JOIN order_items i USING(order_id) LEFT JOIN refunds r USING(item_id)
    WHERE o.status!='cancelled' GROUP BY o.warehouse_id"""
    conn = database(case.seed)
    rows = execute(conn, sql)
    conn.close()
    assert grade(case, {"sql": sql, "rows": rows})["score"] == 0


@pytest.mark.parametrize("sql", ["DELETE FROM orders", "ATTACH DATABASE ':memory:' AS x", "PRAGMA user_version=7",
                                   "SELECT 1; SELECT 2", "SELECT load_extension('anything')"])
def test_queries_cannot_mutate_or_attach(sql):
    conn = database(1)
    with pytest.raises((sqlite3.Error, ValueError)):
        execute(conn, sql)
    assert len(execute(conn, "SELECT * FROM orders")) == 48
    conn.close()


def test_query_work_budget():
    conn = database(1)
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        execute(conn, "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n")
    conn.close()


def test_multiset_semantics_and_answer_check():
    assert not same_rows([[1], [1]], [[1]])
    assert not same_rows([[1.01]], [[1]])
    assert not same_rows([[True]], [[1]])
    assert same_rows([[2], [1.0]], [[1], [2]])
    case = make_case("train", 0)
    assert grade(case, {"sql": case.reference_sql, "rows": []})["score"] == 0


def test_tool_limit_and_deterministic_dataset(tmp_path):
    env = Warehouse(make_case("train", 0))
    assert "schema" in env.call("schema")
    for _ in range(4):
        assert env.call("query", "SELECT 1") == {"rows": [[1]]}
    assert "error" in env.call("query", "SELECT 1")
    env.close()
    a = write_dataset(tmp_path / "a", (12, 6, 6))
    b = write_dataset(tmp_path / "b", (12, 6, 6))
    assert a == b
    assert a["splits"]["train"]["templates"] == [0, 1]
    assert a["splits"]["dev"]["templates"] == [2]
    assert a["splits"]["test"]["templates"] == [3]
    assert json.loads((tmp_path / "a" / "train.jsonl").read_text().splitlines()[0])["seed"] == 100000
