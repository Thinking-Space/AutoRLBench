"""Current tool-history and database-fixture checks; no collection or sampling."""
import json
from pathlib import Path
import re
import tempfile
import curate_reward_trajectories as base

EXTRA = {'shipment_runtime': ('logistics_and_shipment_tracking_1', 0), 'watchlist_runtime': ('stock_trading_1', 2), 'firmware_no_action': ('iot_device_management_3', 5), 'device_count': ('subscription_management_4', 2), 'leaderboard': ('fundraising_donations_5', 9), 'thread_no_action': ('discussion_forum_2', 0)}

def rule_records(project):
    wanted = set(base.RULES.values()) | set(EXTRA.values())
    found = {}
    for number, row in base.json_records(project / "workspace/data/AgentWorldModel-1K/gen_verifier.pure_code.jsonl"):
        key = (row["scenario"], row["task_idx"])
        if key in wanted:
            found[key] = (number, row)
    assert set(found) == wanted
    return found

def get_verifier(rule):
    namespace = {}
    exec(compile(rule["verification"]["code"], "<reviewed read-only reward verifier>", "exec"), namespace)
    return namespace["verify_task_completion"]

def tool_payloads(messages):
    result = []
    for message in messages:
        if not message["label"].startswith("Tool") or message["label"] == "Tool 1:":
            continue
        text = message["text"]
        try:
            value, _ = json.JSONDecoder().raw_decode(text)
        except ValueError:
            value = None
        result.append({"label": message["label"], "text": text, "json": value})
    return result

def history_observations(kind, final, messages):
    calls, invalid = base.parse_tool_calls(messages)
    payloads = tool_payloads(messages)
    if kind in {"shipment_runtime", "watchlist_runtime"}:
        assert payloads and all(x["text"].startswith("Error:") for x in payloads)
        required_write = "create_shipment" if kind == "shipment_runtime" else "create_watchlist"
        if any(required_write in call for call in calls):
            assert any("not initialized" in x["text"] for x in payloads)
        assert re.search(r'cannot|could not|unable|failed|not .*created', final, re.I)
        return {"successful_non_catalog_tool_responses": 0, "error_tool_responses": len(payloads), "task_requires_new_creation": True, "conclusion": "No successful requested creation; full reward despite failed execution.", "idempotence_boundary": "Counted only for explicit NEW-entity instructions and failed execution, not an ordinary already-satisfied update."}
    if kind == "thread_no_action":
        assert calls and all(x == "list_tools" for x in calls) and not invalid
        assert "tool" in final.lower() and re.search(r'list|available', final, re.I)
        return {"environment_action_calls": 0, "tool_catalog_only": True, "task_requires_new_thread_and_post": True, "final_answer_is_tool_listing_not_completion": True}
    if kind == "firmware_no_action":
        assert all(x == "list_tools" or "_list_" in x for x in calls)
        assert not invalid
        empty_resources = [key for x in payloads if isinstance(x["json"], dict) for key in ("device_models", "sites") if x["json"].get(key) == []]
        assert empty_resources
        assert re.search(r'cannot|could not|not .*schedul|not .*complet', final, re.I)
        return {"write_tool_calls": 0, "empty_resource_lookups": empty_resources, "task_requires_new_version_and_job": True}
    if kind == "device_count":
        slots = next(x["json"] for x in payloads if isinstance(x["json"], dict) and "remaining_device_slots" in x["json"])
        distinct = [x["json"] for x in payloads if isinstance(x["json"], dict) and "distinct_device_types" in x["json"]]
        assert slots["max_simultaneous_devices"] == 3 and slots["used_device_slots"] == 0 and slots["remaining_device_slots"] == 3
        assert distinct and all(x["count"] == 0 for x in distinct)
        answer_tail = re.split(r'Final [Aa]nswer\s*:', final)[-1]
        match = re.search(r'(\d+) additional devices', answer_tail)
        assert match and int(match[1]) in {0, 2}
        return {"current_window_tool_max": 3, "current_window_tool_used": 0, "correct_current_window_result": 3, "incorrect_final_result": int(match[1]), "recorded_window": [distinct[-1]["from_time"], distinct[-1]["to_time"]], "clock_caveat": "Old unfiltered play-session rows exist; current-window tools explicitly exclude them. Do not substitute today's date or latest-old-session date."}
    members = next(x["json"]["members"] for x in payloads if isinstance(x["json"], dict) and "members" in x["json"])
    assert len(members) == 6 and any(x["name"] == "Pat Chen" and x["donation_count"] == 4 for x in members)
    assert all(x["name"] in final for x in members)
    assert not re.search(r'(?:donations?\s*[:=]?\s*\d|\d\s+donations?)', final, re.I)
    return {"expected_member_name_amount_donation_count_rank": [[x["name"], x["total_raised"], x["donation_count"], x["rank"]] for x in members], "per_member_donation_counts_omitted": True}

def replay_v2(project, manifest):
    rules = rule_records(project)
    assert base.digest_file(project / "workspace/data/AgentWorldModel-1K/gen_verifier.pure_code.jsonl") == manifest["verifier_file_sha256"]
    results = []
    for case in manifest["cases"]:
        for item in case["artifacts"]:
            assert base.artifact(project, project / item["path"]) == item
        audit = json.loads((project / case["audit_path"]).read_text())
        rows = base.episode_records(project, case)
        assert all(x[2]["score"] == 1.0 for x in rows)
        task, messages = base.transcript(rows[-1][2])
        assert audit["final_answer"] == rows[-1][2]["output"]
        assert audit["task"] == task
        if case["kind"] in EXTRA:
            key = EXTRA[case["kind"]]
            assert task == rules[key][1]["task"]
            assert history_observations(case["kind"], audit["final_answer"], messages) == audit["observations"]
            initial_tables = final_tables = audit["seed_fixture_tables"]
        else:
            key = base.RULES[case["kind"]]
            initial_tables, final_tables = audit["initial_tables"], audit["final_tables"]
        with tempfile.TemporaryDirectory(prefix="reward-enrichment-replay-") as tmp:
            initial, final = Path(tmp) / "initial.db", Path(tmp) / "final.db"
            base.materialize(initial, initial_tables)
            base.materialize(final, final_tables)
            if case["kind"] not in EXTRA:
                observations = base.semantic_observations(case["kind"], final, {"output": audit["final_answer"]}, audit["actual_tool_calls"])
                assert json.loads(json.dumps(observations)) == audit["observations"]
            result = get_verifier(rules[key][1])(str(initial), str(final), audit["final_answer"])
            assert result == {"result": "complete"}
        results.append({"traj_uid": case["traj_uid"], "kind": case["kind"], "verifier_result": result, "semantic_contradiction_confirmed": True, "evidence_level": case["evidence_level"]})
    return {"offline": True, "episodes": len(results), "cases": results}
