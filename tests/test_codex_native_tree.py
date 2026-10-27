"""Current Codex model identity, inherited history and explicit child trees."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path

import pytest

import agent_trajviz as tv
from agent_trajviz.collect.codex import collect_codex_session, main
from agent_trajviz.insight.formats.codex import _convert_codex_to_internal
from agent_trajviz.insight.formats.codex_files import discover_codex_children, own_codex_events, session_meta
from agent_trajviz.insight.metrics import effective_agent

FIXTURE = Path(__file__).parent / "fixtures" / "codex_current"


def _events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _copy_tree(tmp_path):
    path = tmp_path / "capture"
    shutil.copytree(FIXTURE, path)
    return path


def _store(tmp_path):
    home = tmp_path / "codex-home"
    folder = home / "sessions" / "2026" / "10" / "26"
    folder.mkdir(parents=True)
    root = folder / "rollout-root.jsonl"
    child = folder / "rollout-child.jsonl"
    root.write_bytes((FIXTURE / "root.jsonl").read_bytes())
    child.write_bytes((FIXTURE / "root/subagents/agent-child.jsonl").read_bytes())
    database = home / "state_5.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript("""
            CREATE TABLE threads(id TEXT PRIMARY KEY, rollout_path TEXT);
            CREATE TABLE thread_spawn_edges(parent_thread_id TEXT, child_thread_id TEXT, status TEXT);
        """)
        connection.executemany("INSERT INTO threads VALUES (?,?)", [("root", str(root)), ("child", str(child))])
        connection.execute("INSERT INTO thread_spawn_edges VALUES ('root','child','open')")
    return home, database, root, child


def test_current_native_tree_counts_only_own_child_work_and_model():
    run = tv.load(FIXTURE / "root.jsonl")
    assert run.format == "codex"
    assert len(run.steps) == 4 and run.metrics["tool_call_count"] == 2
    assert run.metrics["tokens"]["total"] == 22
    assert run.raw["metadata"]["model"] == "gpt-6.1-sol"
    usage = {}
    for step in run.steps:
        key = effective_agent(step)
        usage[key] = usage.get(key, 0) + step["tokens"]["total"]
    assert usage == {"": 13, "child": 9}
    child_steps = [step for step in run.steps if step["session_id"] == "child"]
    assert all(step["parent_session_id"] == "root" and step["session_depth"] == 1 for step in child_steps)
    assert all(step["model_id"] == "review-model" for step in child_steps)
    assert {step["raw_index"] for step in child_steps} == {8, 9}
    assert all(step["source_path"].endswith("agent-child.jsonl") for step in child_steps)
    calls = [call for step in run.steps for call in step["tool_calls"]]
    assert {call["tool_id"] for call in calls} == {"spawn-1", "child-test"}
    assert next(call for call in calls if call["tool_id"] == "spawn-1")["metadata"]["sessionId"] == "child"
    assert run.raw["_codex_sub_agents"][0]["spawned_by"] == "spawn-1"
    assert not run.warnings


def test_inherited_parent_header_cannot_replace_child_and_input_is_unchanged():
    events = _events(FIXTURE / "root/subagents/agent-child.jsonl")
    original = copy.deepcopy(events)
    converted = _convert_codex_to_internal(events)
    assert converted["metadata"]["session_id"] == "child"
    assert converted["metadata"]["model"] == "review-model"
    assert converted["token_usage"]["total_tokens"] == 9
    assert len(converted["messages"]) == 2
    assert events == original


def test_plain_parent_warns_about_missing_child_without_scanning_state(tmp_path, monkeypatch):
    source = tmp_path / "standalone.jsonl"
    source.write_bytes((FIXTURE / "root.jsonl").read_bytes())

    def private_state(*_args, **_kwargs):
        pytest.fail("file loading must not open the global Codex database")

    monkeypatch.setattr(sqlite3, "connect", private_state)
    run = tv.load(source)
    assert run.metrics["tokens"]["total"] == 13
    assert any("child rollouts are missing" in warning for warning in run.warnings)


def test_wrong_parent_and_symlink_children_are_skipped(tmp_path):
    folder = _copy_tree(tmp_path)
    child = folder / "root/subagents/agent-child.jsonl"
    events = _events(child)
    events[0]["payload"]["parent_thread_id"] = "unrelated"
    events[0]["payload"]["source"]["subagent"]["thread_spawn"]["parent_thread_id"] = "unrelated"
    child.write_text("\n".join(json.dumps(event) for event in events))
    run = tv.load(folder / "root.jsonl")
    assert run.metrics["tokens"]["total"] == 13
    assert any("no verified parent" in warning for warning in run.warnings)
    child.unlink()
    child.symlink_to(FIXTURE / "root/subagents/agent-child.jsonl")
    run = tv.load(folder / "root.jsonl")
    assert run.metrics["tokens"]["total"] == 13
    assert any("symlinked" in warning for warning in run.warnings)


def test_conflicting_direct_and_nested_parent_does_not_link_child(tmp_path):
    folder = _copy_tree(tmp_path)
    child = folder / "root/subagents/agent-child.jsonl"
    events = _events(child)
    events[0]["payload"]["parent_thread_id"] = "other"
    child.write_text("\n".join(json.dumps(event) for event in events))
    assert tv.load(folder / "root.jsonl").metrics["tokens"]["total"] == 13


def test_child_hash_covers_exact_bytes_and_relocation_is_stable(tmp_path):
    folder = _copy_tree(tmp_path)
    run = tv.load(folder / "root.jsonl")
    relocated = tmp_path / "relocated"
    shutil.copytree(folder, relocated)
    assert tv.load(relocated / "root.jsonl").sha256 == run.sha256
    child = folder / "root/subagents/agent-child.jsonl"
    child.write_bytes(child.read_bytes() + b"\n")
    changed = tv.load(folder / "root.jsonl")
    assert changed.sha256 != run.sha256
    assert changed.metrics == run.metrics
    assert changed.raw["_source_merged_count"] == 1


def test_invalid_history_boundary_never_adds_copied_parent_usage():
    events = _events(FIXTURE / "root/subagents/agent-child.jsonl")
    events[0]["payload"]["subagent_history_start_ordinal"] = "invalid"
    own, warnings = own_codex_events(events)
    assert own == [] and warnings


def test_child_discovery_limits_invalid_files_too(tmp_path, monkeypatch):
    from agent_trajviz.insight.formats import codex_files

    folder = _copy_tree(tmp_path)
    children = folder / "root/subagents"
    (children / "agent-a.jsonl").write_text("not-json")
    monkeypatch.setattr(codex_files, "MAX_CHILDREN", 1)
    bundle = discover_codex_children(_events(folder / "root.jsonl"), str(folder / "root.jsonl"))
    assert not bundle.children
    assert any("first 1" in warning for warning in bundle.warnings)


def test_readonly_collector_queries_exact_edges_and_exports_portable_json(tmp_path):
    home, database, root, child = _store(tmp_path)
    before = database.read_bytes()
    result, sources = collect_codex_session("root", codex_home=home)
    assert result["token_usage"]["total_tokens"] == 22
    assert result["_codex_collection"]["child_count"] == 1
    assert len(result["_codex_source_files"]) == 2
    assert {path for path in sources if path.suffix == ".jsonl"} == {root, child}
    assert database.read_bytes() == before
    output = tmp_path / "collected.json"
    assert main(["root", "--codex-home", str(home), "--db", str(database), "-o", str(output)]) == 0
    run = tv.load(output)
    assert run.metrics["tokens"]["total"] == 22 and len(run.agents) == 2
    assert run.metrics["tool_call_count"] == 2
    assert run.raw["_codex_source_files"][0]["sha256"] == hashlib.sha256(root.read_bytes()).hexdigest()
    # The consolidated file carries all parsed work; source rollouts are no
    # longer required when the user downloads/uploads it elsewhere.
    root.unlink()
    child.unlink()
    assert tv.load(output).metrics["tokens"]["total"] == 22


def test_collector_never_reads_unrelated_thread_or_mismatched_edge(tmp_path):
    home, database, _, child = _store(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO threads VALUES ('unrelated','/private/unrelated-session.jsonl')")
    result, _ = collect_codex_session("root", codex_home=home)
    assert result["token_usage"]["total_tokens"] == 22
    events = _events(child)
    events[0]["payload"]["parent_thread_id"] = "wrong"
    child.write_text("\n".join(json.dumps(event) for event in events))
    result, _ = collect_codex_session("root", codex_home=home)
    assert result["token_usage"]["total_tokens"] == 13
    assert any("disagrees" in warning for warning in result["_load_warnings"])


@pytest.mark.parametrize("destination", ["root", "database"])
def test_collector_cannot_overwrite_sources(tmp_path, destination):
    home, database, root, _ = _store(tmp_path)
    target = root if destination == "root" else database
    before = target.read_bytes()
    assert main(["root", "--codex-home", str(home), "-o", str(target)]) == 1
    assert target.read_bytes() == before


def test_collector_refuses_paths_outside_selected_store(tmp_path):
    home, database, _, _ = _store(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE threads SET rollout_path='/private/outside.jsonl' WHERE id='root'")
    with pytest.raises(ValueError, match="escapes"):
        collect_codex_session("root", codex_home=home)


def test_collector_checks_remaining_bytes_before_reading_child(tmp_path, monkeypatch):
    from agent_trajviz.collect import codex

    home, _, root, child = _store(tmp_path)
    original = codex.read_codex_events
    read_paths = []

    def observed(path, **kwargs):
        read_paths.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(codex, "read_codex_events", observed)
    monkeypatch.setattr(codex, "MAX_TOTAL_BYTES", root.stat().st_size + child.stat().st_size - 1)
    result, _ = collect_codex_session("root", codex_home=home)
    assert read_paths == [root]
    assert result["token_usage"]["total_tokens"] == 13
    assert any("size limit" in warning for warning in result["_load_warnings"])


def test_collector_counts_failed_child_inspections_toward_limit(tmp_path, monkeypatch):
    from agent_trajviz.collect import codex

    home, database, _, _ = _store(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.executemany("INSERT INTO thread_spawn_edges VALUES ('root',?,'open')",
                               [("a-invalid",), ("b-invalid",)])
    monkeypatch.setattr(codex, "MAX_CHILDREN", 2)
    result, _ = collect_codex_session("root", codex_home=home)
    assert result["token_usage"]["total_tokens"] == 13
    assert any("inspection limit" in warning for warning in result["_load_warnings"])


def test_session_header_is_first_even_when_parent_copy_is_later():
    events = _events(FIXTURE / "root/subagents/agent-child.jsonl")
    assert session_meta(events)["id"] == "child"
