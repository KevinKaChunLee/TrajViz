"""Native Claude session ingestion through the public API and dashboard path."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import agent_trajviz as tv
from agent_trajviz.insight.session import LoadedSession, load_session


def record(role, content, second=0, *, agent="", mid="", sid="session-1", uuid=""):
    message = {"role": role, "content": content}
    if role == "assistant":
        message.update(id=mid or f"msg-{agent}-{second}", model="claude-test", stop_reason="end_turn",
                       usage={"input_tokens": 10, "output_tokens": 5,
                              "cache_read_input_tokens": 20, "cache_creation_input_tokens": 2})
    return {"type": role, "sessionId": sid, "uuid": uuid or f"uuid-{agent}-{role}-{second}",
            "parentUuid": None, "timestamp": f"2026-01-01T00:00:{second:02d}Z",
            "cwd": "/workspace", **({"agentId": agent, "isSidechain": True} if agent else {}),
            "message": message}


def write_log(path: Path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in events), encoding="utf-8")
    return path


def make_tree(directory):
    parent = write_log(directory / "session-1.jsonl", [
        {"type": "queue-operation", "operation": "dequeue", "sessionId": "session-1"},
        record("user", "Investigate the parser", 0),
        record("assistant", [{"type": "tool_use", "id": "spawn-a", "name": "Agent",
                              "input": {"prompt": "Inspect parsing", "subagent_type": "Explore"}}], 1),
        {**record("user", [{"type": "tool_result", "tool_use_id": "spawn-a", "content": "Found the parser"}], 5),
         "toolUseResult": {"agentId": "a", "description": "Inspect parsing"}},
        record("assistant", [{"type": "text", "text": "Done"}], 9),
    ])
    child = write_log(directory / "session-1" / "subagents" / "agent-a.jsonl", [
        record("user", "Inspect parsing", 2, agent="a"),
        record("assistant", [{"type": "tool_use", "id": "read-a", "name": "Read",
                              "input": {"file_path": "/workspace/parser.py"}}], 3, agent="a"),
        record("user", [{"type": "tool_result", "tool_use_id": "read-a", "content": "def parse(): pass"}],
               4, agent="a"),
        record("assistant", [{"type": "text", "text": "Found it"}], 5, agent="a"),
    ])
    return parent, child


def test_load_discovers_children_and_preserves_calls_and_usage(tmp_path):
    parent, _ = make_tree(tmp_path)
    run = tv.load(parent)
    assert run.format == "claude_code" and run.format_label == "Claude Code"
    assert len(run.steps) == 6  # Tool-result-only records attach to their calls.
    assert {s["agent"] for s in run.steps} == {"", "a"}
    assert run.metrics["tokens"]["total"] == 4 * 37
    assert run.metrics["tool_call_count"] == 2
    child_call = next(tc for s in run.steps for tc in s["tool_calls"] if tc["tool_id"] == "read-a")
    assert child_call["output"] == "def parse(): pass" and child_call["status"] == "success"
    assert child_call["duration_ms"] == 1000
    assert any(a["spawned_by_step"] is not None for a in run.agents if a["agent_id"])
    assert len(run.views()) == len(tv.VIEW_CATALOG)
    session = load_session(str(parent))
    assert isinstance(session, LoadedSession) and session.format == "claude_code"


def test_current_bookkeeping_records_are_not_replayed_as_conversation_or_usage():
    # Sanitized shapes from Claude Code 2.1.310: request blobs contain earlier
    # messages and cost-state contains aggregate counters, neither is a turn.
    path = Path(__file__).parent / "fixtures/claude_current/session-current.jsonl"
    run = tv.load(path)
    assert len(run.steps) == 6 and len(run.agents) == 2
    assert run.metrics["tool_call_count"] == 2
    assert run.metrics["tokens"]["total"] == 323
    assert [sum(step["tokens"][key] for step in run.steps) for key in
            ("input", "output", "cache_read", "cache_write")] == [10, 34, 260, 19]
    child_call = next(call for step in run.steps for call in step["tool_calls"]
                      if call["tool_id"] == "reviewer-read")
    assert child_call["duration_ms"] == 17
    assert child_call["metadata"]["timing_source"] == "tool_result"
    child = next(agent for agent in run.agents if agent["agent_id"] == "reviewer-current")
    assert child["spawned_by_step"] is not None
    assert run.raw["timing"]["total_duration"] == 8.0
    assert run.raw["_claude_code_control_events"][0]["kind"] == "request_interrupted"
    assert any("request interruption" in warning for warning in run.warnings)
    restored = tv.Trajectory.from_dict(run.to_dict(include_raw=True))
    assert {key: view.to_dict() for key, view in run.views().items()} == {
        key: view.to_dict() for key, view in restored.views().items()}


def test_interruption_marker_inside_a_tool_result_keeps_its_outcome(tmp_path):
    parent = write_log(tmp_path / "session-1.jsonl", [
        record("assistant", [{"type": "tool_use", "id": "call", "name": "Bash",
                              "input": {"command": "sleep 60"}}], 1),
        record("user", [{"type": "tool_result", "tool_use_id": "call", "is_error": True,
                         "content": "[Request interrupted by user]"}], 2),
    ])
    run = tv.load(parent)
    call = run.steps[0]["tool_calls"][0]
    assert call["status"] == "error" and call["output"] == "[Request interrupted by user]"
    assert run.raw["timing"]["total_duration"] == 1.0


def test_nested_subagents_remain_distinct(tmp_path):
    parent, _ = make_tree(tmp_path)
    write_log(tmp_path / "session-1/subagents/agent-a/subagents/agent-b.jsonl", [
        record("assistant", "Nested result", 4, agent="b"),
    ])
    run = tv.load(parent)
    nested = next(s for s in run.steps if s["agent"] == "b")
    assert nested["is_sub_agent"] and nested["parent_agent_id"] == "a"
    assert nested["session_depth"] == 2
    assert {a["agent_id"] for a in run.agents if a["agent_id"]} == {"a", "b"}
    diagnostics = run.view("diagnostics").data
    assert diagnostics["subagent_session_count"] == 2
    assert diagnostics["subagent_total_tokens"] == 3 * 37


def test_legacy_siblings_are_matched_to_loaded_session(tmp_path):
    parent = write_log(tmp_path / "session-1.jsonl", [record("user", "Hi")])
    write_log(tmp_path / "agent-own.jsonl", [record("assistant", "Mine", 1, agent="own")])
    write_log(tmp_path / "agent-foreign.jsonl", [record("assistant", "Private", 1, agent="foreign", sid="other")])
    run = tv.load(parent)
    assert {s["agent"] for s in run.steps} == {"", "own"}
    assert all(s["text_preview"] != "Private" for s in run.steps)


def test_loading_child_does_not_merge_siblings_or_parent(tmp_path):
    parent, child = make_tree(tmp_path)
    write_log(child.parent / "agent-other.jsonl", [record("assistant", "Other", 6, agent="other")])
    run = tv.load(child)
    assert {s["agent"] for s in run.steps} == {"a"}
    assert len(run.steps) == 3
    assert parent.read_text(encoding="utf-8")  # Parent remains untouched.


def test_directories_resolve_unambiguously(tmp_path):
    parent, _ = make_tree(tmp_path)
    assert tv.load(tmp_path).steps == tv.load(parent).steps
    assert tv.load(tmp_path / "session-1").steps == tv.load(parent).steps
    write_log(tmp_path / "another.jsonl", [record("user", "Other", sid="other")])
    with pytest.raises(tv.LoadError, match="multiple sessions"):
        tv.load(tmp_path)


def test_malformed_child_warns_without_losing_parent(tmp_path):
    parent, child = make_tree(tmp_path)
    child.write_text('{"type":broken}\n', encoding="utf-8")
    run = tv.load(parent)
    assert run.steps and {s["agent"] for s in run.steps} == {""}
    assert any("agent-a.jsonl" in warning for warning in run.warnings)


def test_child_symlinks_are_not_followed(tmp_path):
    parent, child = make_tree(tmp_path)
    other = tmp_path / "unrelated" / "agent-hidden.jsonl"
    write_log(other, [record("assistant", "Do not merge", 7, agent="hidden")])
    link = child.parent / "agent-hidden.jsonl"
    try:
        link.symlink_to(other)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert "hidden" not in {s["agent"] for s in tv.load(parent).steps}


def test_source_identity_covers_child_bytes_and_survives_relocation(tmp_path):
    parent, child = make_tree(tmp_path / "first")
    initial = tv.load(parent)
    assert initial.sha256 != hashlib.sha256(parent.read_bytes()).hexdigest()
    shutil.copytree(parent.parent, tmp_path / "moved")
    assert tv.load(tmp_path / "moved/session-1.jsonl").sha256 == initial.sha256
    child.write_text(child.read_text(encoding="utf-8").replace("Found it", "Different"), encoding="utf-8")
    assert tv.load(parent).sha256 != initial.sha256


def test_source_identity_covers_inferred_child_hierarchy(tmp_path):
    parent, child = make_tree(tmp_path)
    nested = child.parent / "agent-b.jsonl"
    write_log(nested, [record("assistant", "Nested result", 4, agent="b")])
    before = tv.load(parent)
    target = child.parent / "agent-a/subagents/agent-b.jsonl"
    target.parent.mkdir(parents=True)
    nested.rename(target)
    after = tv.load(parent)
    assert before.sha256 != after.sha256
    assert next(s for s in before.steps if s["agent"] == "b")["parent_agent_id"] == ""
    assert next(s for s in after.steps if s["agent"] == "b")["parent_agent_id"] == "a"


def test_context_windows_are_separate_for_children_sharing_session_id(tmp_path):
    parent, _ = make_tree(tmp_path)
    run = tv.load(parent)
    keys = {a["key"] for a in run.view("context_pressure").data["agents"]}
    assert "a" in keys
    assert len(keys) >= 2


def test_native_compaction_uses_reported_pre_compaction_occupancy(tmp_path):
    parent = write_log(tmp_path / "session-1.jsonl", [
        record("assistant", "Before", 1),
        {"type": "system", "subtype": "compact_boundary", "sessionId": "session-1",
         "uuid": "compact", "timestamp": "2026-01-01T00:00:02Z",
         "compactMetadata": {"trigger": "auto", "preTokens": 100000}},
        record("assistant", "After", 3),
    ])
    events = tv.load(parent).context_pressure()["series"]["events"]
    compact = next(event for event in events if event["kind"] == "compaction_message")
    assert compact["occupancy_before"] == 100000
    assert compact["occupancy_after"] == 32


@pytest.mark.parametrize("bad", ['{"type": [], "sessionId": "session-1"}\n',
                                 '{"type":"user","sessionId":"session-1","message":' + '[' * 2000 + ']}\n'])
def test_wrong_typed_or_deep_child_keeps_parent(tmp_path, bad):
    parent, child = make_tree(tmp_path)
    child.write_text(bad, encoding="utf-8")
    run = tv.load(parent)
    assert run.steps and run.warnings
    assert {s["agent"] for s in run.steps} == {""}


def zip_tree(directory, target, prefix="bundle/"):
    with zipfile.ZipFile(target, "w") as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, prefix + path.relative_to(directory).as_posix())
    return target


@pytest.mark.parametrize("prefix", ["", "bundle/"])
def test_zip_merges_children_without_host_filesystem_access(tmp_path, prefix):
    parent, _ = make_tree(tmp_path / "tree")
    archive = zip_tree(parent.parent, tmp_path / "session.zip", prefix)
    loaded = tv.load(archive)
    def semantic_steps(run):
        return [{k: v for k, v in step.items() if k != "source_path"} for step in run.steps]
    assert semantic_steps(loaded) == semantic_steps(tv.load(parent))
    assert loaded.sha256 == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert isinstance(load_session(str(archive)), LoadedSession)
    with pytest.raises(tv.LoadError) as caught:
        tv.load(archive, format="codex")
    assert caught.value.code == "mismatch" and caught.value.detected == "claude_code"


def test_zip_refuses_ambiguous_sessions(tmp_path):
    parent, _ = make_tree(tmp_path / "tree")
    write_log(parent.parent / "other.jsonl", [record("user", "Other", sid="other")])
    archive = zip_tree(parent.parent, tmp_path / "session.zip")
    with pytest.raises(tv.LoadError, match="multiple parent sessions"):
        tv.load(archive)


def test_zip_ignores_unrelated_non_session_jsonl(tmp_path):
    parent, _ = make_tree(tmp_path / "tree")
    (parent.parent / "debug.jsonl").write_text('{"debug":"metadata"}\n', encoding="utf-8")
    archive = zip_tree(parent.parent, tmp_path / "session.zip")
    assert {s["agent"] for s in tv.load(archive).steps} == {"", "a"}


def test_zip_ignores_traversal_and_symlink_children(tmp_path):
    parent, _ = make_tree(tmp_path / "tree")
    archive = zip_tree(parent.parent, tmp_path / "session.zip")
    extra = json.dumps(record("assistant", "Hidden", 6, agent="hidden")) + "\n"
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("bundle/session-1/subagents/../../agent-hidden.jsonl", extra)
        info = zipfile.ZipInfo("bundle/session-1/subagents/agent-link.jsonl")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, extra)
    assert {s["agent"] for s in tv.load(archive).steps} == {"", "a"}


def test_zip_child_failure_keeps_parent(tmp_path):
    parent, child = make_tree(tmp_path / "tree")
    child.write_text("bad json\n", encoding="utf-8")
    archive = zip_tree(parent.parent, tmp_path / "session.zip")
    run = tv.load(archive)
    assert run.steps and run.warnings
    assert {s["agent"] for s in run.steps} == {""}


def test_dsh_zip_can_still_contain_unrelated_debug_jsonl(tmp_path):
    from agent_trajviz.insight.loaders import detect_format, load_trajectory

    archive = tmp_path / "dsh.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("session.jsonl", json.dumps({"type": "session", "id": "dsh-root", "createdAt": 1000}) + "\n")
        bundle.writestr("debug.jsonl", '{"debug":"metadata"}\n')
    raw = load_trajectory(str(archive))
    assert "_error" not in raw and detect_format(raw) == "dsh"


def test_zip_identity_and_conversion_use_same_read_bytes(tmp_path, monkeypatch):
    from agent_trajviz.insight import loaders

    parent, _ = make_tree(tmp_path / "tree")
    archive = zip_tree(parent.parent, tmp_path / "session.zip")
    original_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    parse = loaders.parse_claude_zip

    def replace_after_read(path, **kwargs):
        Path(path).write_bytes(b"changed after the loader read the file")
        return parse(path, **kwargs)

    monkeypatch.setattr(loaders, "parse_claude_zip", replace_after_read)
    run = tv.load(archive)
    assert run.sha256 == original_sha and {s["agent"] for s in run.steps} == {"", "a"}


def test_native_round_trip_preserves_all_views_and_child_data(tmp_path):
    parent, _ = make_tree(tmp_path / "tree")
    run = tv.load(parent)
    tv.export_json(run, tmp_path / "out")
    again = tv.load(tmp_path / "out/trajectory.json")
    assert again.steps == run.steps
    assert {key: view.to_dict() for key, view in again.views().items()} == {
        key: view.to_dict() for key, view in run.views().items()}


def test_native_cli_exports_views(tmp_path):
    parent, _ = make_tree(tmp_path / "tree")
    out = tmp_path / "views"
    result = subprocess.run([sys.executable, "-m", "agent_trajviz", "--json", str(parent), "-o", str(out)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert len(list(out.glob("*.json"))) == len(tv.VIEW_CATALOG) + 2


def test_legacy_export_and_format_id_are_removed(tmp_path):
    legacy = tmp_path / "old.json"
    legacy.write_text(json.dumps({"format": "ccsession-trajectory", "trajectory": [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]}]}), encoding="utf-8")
    with pytest.raises(tv.LoadError) as caught:
        tv.load(legacy)
    assert caught.value.code == "unknown_format"
    assert "ccsession" not in tv.FORMATS
    with pytest.raises(ValueError, match="unknown format"):
        tv.load(legacy, format="ccsession")
