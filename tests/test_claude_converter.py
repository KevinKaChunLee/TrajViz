"""Native Claude transcript conversion without filesystem discovery."""

import copy
import unittest

from agent_trajviz.insight.formats.claude_code import convert_claude_code_events
from agent_trajviz.insight.metrics import effective_agent
from agent_trajviz.insight.parser import parse_steps


def event(role, content, *, second=None, mid=None, uuid=None, agent=None, **message):
    record = {
        "type": role, "sessionId": "native-session", "uuid": uuid or f"{role}-{mid or second}",
        "parentUuid": None, "cwd": "/project", "message": {
            "role": role, "content": content, **message,
        },
    }
    if second is not None:
        record["timestamp"] = f"2026-01-01T00:{second // 60:02d}:{second % 60:02d}.000Z"
    if mid:
        record["message"]["id"] = mid
    if agent:
        record.update(agentId=agent, isSidechain=True)
    return record


def tool(tool_id="call", name="Bash"):
    return {"type": "tool_use", "id": tool_id, "name": name, "input": {"command": "pwd"}}


def result(tool_id="call", text="/project", error=False):
    return {"type": "tool_result", "tool_use_id": tool_id, "content": text, "is_error": error}


def progress(agent, record=None, *, parent="", tool_id="spawn"):
    data = {"type": "agent_progress", "agentId": agent}
    if record is not None:
        data["message"] = record
    envelope = {"type": "progress", "sessionId": "native-session", "parentToolUseID": tool_id,
                "data": data, "timestamp": "2026-01-01T00:00:04.000Z"}
    if parent:
        envelope["agentId"] = parent
    return envelope


class ClaudeConverterTests(unittest.TestCase):
    def convert(self, records, **kwargs):
        raw = convert_claude_code_events(records, **kwargs)
        self.assertNotIn("_error", raw)
        return raw, parse_steps(raw)

    def test_grouped_response_uses_final_usage_once_and_keeps_thinking(self):
        records = [event("user", "Investigate", second=0)]
        for i, block in enumerate([
            {"type": "thinking", "thinking": "Read the implementation.", "signature": "sig"},
            {"type": "text", "text": "I will read it."}, tool(),
        ]):
            records.append(event("assistant", [block], second=i + 1, mid="response", uuid=f"block-{i}",
                                 usage={"input_tokens": 2, "output_tokens": i + 1,
                                        "cache_read_input_tokens": 10, "cache_creation_input_tokens": 4}))
        records.append(copy.deepcopy(records[-1]))
        records.append(event("user", [result()], second=5))
        raw, steps = self.convert(records)
        self.assertEqual(len(steps), 2)
        response = steps[-1]
        self.assertEqual([p["type"] for p in response["parts"]], ["reasoning", "text", "tool_call"])
        self.assertEqual(response["parts"][0]["text"], "Read the implementation.")
        self.assertEqual(response["tokens"], {"total": 19, "input": 2, "output": 3,
                                             "cache_read": 10, "cache_write": 4})
        self.assertNotIn("reasoning", response["tokens"])
        self.assertEqual(raw["token_usage"]["total_tokens"], 19)
        self.assertEqual(response["tool_calls"][0]["output"], "/project")

    def test_repeated_uuid_can_update_usage_and_content_without_duplicating(self):
        first = event("assistant", [{"type": "text", "text": "Partial"}], second=1,
                      mid="response", uuid="same", usage={"output_tokens": 2})
        final = event("assistant", [{"type": "text", "text": "Final"}], second=2,
                      mid="response", uuid="same", usage={"output_tokens": 3}, stop_reason="end_turn")
        _, steps = self.convert([first, final, copy.deepcopy(final)])
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["text_preview"], "Final")
        self.assertEqual(steps[0]["tokens"]["output"], 3)
        self.assertEqual(steps[0]["finish"], "end_turn")

    def test_identical_text_blocks_from_different_entries_are_preserved(self):
        records = [event("assistant", [{"type": "text", "text": "same"}], second=i,
                         mid="response", uuid=f"block-{i}") for i in (1, 2)]
        _, steps = self.convert(records)
        self.assertEqual([p["text"] for p in steps[0]["parts"]], ["same", "same"])

    def test_cumulative_message_snapshot_does_not_repeat_earlier_blocks(self):
        thinking = {"type": "thinking", "thinking": "Plan"}
        text = {"type": "text", "text": "Done"}
        _, steps = self.convert([
            event("assistant", [thinking], second=1, mid="response", uuid="first"),
            event("assistant", [thinking, text], second=2, mid="response", uuid="snapshot"),
        ])
        self.assertEqual([p["type"] for p in steps[0]["parts"]], ["reasoning", "text"])

    def test_tool_results_and_repeated_response_ids_are_scoped_to_agent(self):
        records = [event("assistant", [tool()], second=1, mid="same"),
                   event("user", [result(text="main")], second=3)]
        children = {"child": [event("assistant", [tool()], second=2, mid="same", agent="child"),
                              event("user", [result(text="child error", error=True)], second=4, agent="child")]}
        _, steps = self.convert(records, child_event_lists=children)
        self.assertEqual(len(steps), 2)
        main, child = steps
        self.assertEqual(main["tool_calls"][0]["output"], "main")
        self.assertEqual(child["tool_calls"][0]["output"], "child error")
        self.assertEqual(child["error_count"], 1)
        self.assertEqual(child["tool_calls"][0]["error"], "child error")
        self.assertEqual(effective_agent(main), "")
        self.assertEqual(effective_agent(child), "child")

    def test_inline_child_fallback_and_spawn_correlation(self):
        child = event("assistant", [{"type": "text", "text": "Child answer"}],
                      second=3, mid="child-msg", agent="child", usage={"output_tokens": 6})
        raw, steps = self.convert([
            event("assistant", [tool("spawn", "Agent")], second=1),
            progress("child", child), progress("child", copy.deepcopy(child)),
        ])
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[-1]["agent"], "child")
        self.assertEqual(raw["token_usage"]["total_tokens"], 6)
        self.assertEqual(raw["_claude_code_sub_agents"][0]["spawned_by"], "spawn")
        self.assertEqual(raw["_claude_code_sub_agents"][0]["source"], "inline")
        self.assertTrue(any("inline progress" in warning for warning in raw["_load_warnings"]))

    def test_child_file_is_authoritative_over_inline_progress(self):
        raw, steps = self.convert([
            event("user", "Main", second=0),
            progress("child", event("assistant", "Snapshot", second=2, mid="response", agent="child",
                                    usage={"output_tokens": 500})),
        ], child_event_lists={"child": {"events": [
            event("assistant", "Complete", second=3, mid="response", agent="child", usage={"output_tokens": 7}),
        ], "source_path": "/session/subagents/agent-child.jsonl"}})
        self.assertEqual(steps[-1]["text_preview"], "Complete")
        self.assertEqual(raw["token_usage"]["total_tokens"], 7)
        self.assertEqual(raw["_claude_code_sub_agents"][0]["source"], "file")
        self.assertFalse(raw["_load_warnings"])

    def test_inline_api_message_replay_without_transcript_uuid_is_deduplicated(self):
        message = {"role": "assistant", "id": "child-response",
                   "content": [{"type": "text", "text": "Child answer"}],
                   "usage": {"output_tokens": 4}}
        raw, steps = self.convert([event("user", "Main", second=0),
                                   progress("child", message), progress("child", copy.deepcopy(message))])
        child = next(step for step in steps if step["agent"] == "child")
        self.assertEqual(child["parts"], [{"type": "text", "text": "Child answer"}])
        self.assertEqual(raw["token_usage"]["total_tokens"], 4)

    def test_nested_progress_preserves_hierarchy(self):
        grandchild = event("assistant", "Nested", second=3, mid="nested", agent="grandchild")
        raw, steps = self.convert([
            event("user", "Main", second=0),
            progress("child", event("assistant", "Child", second=2, agent="child")),
        ], child_event_lists={"child": {"events": [
            event("assistant", "Child complete", second=2, agent="child"),
            progress("grandchild", grandchild, parent="child", tool_id="nested-spawn"),
        ]}})
        nested = next(step for step in steps if step["agent"] == "grandchild")
        self.assertEqual(nested["parent_agent_id"], "child")
        self.assertEqual(nested["session_depth"], 2)
        summary = next(info for info in raw["_claude_code_sub_agents"] if info["agent_id"] == "grandchild")
        self.assertEqual(summary["spawned_by"], "nested-spawn")

    def test_missing_referenced_child_is_reported(self):
        response = event("user", [result("spawn")], second=2)
        response["toolUseResult"] = {"agentId": "missing"}
        raw, _ = self.convert([event("assistant", [tool("spawn", "Task")], second=1), response,
                               progress("also-missing")])
        self.assertTrue(any("missing" in warning and "unavailable" in warning for warning in raw["_load_warnings"]))
        self.assertTrue(any("also-missing" in warning for warning in raw["_load_warnings"]))

    def test_string_user_content_and_mixed_tool_results_preserve_prompt(self):
        _, steps = self.convert([
            event("user", "Investigate this", second=0),
            event("assistant", [tool()], second=1),
            event("user", [result(), {"type": "text", "text": "Please continue"}], second=2),
        ])
        self.assertEqual([s["text_preview"] for s in steps], ["Investigate this", "[Tool: Bash]", "Please continue"])
        self.assertEqual(steps[1]["tool_calls"][0]["status"], "success")

    def test_compaction_is_an_explicit_checkpoint(self):
        raw, steps = self.convert([
            {"type": "file-history-snapshot", "snapshot": {}},
            {"type": "queue-operation", "sessionId": "native-session", "operation": "dequeue"},
            event("user", "Explain", second=0),
            {"type": "system", "subtype": "compact_boundary", "sessionId": "native-session",
             "uuid": "compact", "timestamp": "2026-01-01T00:00:02.000Z",
             "compactMetadata": {"trigger": "auto", "preTokens": 100000}},
        ])
        self.assertEqual(len(steps), 2)
        compact = steps[-1]
        self.assertEqual(compact["role"], "compaction")
        self.assertTrue(compact["is_compaction_checkpoint"])
        self.assertEqual(compact["message_type"], "compaction")
        self.assertEqual(compact["compaction_reason"], "auto")
        self.assertEqual(raw["stats"]["assistant_messages"], 0)

    def test_timestamps_bound_observed_latency_without_human_idle(self):
        _, steps = self.convert([
            event("user", "First", second=0),
            event("assistant", [tool()], second=1),
            event("user", [result()], second=3),
            event("assistant", "Done", second=4),
            event("user", "Next question after a minute", second=64),
            event("assistant", "Answer", second=65),
        ])
        call = steps[1]["tool_calls"][0]
        self.assertEqual(call["duration_ms"], 2000)
        self.assertEqual(call["metadata"]["timing_source"], "transcript")
        self.assertEqual(steps[1]["duration"], 2)
        self.assertIsNone(steps[2]["duration"])
        self.assertIsNone(steps[-1]["duration"])

    def test_reported_tool_duration_takes_precedence_over_transcript_latency(self):
        output = event("user", [result()], second=3)
        output["toolUseResult"] = {"totalDurationMs": 1200}
        _, steps = self.convert([event("assistant", [tool()], second=1), output])
        call = steps[0]["tool_calls"][0]
        self.assertEqual(call["duration_ms"], 1200)
        self.assertEqual(call["metadata"]["timing_source"], "tool_result")
        self.assertEqual(call["time_end"] - call["time_start"], 2000)

    def test_child_overlap_does_not_backfill_final_parent_duration(self):
        _, steps = self.convert([event("assistant", "Parent finished", second=1)],
                                child_event_lists={"child": [event("assistant", "Child finished", second=10, agent="child")]})
        self.assertTrue(all(step["duration"] is None for step in steps))

    def test_turn_duration_is_preserved_without_assigning_it_to_final_message(self):
        raw, steps = self.convert([
            event("assistant", "Done", second=2),
            {"type": "system", "subtype": "turn_duration", "sessionId": "native-session",
             "durationMs": 1800, "timestamp": "2026-01-01T00:00:03.000Z"},
        ])
        self.assertEqual(raw["_claude_code_turn_durations"][0]["duration_ms"], 1800)
        self.assertIsNone(steps[-1]["duration"])

    def test_standalone_child_identity_from_record_or_filename(self):
        for record, path in [
            (event("assistant", "Child", second=1, agent="native-child"), "/project/session.jsonl"),
            (event("assistant", "Child", second=1), "/project/agent-native-child.jsonl"),
        ]:
            with self.subTest(path=path):
                _, steps = self.convert([record], source_path=path)
                self.assertTrue(steps[0]["is_sub_agent"])
                self.assertEqual(steps[0]["agent"], "native-child")
                self.assertEqual(steps[0]["session_id"], "native-session")

    def test_native_metadata_only_session_is_valid_and_generic_messages_are_rejected(self):
        _, steps = self.convert([{"type": "system", "subtype": "init", "sessionId": "native-session"}])
        self.assertEqual(steps, [])
        invalid = convert_claude_code_events([{"type": "assistant", "message": {"role": "assistant", "content": "Generic"}}])
        self.assertIn("_error", invalid)

    def test_schema_is_consistent_for_user_and_assistant(self):
        _, steps = self.convert([event("user", "Question", second=0), event("assistant", "Answer", second=1)])
        for step in steps:
            self.assertFalse(step["is_sub_agent"])
            self.assertEqual(step["agent"], "")
            self.assertEqual(step["session_depth"], 0)
            for field in ("tokens", "parts", "tool_calls", "duration", "message_id", "parent_id",
                          "session_id", "parent_session_id", "session_title", "compaction_reason"):
                self.assertIn(field, step)

    def test_source_positions_survive_metadata_filtering_and_chronological_sort(self):
        records = [
            {"type": "queue-operation", "sessionId": "native-session"},
            event("assistant", "Later response", second=10),
            {"type": "file-history-snapshot", "snapshot": {}},
            event("user", "Earlier prompt", second=0),
            progress("inline", event("assistant", "Inline response", second=3, agent="inline")),
        ]
        children = {"file": {"source_path": "/project/subagents/agent-file.jsonl", "events": [
            {"type": "file-history-snapshot", "snapshot": {}},
            event("assistant", "File response", second=4, agent="file"),
        ]}}
        original_records, original_children = copy.deepcopy(records), copy.deepcopy(children)
        _, steps = self.convert(records, source_path="/project/session.jsonl", child_event_lists=children)
        self.assertEqual([s["text_preview"] for s in steps],
                         ["Earlier prompt", "Inline response", "File response", "Later response"])
        self.assertEqual([s["raw_index"] for s in steps], [3, 4, 1, 1])
        self.assertEqual([s["source_path"] for s in steps], [
            "/project/session.jsonl", "/project/session.jsonl",
            "/project/subagents/agent-file.jsonl", "/project/session.jsonl",
        ])
        self.assertEqual(records, original_records)
        self.assertEqual(children, original_children)


if __name__ == "__main__":
    unittest.main()
