"""Every view's data has a JSON Schema, and every view the package produces satisfies it.

The schemas ship in the package (`agent_trajviz/schemas/`) for consumers in
any language; `view_schema()` returns one. The tests validate strictly: a
record that gains a field the schema does not document fails here, so a
change to a view's data is a deliberate edit of its schema. (The published
schemas themselves allow extra fields, so a consumer validating newer data
against an older schema is not broken by an addition.)
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

import agent_trajviz as tv
from agent_trajviz import views as views_mod

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CC = FIXTURES / "reference" / "data" / "trajectory" / "claude_code"
RUNS = [CC / "django__django-11477.json", CC / "astropy__astropy-13033.json",
        FIXTURES / "codearts_minimal.json", FIXTURES / "cursor_minimal.json", FIXTURES / "icode_minimal.json"]
T0 = 1_700_000_000_000

# Runs at the edges: one step, no tokens or timing, failures, sub-agents, a compaction.
EDGE_RUNS = {
    "single": [{"role": "user", "text_preview": "hi"}],
    "chat": [{"role": "user", "text_preview": "explain"}, {"role": "assistant", "text_preview": "done"}],
    "busy": [
        {"role": "user", "text_preview": "Fix it", "time_created_ms": T0, "session_id": "root"},
        {"role": "assistant", "session_id": "root", "time_created_ms": T0 + 1000, "time_completed_ms": T0 + 2500,
         "tokens": {"input": 900, "output": 40, "cache_read": 300, "context_window": 1200},
         "tool_calls": [{"tool_name": "bash", "input": {"command": "pytest"}, "output": "1 failed", "error": "exit 1"}]},
        {"role": "assistant", "session_id": "child", "agent": "explorer", "is_sub_agent": True,
         "time_created_ms": T0 + 3000, "time_completed_ms": T0 + 3400, "tokens": {"input": 200, "output": 10},
         "tool_calls": [{"tool_name": "read", "input": {"file_path": "src/a.py"}, "output": "x"}]},
        {"role": "compaction", "message_type": "compaction", "text_preview": "summary", "time_created_ms": T0 + 5000},
        {"role": "assistant", "session_id": "root", "time_created_ms": T0 + 6000, "time_completed_ms": T0 + 9000,
         "tokens": {"input": 1500, "output": 60, "reasoning": 20},
         "tool_calls": [{"tool_name": "edit", "input": {"file_path": "src/a.py", "old_string": "a", "new_string": "b"},
                         "output": "ok"}]},
    ],
}


def _labels(n: int, unknown: bool) -> dict:
    phases = [("understand", "code_reading"), ("implement", "code_editing"), ("validate", "test_running")]
    steps = [{"index": 0, "role": "user", "phase": "user", "action": "user_prompt", "duration_s": 0}]
    for i in range(1, n):
        phase, action = ("unknown", "unknown") if unknown and i == n - 1 else phases[i % 3]
        steps.append({"index": i, "role": "assistant", "phase": phase, "action": action, "duration_s": 1.5 * i})
    return {"schema_version": "trajectory_labels.v2", "taxonomy_version": "v1", "model": "label-model",
            "trajectory_file": "run.json", "steps": steps}


def _strict(node):
    """The schema with `additionalProperties: false` on every record."""
    if isinstance(node, dict):
        node = {k: _strict(v) for k, v in node.items()}
        if "properties" in node and "additionalProperties" not in node:
            node["additionalProperties"] = False
        return node
    if isinstance(node, list):
        return [_strict(v) for v in node]
    return node


class SchemaTests(unittest.TestCase):
    validators: dict

    @classmethod
    def setUpClass(cls):
        cls.validators = {vid: jsonschema.Draft202012Validator(_strict(schema))
                          for vid, schema in tv.view_schemas().items()}

    def assertValid(self, views: dict, where: str) -> None:
        for vid, view in views.items():
            doc = json.loads(view.to_json())
            errors = [f"{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}"
                      for e in self.validators[vid].iter_errors(doc)]
            self.assertEqual(errors, [], f"{vid} ({where})")

    def test_every_view_id_has_a_well_formed_schema(self):
        traj = tv.load(RUNS[0])
        other = tv.load(RUNS[1])
        produced = {
            **traj.views(),
            **tv.comparison_views(tv.compare(traj, other)),
            **tv.run_group_views(tv.compare_runs([traj, other])),
            **tv.label_views(_labels(6, True)),
            **tv.label_comparison_views(_labels(4, False), _labels(6, True)),
        }
        schemas = tv.view_schemas()
        self.assertEqual(set(schemas), set(produced))
        self.assertEqual(list(schemas)[:len(tv.VIEW_CATALOG)], list(tv.VIEW_CATALOG))
        for vid, schema in schemas.items():
            with self.subTest(view=vid):
                jsonschema.Draft202012Validator.check_schema(schema)
                view = produced[vid]
                self.assertEqual((schema["title"], schema["description"]), (view.title, view.description))
                props = schema["properties"]
                self.assertEqual((props["id"]["const"], props["kind"]["const"]), (vid, view.kind))
                self.assertEqual(props["schema_version"]["const"], tv.VIEWS_SCHEMA_VERSION)

    def test_every_view_of_every_fixture_run_validates(self):
        for path in RUNS:
            self.assertValid(tv.load(path).views(), path.name)

    def test_edge_case_runs_validate_with_every_option(self):
        made = {name: tv.Trajectory.from_steps(steps) for name, steps in EDGE_RUNS.items()}
        for name, run in made.items():
            self.assertValid(run.views(), name)
            for options in ({"window_limit": 1000}, {"window_limit": 1000, "snapshot_step": 1}):
                self.assertValid({"context_pressure": run.view("context_pressure", **options),
                                  "context_usage": run.view("context_usage", **options)}, f"{name} {options}")
        for a in made.values():
            for b in made.values():
                self.assertValid(tv.comparison_views(tv.compare(a, b)), f"{a.steps[0]['text_preview']}")

    def test_comparison_run_group_and_label_views_validate(self):
        ref, cmp_ = tv.load(RUNS[0]), tv.load(RUNS[1])
        patch = FIXTURES / "reference" / "data" / "patch"
        anchor = next(iter(sorted(patch.rglob("*.diff"))), None) or next(iter(sorted(patch.rglob("*.patch"))), None)
        self.assertIsNotNone(anchor)
        self.assertValid(tv.comparison_views(tv.compare(ref, cmp_)), "compare")
        self.assertValid(tv.comparison_views(tv.compare(ref, cmp_, anchor_patch=anchor, fuzzy_commands=True)),
                         "compare with anchor")
        group = tv.compare_runs([ref, cmp_, *RUNS[2:], FIXTURES / "missing.json"])
        self.assertValid(tv.run_group_views(group), "run group with an unreadable file")
        for n, unknown in ((2, False), (6, True)):
            self.assertValid(tv.label_views(_labels(n, unknown)), f"labels {n}")
        self.assertValid(tv.label_comparison_views(_labels(2, False), _labels(6, True)), "label comparison")

    def test_unknown_ids_raise_key_error(self):
        for bad in ("no_such_view", "../schemas/metrics", "index"):
            with self.subTest(view_id=bad), self.assertRaises(KeyError):
                tv.view_schema(bad)

    def test_a_schema_is_a_copy(self):
        tv.view_schema("metrics")["title"] = "changed"
        self.assertNotEqual(tv.view_schema("metrics")["title"], "changed")


class RegisteredViewSchemaTests(unittest.TestCase):
    def setUp(self):
        saved = dict(views_mod._SPECS), dict(views_mod._CATALOG)

        def restore():
            for live, old in zip((views_mod._SPECS, views_mod._CATALOG), saved, strict=True):
                live.clear()
                live.update(old)

        self.addCleanup(restore)

    def test_a_registered_view_has_the_schema_it_was_given(self):
        data_schema = {"type": "object", "required": ["calls"], "properties": {"calls": {"type": "integer"}}}
        tv.register_view("call_count", lambda run: {"calls": sum(len(s["tool_calls"]) for s in run.steps)},
                         title="Tool calls", kind="summary", schema=data_schema)
        schema = tv.view_schema("call_count")
        self.assertEqual(schema["properties"]["data"], data_schema)
        self.assertEqual(schema["properties"]["kind"]["const"], "summary")
        self.assertIn("call_count", tv.view_schemas())
        view = tv.load(RUNS[2]).view("call_count")
        jsonschema.validate(json.loads(view.to_json()), schema)

    def test_without_a_schema_the_data_is_unconstrained(self):
        tv.register_view("anything", lambda run: [1, "two"], title="Anything")
        self.assertEqual(tv.view_schema("anything")["properties"]["data"], {})

    def test_a_schema_must_be_a_mapping(self):
        with self.assertRaises(TypeError):
            tv.register_view("bad_schema", lambda run: 1, title="x", schema="not a schema")  # type: ignore[arg-type]


class SchemaCommandTests(unittest.TestCase):
    def test_the_command_writes_every_schema(self):
        from agent_trajviz.insight import __main__ as cli

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["--schemas", tmp]), 0)
            written = sorted(p.name for p in Path(tmp).iterdir())
        self.assertEqual(out.getvalue().strip(), tmp)
        self.assertEqual(written, sorted(f"{vid}.schema.json" for vid in tv.view_schemas()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
