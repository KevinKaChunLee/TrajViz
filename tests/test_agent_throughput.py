"""Each agent card's rate must be the same measurement as the session's.

The card divided every token the agent touched -- cache reads included -- by
its wall-clock time and labelled the result "Tok/s". On cache-heavy sessions
that is 70-120x the generation rate the Overview reports for the same steps,
so the two surfaces disagreed by two orders of magnitude while appearing to
show one quantity. The card now shows output tokens over generation time,
computed by the code that produces the Overview figure.
"""

from __future__ import annotations

import unittest

from agent_trajviz.insight.assistant import build_analysis_brief
from agent_trajviz.insight.formatting import _build_per_message_md
from agent_trajviz.insight.metrics import (
    build_message_metrics,
    compute_agent_summary,
    compute_metrics,
)
from agent_trajviz.insight.rendering import render_agent_summary_cards


def _step(index, *, agent="", duration=10.0, output=100, cache_read=0, tool_calls=None):
    calls = tool_calls or []
    total = 50 + output + cache_read
    return {
        "index": index,
        "role": "assistant",
        "agent": agent,
        "duration": duration,
        "parts": [],
        "tool_calls": calls,
        "tool_call_count": len(calls),
        "error_count": 0,
        "finish": "stop",
        "time_created_ms": index * 60_000,
        "time_completed_ms": None if duration is None else index * 60_000 + int(duration * 1000),
        "tokens": {"total": total, "input": 50, "output": output, "reasoning": 0,
                   "cache_read": cache_read, "cache_write": 0},
    }


def _bash(start_ms, end_ms):
    return {"tool_name": "bash", "status": "success", "time_start": start_ms, "time_end": end_ms}


def _rows(steps):
    return {row["agent_id"]: row for row in compute_agent_summary(steps, {})}


class AgentRateTests(unittest.TestCase):
    def test_cache_reads_do_not_inflate_an_agents_rate(self):
        # 100 output tokens in 10 s is 10 tok/s however much cached context the
        # step re-read; the old rate here was 100,150 / 10 = 10,015.
        row = _rows([_step(0, cache_read=100_000)])[""]
        self.assertEqual(row["output_tokens_per_sec"], 10.0)

    def test_single_agent_rate_matches_the_overview(self):
        steps = [
            _step(0, duration=12.0, output=300, cache_read=40_000,
                  tool_calls=[_bash(0, 4_000)]),
            _step(1, duration=8.0, output=120, cache_read=55_000),
            _step(2, duration=None, output=900),  # untimed: excluded from both
        ]
        session = compute_metrics(steps, {}, build_message_metrics(steps))
        row = _rows(steps)[""]
        self.assertEqual(row["output_tokens_per_sec"], session["output_tokens_per_sec"])
        self.assertEqual(row["output_throughput_timed_steps"],
                         session["output_throughput_timed_steps"])

    def test_each_agent_is_measured_over_its_own_steps(self):
        steps = [
            _step(0, agent="main", duration=10.0, output=200),
            _step(1, agent="worker", duration=4.0, output=400,
                  tool_calls=[_bash(60_000, 62_000)]),  # 2 s generation window
            _step(2, agent="main", duration=10.0, output=100),
        ]
        rows = _rows(steps)
        self.assertEqual(rows["main"]["output_tokens_per_sec"], 15.0)
        self.assertEqual(rows["worker"]["output_tokens_per_sec"], 200.0)

    def test_an_agent_with_no_timed_steps_has_no_rate(self):
        rows = _rows([_step(0, agent="main"), _step(1, agent="worker", duration=None)])
        self.assertIsNone(rows["worker"]["output_tokens_per_sec"])
        self.assertEqual(rows["worker"]["output_throughput_timed_steps"], 0)

    def test_the_wall_clock_total_rate_is_gone_from_the_row(self):
        self.assertNotIn("tokens_per_second", _rows([_step(0)])[""])


class AgentCardTests(unittest.TestCase):
    def _card(self, steps):
        return render_agent_summary_cards(compute_agent_summary(steps, {}))

    def test_card_names_the_rate_it_shows(self):
        html = self._card([_step(0, cache_read=100_000)])
        self.assertIn("Gen tok/s", html)
        self.assertIn(">10.0<", html)
        self.assertNotIn(">10,015<", html)

    def test_card_says_n_a_rather_than_zero_without_timing(self):
        html = self._card([_step(0, duration=None)])
        self.assertIn(">n/a<", html)

    def test_card_counts_assistant_steps_and_says_so(self):
        self.assertIn("2 assistant steps", self._card([_step(0), _step(1)]))
        self.assertIn("1 assistant step<", self._card([_step(0)]))

    def test_card_tooltip_states_rate_coverage(self):
        html = self._card([_step(0), _step(1, duration=None)])
        self.assertIn("1 of 2 assistant steps", html)


class BriefTests(unittest.TestCase):
    def test_brief_gives_each_agent_its_generation_rate(self):
        steps = [
            {"index": 0, "role": "user", "agent": "", "duration": None, "parts": [],
             "tool_calls": [], "tool_call_count": 0, "finish": "", "time_created_ms": 0,
             "time_completed_ms": None,
             "tokens": {"total": 0, "input": 0, "output": 0, "reasoning": 0,
                        "cache_read": 0, "cache_write": 0}},
            _step(1, agent="main", output=200),
            _step(2, agent="worker", output=50),
        ]
        brief = build_analysis_brief(steps, {})
        self.assertIn("gen 20.0 tok/s", brief)
        self.assertIn("gen 5.0 tok/s", brief)
        # The all-tokens wall-clock rate is still in the brief, under a name
        # that cannot be read as generation speed.
        self.assertNotIn("\ntokens_per_second:", brief)
        self.assertIn("processed_tokens_per_second_incl_cache:", brief)


class PerMessageTableTests(unittest.TestCase):
    def test_per_message_rate_column_is_labelled_processed(self):
        md = _build_per_message_md(build_message_metrics([_step(0, cache_read=100_000)]))
        self.assertIn("| Processed tok/s |", md)
        self.assertNotIn("| Tok/s |", md)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
