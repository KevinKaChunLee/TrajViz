"""A step whose export reports an impossible token count is not drawn as fact.

Some OpenCode exports write a NEGATIVE input (the cache read is subtracted
from an input that was already cache-exclusive). The metrics already leave
such a field out of the session totals and mark them partial, but the
per-step token chart still stacked the remaining fields: a step reporting a
total of 37,889 was drawn 45,232 tokens tall, with nothing on the chart to say
its breakdown could not be trusted. Such a step is now drawn as its reported
total in a separate, labelled series, and the title counts them.
"""

from __future__ import annotations

import unittest

from agent_trajviz.insight.charts.usage import build_token_chart
from agent_trajviz.insight.help import HELP_TEXT


def _step(i, *, total, inp, out=0, rea=0, cr=0, cw=0):
    return {"index": i, "role": "assistant", "agent": "", "model_id": "m",
            "tokens": {"total": total, "input": inp, "output": out, "reasoning": rea,
                       "cache_read": cr, "cache_write": cw},
            "tool_calls": [], "parts": []}


# The real record: total == input + output + cache_read, with input negative.
CORRUPT = _step(1, total=37_889, inp=-7_343, out=176, rea=28, cr=45_056)
HEALTHY = _step(0, total=51_350, inp=1_000, out=200, rea=150, cr=50_000)


def _heights(fig) -> dict[int, float]:
    per: dict[int, float] = {}
    for trace in fig.data:
        for x, y in zip(trace.x or [], trace.y or [], strict=True):
            per[x] = per.get(x, 0) + (y or 0)
    return per


class ImpossibleBreakdownTests(unittest.TestCase):
    def test_bar_is_the_reported_total_not_the_remaining_fields(self):
        for fmt in (None, "opencode"):
            with self.subTest(format=fmt):
                heights = _heights(build_token_chart([HEALTHY, CORRUPT], format=fmt))
                self.assertEqual(heights[1], 37_889)
                self.assertEqual(heights[0], 51_350)

    def test_the_step_is_drawn_in_its_own_labelled_series(self):
        fig = build_token_chart([HEALTHY, CORRUPT])
        flagged = [t for t in fig.data if "breakdown" in (t.name or "").lower()]
        self.assertEqual(len(flagged), 1)
        trace = flagged[0]
        self.assertEqual([x for x, y in zip(trace.x, trace.y, strict=True) if y], [1])
        # The hover names the field and the value the export reported.
        self.assertIn("input", str(trace.customdata[1]))
        self.assertIn("-7,343", str(trace.customdata[1]))

    def test_healthy_steps_keep_their_breakdown(self):
        fig = build_token_chart([HEALTHY, CORRUPT])
        cache = next(t for t in fig.data if t.name == "Cache Read")
        self.assertEqual(list(cache.y), [50_000, 0])

    def test_title_counts_the_flagged_steps(self):
        title = build_token_chart([HEALTHY, CORRUPT, _step(2, total=10, inp=-5, cr=15)]).layout.title.text
        self.assertIn("2 steps", title)
        self.assertEqual(build_token_chart([HEALTHY]).layout.title.text, "Token Usage by Step")

    def test_a_clean_trajectory_gets_no_extra_series(self):
        names = [t.name for t in build_token_chart([HEALTHY]).data]
        self.assertFalse(any("breakdown" in (n or "").lower() for n in names))

    def test_an_impossible_total_is_not_drawn_negative(self):
        fig = build_token_chart([_step(0, total=-50, inp=0)])
        self.assertTrue(all((y or 0) >= 0 for t in fig.data for y in (t.y or [])))
        self.assertIn("1 step", fig.layout.title.text)


class TokensHelpTests(unittest.TestCase):
    def test_help_says_what_happens_to_each_kind_of_bad_count(self):
        text = HELP_TEXT["tokens"]
        # Negative counts leave the field's total; the per-step chart says so.
        self.assertIn("negative", text)
        self.assertIn("per-step chart", text)
        # A cache read above the step total is a share problem: the fields are
        # still summed, so the help must not claim they are excluded.
        self.assertNotIn("more cache-read tokens than the step total) is excluded", text)
        self.assertIn("cache share", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
