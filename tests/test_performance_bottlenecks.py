"""Performance bottleneck detection — outliers with actionable causes."""

import unittest

from agent_trajviz.insight.diagnostics import decompose_hotspot_duration, detect_performance_bottlenecks


def _asst(idx: int, duration: float, *, tools=None, tokens=None) -> dict:
    return {
        "index": idx,
        "role": "assistant",
        "duration": duration,
        "tokens": tokens or {"total": 1000, "input": 800, "output": 200, "cache_read": 0},
        "tool_calls": tools or [],
    }


def _tool(name: str, duration_ms: float, **inp) -> dict:
    return {
        "tool_name": name,
        "input": inp,
        "status": "success",
        "duration_ms": duration_ms,
    }


class PerformanceBottleneckTests(unittest.TestCase):
    def test_model_request_interval_does_not_include_attached_tool_or_idle(self):
        step = _asst(0, 5, tools=[_tool("Bash", 10_000, command="pytest")])
        step["timing_source"] = "model_request"
        decomposition = decompose_hotspot_duration(step, idle_gap=120)
        self.assertEqual(decomposition["inference_s"], 5)
        self.assertEqual(decomposition["inference_pct"], 100)
        self.assertEqual(decomposition["tool_s"], 0)
        self.assertEqual(decomposition["idle_s"], 0)
        self.assertFalse(decomposition["timing_incomplete"])
        self.assertIsNone(decomposition["dominant_tool"])
        self.assertEqual(step["tool_calls"][0]["duration_ms"], 10_000)
        self.assertEqual(detect_performance_bottlenecks([step], [{"index": 0, "idle_before_s": 120}]), [])

    def test_model_request_outlier_is_inference_even_with_longer_tools(self):
        steps = [_asst(i, 1) for i in range(4)]
        outlier = _asst(4, 40, tools=[_tool("Bash", 80_000, command="pytest")])
        outlier["timing_source"] = "model_request"
        steps.append(outlier)
        analytics = [{"index": i, "idle_before_s": 120 if i == 4 else 0, "cache_ratio": 0.8}
                     for i in range(5)]
        found = detect_performance_bottlenecks(steps, analytics)
        issue = next(item for item in found if item["step_idx"] == 4)
        self.assertEqual(issue["cause"], "inference")
        self.assertEqual(issue["decomposition"]["inference_s"], 40)

    def test_uniform_slow_steps_are_not_bottlenecks(self):
        # All similar durations → no session-relative outlier Issues.
        steps = [_asst(i, 20.0) for i in range(5)]
        analytics = [{"index": i, "idle_before_s": 0, "cache_ratio": 0.8} for i in range(5)]
        self.assertEqual(detect_performance_bottlenecks(steps, analytics), [])

    def test_tool_outlier_is_bottleneck(self):
        steps = [
            _asst(0, 2.0, tools=[_tool("Read", 200, file_path="a.py")]),
            _asst(1, 2.5, tools=[_tool("Read", 300, file_path="b.py")]),
            _asst(2, 3.0, tools=[_tool("Edit", 400, file_path="a.py")]),
            _asst(3, 2.0, tools=[_tool("Read", 250, file_path="c.py")]),
            _asst(
                4, 40.0,
                tools=[_tool("Bash", 38000, command="npm test")],
            ),
        ]
        analytics = [{"index": i, "idle_before_s": 0.1, "cache_ratio": 0.7} for i in range(5)]
        found = detect_performance_bottlenecks(steps, analytics)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["cause"], "tool")
        self.assertEqual(found[0]["step_idx"], 4)
        self.assertIn("Bash", found[0]["title"])

    def test_idle_queue_outlier(self):
        steps = [
            _asst(0, 2.0),
            _asst(1, 2.0),
            _asst(2, 2.0),
            _asst(3, 3.0),  # short step after long idle
        ]
        analytics = [
            {"index": 0, "idle_before_s": 0, "cache_ratio": 0.8},
            {"index": 1, "idle_before_s": 0.2, "cache_ratio": 0.8},
            {"index": 2, "idle_before_s": 0.2, "cache_ratio": 0.8},
            {"index": 3, "idle_before_s": 25.0, "cache_ratio": 0.8},
        ]
        found = detect_performance_bottlenecks(steps, analytics)
        self.assertTrue(found)
        self.assertEqual(found[0]["cause"], "idle")
        self.assertEqual(found[0]["step_idx"], 3)
        self.assertIn("Idle", found[0]["title"])

    def test_mere_top_n_without_cause_skipped(self):
        # Long inference but below context/inference cause thresholds → not an Issue.
        steps = [
            _asst(0, 2.0),
            _asst(1, 2.0),
            _asst(2, 2.0),
            _asst(3, 12.0, tokens={"total": 2000, "input": 1500, "output": 500, "cache_read": 1000}),
        ]
        analytics = [{"index": i, "idle_before_s": 0, "cache_ratio": 0.8} for i in range(4)]
        self.assertEqual(detect_performance_bottlenecks(steps, analytics), [])


if __name__ == "__main__":
    unittest.main()
