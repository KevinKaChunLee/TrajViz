"""Intervention significance: deterministic, correct, and corrected for multiplicity.

The test used to be SciPy's Wilcoxon when SciPy happened to be importable and a
sign test otherwise, so the same before/after data could be significant on one
machine and not on another. It is now a pure-Python Wilcoxon signed-rank test,
checked here against SciPy where SciPy is available, and the per-metric
p-values are Holm-adjusted before anything is called significant.
"""

from __future__ import annotations

import random
import subprocess
import sys
import unittest
from pathlib import Path

from agent_trajviz.converge import intervention

try:
    from scipy.stats import wilcoxon as scipy_wilcoxon
except ImportError:  # pragma: no cover - SciPy is not a dependency
    scipy_wilcoxon = None

REPO = Path(__file__).resolve().parents[1]


@unittest.skipIf(scipy_wilcoxon is None, "SciPy not installed")
class AgreesWithScipyTests(unittest.TestCase):
    def test_exact_p_values_match_for_tie_free_samples(self):
        rng = random.Random(7)
        for n in range(2, 21):
            for _ in range(5):
                # distinct magnitudes and random signs: no ties, no zeros
                diffs = [m * rng.choice((-1, 1)) for m in rng.sample(range(1, 201), n)]
                with self.subTest(n=n, diffs=diffs):
                    ours, method = intervention._wilcoxon_signed_rank_p([float(d) for d in diffs])
                    self.assertEqual(method, "wilcoxon_exact")
                    theirs = scipy_wilcoxon(diffs, method="exact").pvalue
                    self.assertAlmostEqual(ours, theirs, places=9)

    def test_normal_approximation_matches_with_ties(self):
        rng = random.Random(11)
        for _ in range(20):
            diffs = [rng.choice([-3, -2, -1, 1, 2, 3, 4]) for _ in range(rng.randint(8, 40))]
            with self.subTest(diffs=diffs):
                ours, method = intervention._wilcoxon_signed_rank_p([float(d) for d in diffs])
                self.assertEqual(method, "wilcoxon_normal")
                theirs = scipy_wilcoxon(diffs, method="approx").pvalue
                self.assertAlmostEqual(ours, theirs, places=6)


class DeterminismTests(unittest.TestCase):
    def test_the_result_does_not_depend_on_scipy_being_installed(self):
        before, after = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2.0, 2.5, 5.0, 4.5, 7.0, 8.5, 9.0]
        here = intervention.paired_significance(before, after)
        code = f"""
import importlib.abc, sys
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "scipy":
            raise ModuleNotFoundError(name)
sys.meta_path.insert(0, _Block()); sys.path.insert(0, {str(REPO)!r})
from agent_trajviz.converge import intervention
print(intervention.paired_significance({before!r}, {after!r}))
"""
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), str(here))
        self.assertTrue(here["test_method"].startswith("wilcoxon"))


class HolmTests(unittest.TestCase):
    def test_holm_adjustment(self):
        adjusted = intervention.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertEqual(adjusted, {"a": 0.03, "c": 0.06, "b": 0.06})

    def test_significance_is_decided_on_the_adjusted_p_value(self):
        from agent_trajviz.converge.batch import BatchResult

        def result(recall: float) -> BatchResult:
            report = {"alignment": {"reference_recall": recall, "behavioral_precision": 0.5,
                                    "alignment_f1": 0.5, "overhead_ratio": 1.0},
                      "patterns": [], "outcome": {}, "milestones": {}}
            return BatchResult(task_id=str(recall), report=report, error=None)

        paired = [(result(0.1 * i), result(0.1 * i + 0.3 + 0.01 * i)) for i in range(1, 7)]
        deltas = intervention.compute_metric_deltas(paired)
        recall = deltas["reference_recall"]
        self.assertEqual(recall["p_value"], 0.03125)  # exact two-sided, n=6, all positive
        self.assertIsNotNone(recall["p_value_adjusted"])
        self.assertGreaterEqual(recall["p_value_adjusted"], recall["p_value"])
        self.assertEqual(recall["significant"], recall["p_value_adjusted"] < 0.05)


if __name__ == "__main__":
    unittest.main()
