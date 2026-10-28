"""Before/after intervention comparison: paired deltas, statistical testing, guardrail detection."""

from __future__ import annotations

import math
from typing import Any

from .batch import BatchResult
from .divergence import count_patterns


# ---------------------------------------------------------------------------
# Metric direction definitions
# ---------------------------------------------------------------------------

# "higher" = higher is better; "lower" = lower is better
METRIC_DIRECTIONS: dict[str, str] = {
    "reference_recall": "higher",
    "behavioral_precision": "higher",
    "alignment_f1": "higher",
    "overhead_ratio": "lower",
    "harmful_ratio": "lower",
    "anchor_write_precision_ref": "higher",
    "anchor_write_precision_cmp": "higher",
    "anchor_write_recall_ref": "higher",
    "anchor_write_recall_cmp": "higher",
    "off_patch_write_ratio_ref": "lower",
    "off_patch_write_ratio_cmp": "lower",
}

# Pattern count metrics — lower count is better
PATTERN_DIRECTIONS: dict[str, str] = {
    "write_retry": "lower",
    "reverted_and_rewritten": "lower",
    "iterative_refinement": "lower",
    "broad_exploration": "lower",
    "error_recovery_overhead": "lower",
    "redundant_search": "lower",
    "dead_end_branch": "lower",
    "ordering_inefficiency": "lower",
    "premature_validation": "lower",
}


# ---------------------------------------------------------------------------
# Task pairing
# ---------------------------------------------------------------------------

def pair_tasks(
    before_results: list[BatchResult],
    after_results: list[BatchResult],
) -> tuple[list[tuple[BatchResult, BatchResult]], list[BatchResult], list[BatchResult]]:
    """Match tasks between before and after batches by task_id.

    Returns (paired, before_only, after_only). Unpaired entries include failed
    comparisons, so an input error cannot silently disappear from the report.
    Each batch must have unique, non-empty string task identifiers.
    """
    def index_results(results: list[BatchResult], side: str) -> dict[str, BatchResult]:
        indexed: dict[str, BatchResult] = {}
        for result in results:
            if not isinstance(result.task_id, str) or not result.task_id.strip():
                raise ValueError(f"{side} task_id must be a non-empty string")
            if result.task_id in indexed:
                raise ValueError(f"{side} has duplicate task_id {result.task_id!r}")
            indexed[result.task_id] = result
        return indexed

    before_map = index_results(before_results, "Before batch")
    after_map = index_results(after_results, "After batch")
    paired_ids = {tid for tid in before_map.keys() & after_map.keys()
                  if before_map[tid].report is not None and after_map[tid].report is not None}

    paired = [(before_map[tid], after_map[tid]) for tid in sorted(paired_ids)]
    before_only = [before_map[tid] for tid in sorted(set(before_map.keys()) - paired_ids)]
    after_only = [after_map[tid] for tid in sorted(set(after_map.keys()) - paired_ids)]

    return paired, before_only, after_only


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

def _extract_metric(report: dict, metric_name: str) -> float | None:
    """Extract a metric value from a comparison report."""
    # Standard alignment metrics
    if metric_name in ("reference_recall", "behavioral_precision", "alignment_f1",
                       "overhead_ratio", "harmful_ratio"):
        return report.get("alignment", {}).get(metric_name)

    # Anchor metrics: "anchor_write_precision_ref" → strip "_ref" and "anchor_" prefix
    # to get "write_precision" which matches anchor.py output keys
    aa = report.get("anchor_analysis")
    if aa is None:
        return None
    if metric_name.endswith("_ref"):
        base = metric_name[:-4]  # strip "_ref"
        if base.startswith("anchor_"):
            base = base[7:]  # strip "anchor_" → "write_precision"
        return aa.get("reference", {}).get(base)
    if metric_name.endswith("_cmp"):
        base = metric_name[:-4]  # strip "_cmp"
        if base.startswith("anchor_"):
            base = base[7:]  # strip "anchor_" → "write_precision"
        return aa.get("compared", {}).get(base)

    return None


def _count_pattern(report: dict, pattern_type: str) -> int:
    """Count occurrences of a pattern type in a report.

    Delegates to the shared predicate so this and eval_layers cannot disagree
    about umbrella types such as 'write_retry'.
    """
    return count_patterns(report.get("patterns", []), pattern_type)


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_metric_deltas(
    paired: list[tuple[BatchResult, BatchResult]],
    significance_threshold: float = 0.05,
) -> dict[str, dict[str, Any]]:
    """Compute before/after deltas and their significance for each metric.

    Returns dict mapping metric name to {before_mean, after_mean, delta, direction,
    before_values, after_values, p_value, p_value_adjusted (Holm, across the
    metrics), significant (adjusted p < threshold), test_method,
    significance_warning}.
    """
    deltas: dict[str, dict[str, Any]] = {}

    for metric_name, good_dir in METRIC_DIRECTIONS.items():
        before_vals = []
        after_vals = []
        for br, ar in paired:
            bv = _extract_metric(br.report, metric_name)
            av = _extract_metric(ar.report, metric_name)
            if bv is not None and av is not None:
                before_vals.append(bv)
                after_vals.append(av)

        if not before_vals:
            continue

        before_mean = sum(before_vals) / len(before_vals)
        after_mean = sum(after_vals) / len(after_vals)
        delta = after_mean - before_mean

        if good_dir == "higher":
            direction = "improved" if delta > 0.01 else ("regressed" if delta < -0.01 else "unchanged")
        else:
            direction = "improved" if delta < -0.01 else ("regressed" if delta > 0.01 else "unchanged")

        deltas[metric_name] = {
            "before_mean": round(before_mean, 4),
            "after_mean": round(after_mean, 4),
            "delta": round(delta, 4),
            "direction": direction,
            "before_values": before_vals,
            "after_values": after_vals,
        }

    # One test per metric, then a Holm correction across them: testing several
    # metrics at 0.05 each would otherwise flag a chance difference as an effect.
    tests = {name: paired_significance(d["before_values"], d["after_values"], significance_threshold)
             for name, d in deltas.items()}
    adjusted = holm_adjust({name: t["p_value"] for name, t in tests.items() if t["p_value"] is not None})
    for name, d in deltas.items():
        p_adj = adjusted.get(name)
        d.update({
            "p_value": tests[name]["p_value"],
            "p_value_adjusted": p_adj,
            "significant": p_adj is not None and p_adj < significance_threshold,
            "test_method": tests[name]["test_method"],
            "significance_warning": tests[name]["warning"],
        })
    return deltas


def compute_pattern_deltas(
    paired: list[tuple[BatchResult, BatchResult]],
) -> dict[str, dict[str, Any]]:
    """Compute pattern frequency deltas between before and after batches.

    Candidate keys include each pattern's ``parent_type`` as well as its
    ``type``: without it the umbrella 'write_retry' never became a key and
    its PATTERN_DIRECTIONS entry was dead. This is additive — the per-``type``
    entries keep their previous frequencies.
    """
    all_types: set[str] = set()
    for br, ar in paired:
        for p in br.report.get("patterns", []):
            all_types.add(p.get("type", ""))
            all_types.add(p.get("parent_type") or "")
        for p in ar.report.get("patterns", []):
            all_types.add(p.get("type", ""))
            all_types.add(p.get("parent_type") or "")

    deltas: dict[str, dict] = {}
    total = len(paired)

    for ptype in sorted(all_types):
        if not ptype:
            continue
        before_count = sum(1 for br, _ in paired if _count_pattern(br.report, ptype) > 0)
        after_count = sum(1 for _, ar in paired if _count_pattern(ar.report, ptype) > 0)
        before_prev = before_count / total if total > 0 else 0
        after_prev = after_count / total if total > 0 else 0

        good_dir = PATTERN_DIRECTIONS.get(ptype, "lower")
        delta = after_prev - before_prev
        if good_dir == "lower":
            direction = "improved" if delta < -0.01 else ("regressed" if delta > 0.01 else "unchanged")
        else:
            direction = "improved" if delta > 0.01 else ("regressed" if delta < -0.01 else "unchanged")

        deltas[ptype] = {
            "before_frequency": round(before_prev, 4),
            "after_frequency": round(after_prev, 4),
            "delta": round(delta, 4),
            "direction": direction,
        }

    return deltas


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def _wilcoxon_signed_rank_p(diffs: list[float]) -> tuple[float, str]:
    """Two-sided p-value of the Wilcoxon signed-rank test, and the method used.

    Zero differences are dropped (Wilcoxon's convention). Without ties and with
    at most 50 differences the p-value is exact, from the null distribution of
    the positive-rank sum over all 2**n sign assignments; otherwise it is the
    normal approximation with the tie-corrected variance. Pure Python, so the
    result does not depend on which optional packages are installed.
    """
    nonzero = [d for d in diffs if d != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0, "wilcoxon_exact"
    order = sorted(range(n), key=lambda i: abs(nonzero[i]))
    ranks = [0.0] * n
    tie_term = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nonzero[order[j + 1]]) == abs(nonzero[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j + 2) / 2  # ranks are 1-based; ties share the mean
        tied = j - i + 1
        tie_term += tied ** 3 - tied
        i = j + 1
    w_plus = sum(rank for rank, d in zip(ranks, nonzero, strict=True) if d > 0)

    if tie_term == 0 and n <= 50:
        max_sum = n * (n + 1) // 2
        counts = [1] + [0] * max_sum  # counts[s]: subsets of {1..n} with rank sum s
        for rank in range(1, n + 1):
            for total in range(max_sum, rank - 1, -1):
                counts[total] += counts[total - rank]
        w = round(w_plus)
        lower = sum(counts[: w + 1])
        upper = sum(counts[w:])
        return min(1.0, 2 * min(lower, upper) / 2 ** n), "wilcoxon_exact"

    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48
    if variance <= 0:
        return 1.0, "wilcoxon_normal"
    z = (w_plus - mean) / math.sqrt(variance)
    return math.erfc(abs(z) / math.sqrt(2)), "wilcoxon_normal"


def paired_significance(
    before_values: list[float],
    after_values: list[float],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Wilcoxon signed-rank test of paired before/after values (unadjusted).

    ``significant`` here compares the raw p-value with *threshold*; when several
    metrics are tested together, :func:`compute_metric_deltas` adjusts for that.
    """
    n = len(before_values)
    if n < 2:
        return {"p_value": None, "significant": False, "test_method": "none",
                "warning": "fewer than 2 paired observations"}
    warning = "fewer than 6 paired observations — low statistical power" if n < 6 else None
    diffs = [a - b for a, b in zip(after_values, before_values, strict=False)]
    p_value, method = _wilcoxon_signed_rank_p(diffs)
    return {
        "p_value": round(p_value, 6),
        "significant": p_value < threshold,
        "test_method": method,
        "warning": warning,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, controlling the family-wise error rate."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for position, (name, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (m - position) * p_value))
        adjusted[name] = round(running, 6)
    return adjusted


# ---------------------------------------------------------------------------
# Guardrail detection
# ---------------------------------------------------------------------------

def detect_guardrail_regressions(
    metric_deltas: dict[str, dict],
    relative_threshold: float = 0.10,
    significance_threshold: float = 0.05,
) -> list[dict[str, Any]]:
    """Detect when improving one metric causes regression in others.

    For each improved metric, check all others for regression beyond threshold.
    """
    # Find improved metrics
    improved = [name for name, d in metric_deltas.items() if d["direction"] == "improved"]

    if not improved:
        return []

    warnings: list[dict] = []
    seen: set[str] = set()

    for regressed_name, d in metric_deltas.items():
        if d["direction"] != "regressed":
            continue
        if regressed_name in seen:
            continue

        before_mean = d["before_mean"]
        delta = abs(d["delta"])

        # Check relative threshold. A regression away from a zero baseline is an
        # unbounded relative change, so treat any real movement as exceeding it
        # instead of silently scoring it 0.
        if before_mean != 0:
            relative_regression = delta / abs(before_mean)
        elif delta != 0:
            relative_regression = 1.0
        else:
            relative_regression = 0.0
        exceeds_relative = relative_regression >= relative_threshold

        # Significance as compute_metric_deltas adjusted it; deltas built by
        # other callers fall back to an unadjusted test.
        if "significant" in d:
            sig = {"p_value": d.get("p_value_adjusted", d.get("p_value")), "significant": d["significant"]}
        else:
            sig = paired_significance(d.get("before_values", []), d.get("after_values", []), significance_threshold)
        exceeds_significance = sig.get("significant", False)

        if exceeds_relative or exceeds_significance:
            seen.add(regressed_name)
            warnings.append({
                "metric": regressed_name,
                "before_mean": d["before_mean"],
                "after_mean": d["after_mean"],
                "delta": d["delta"],
                "direction": "regressed",
                "relative_regression": round(relative_regression, 4),
                "p_value": sig.get("p_value"),
                "significant": sig.get("significant", False),
                "warning": (
                    f"{regressed_name} regressed from {d['before_mean']:.4f} to "
                    f"{d['after_mean']:.4f} ({relative_regression*100:.1f}% relative"
                    f"{', p=' + str(sig['p_value']) if sig.get('p_value') is not None else ''}"
                    f")"
                ),
            })

    return warnings


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

def generate_recommendation(
    metric_deltas: dict[str, dict],
    guardrails: list[dict],
) -> str:
    """Generate a one-line recommendation based on intervention effect."""
    if not metric_deltas:
        return "No comparable metric observations are available."
    improved = [n for n, d in metric_deltas.items() if d["direction"] == "improved"]
    regressed = [n for n, d in metric_deltas.items() if d["direction"] == "regressed"]

    if not improved and not regressed:
        return "No significant changes detected."
    if improved and not guardrails:
        if regressed:
            return (f"Intervention improved {', '.join(improved[:2])} with minor "
                    f"regressions in {', '.join(regressed[:2])} (below guardrail thresholds).")
        return f"Intervention improved {', '.join(improved[:3])} without regressions."
    if improved and guardrails:
        regressed_names = [g["metric"] for g in guardrails]
        return (
            f"Intervention improved {', '.join(improved[:2])} but regressed "
            f"{', '.join(regressed_names[:2])}. Review tradeoffs."
        )
    if regressed and not improved:
        return f"Intervention regressed {', '.join(regressed[:3])} without improvements."
    return "Mixed results — review per-metric deltas."


# ---------------------------------------------------------------------------
# Intervention report
# ---------------------------------------------------------------------------

def build_intervention_report(
    before_manifest: str,
    after_manifest: str,
    paired: list[tuple[BatchResult, BatchResult]],
    before_only: list[BatchResult],
    after_only: list[BatchResult],
    metric_deltas: dict[str, dict],
    pattern_deltas: dict[str, dict],
    guardrails: list[dict],
) -> dict[str, Any]:
    """Assemble the full intervention comparison report."""
    clean_deltas = {}
    for name, d in metric_deltas.items():
        if "significant" not in d:  # deltas assembled outside compute_metric_deltas
            sig = paired_significance(d.get("before_values", []), d.get("after_values", []))
            d = {**d, "p_value": sig["p_value"], "p_value_adjusted": None, "significant": sig["significant"],
                 "test_method": sig["test_method"], "significance_warning": sig["warning"]}
        clean_deltas[name] = {
            "before_mean": d["before_mean"],
            "after_mean": d["after_mean"],
            "delta": d["delta"],
            "direction": d["direction"],
            "p_value": d.get("p_value"),
            "p_value_adjusted": d.get("p_value_adjusted"),
            "significant": d.get("significant", False),
            "test_method": d.get("test_method"),
            "significance_warning": d.get("significance_warning"),
        }

    recommendation = (generate_recommendation(metric_deltas, guardrails) if paired else
                      "Intervention effects are unavailable: no successful task pairs are available.")

    return {
        "before_manifest": before_manifest,
        "after_manifest": after_manifest,
        "paired_tasks": len(paired),
        "unpaired_before": len(before_only),
        "unpaired_after": len(after_only),
        "input_errors": [
            {"side": side, "task_id": result.task_id,
             "error": result.error or "Comparison report is unavailable."}
            for side, results in (("before", before_only), ("after", after_only))
            for result in results if result.report is None
        ],
        "intervention_effect": clean_deltas,
        "significance": "Wilcoxon signed-rank per metric; Holm-adjusted across metrics, alpha 0.05",
        "guardrail_warnings": guardrails,
        "pattern_deltas": pattern_deltas,
        "recommendation": recommendation,
    }
