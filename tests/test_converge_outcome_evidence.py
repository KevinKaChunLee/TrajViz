"""Comparison effects require observed execution, not just attempted calls."""
from copy import deepcopy

import pytest

import agent_trajviz as tv
from agent_trajviz.converge.alignment import build_comparison_report_from_steps
from agent_trajviz.converge.canonical import assign_effect_labels, canonicalize_steps
from agent_trajviz.converge.cli import _print_summary


def _write(*, observed=True, status='completed', error=None):
    return {'tool_name': 'Edit', 'input': {'file_path': 'example.py'}, 'status': status,
            'error': error, 'metadata': {'_outcome_observed': observed}}


def _actions(calls):
    run = tv.Trajectory.from_steps([{'role': 'assistant', 'tool_calls': [call]} for call in calls])
    original = deepcopy(run.steps)
    actions = canonicalize_steps(run.steps)
    assign_effect_labels(actions, run.steps)
    assert run.steps == original
    return actions


@pytest.mark.parametrize('call', [
    _write(observed=False),
    {'tool_name': 'Bash', 'input': {'command': 'pytest tests/test_example.py'},
     'status': 'completed', 'metadata': {'_outcome_observed': False}},
    {'tool_name': 'Read', 'input': {'file_path': 'tests/test_example.py'},
     'status': 'completed', 'metadata': {'_outcome_observed': False}},
])
def test_unobserved_outcome_cannot_claim_success_survival_or_justification(call):
    action, = _actions([call])
    assert action.effect_label == 'unknown'
    assert action.effect_detail == {'reason': 'tool_outcome_unobserved'}


def test_unknown_write_does_not_prove_reversion_or_survival_of_earlier_write():
    actions = _actions([_write(), _write(observed=False)])
    assert [a.effect_label for a in actions] == ['unknown', 'unknown']
    assert actions[0].effect_detail['reason'] == 'later_write_outcome_unobserved'
    assert all('survives_to_final_patch' not in a.effect_detail for a in actions)


def test_observed_later_write_does_not_claim_an_unknown_attempt_executed():
    actions = _actions([_write(observed=False), _write()])
    assert [a.effect_label for a in actions] == ['unknown', 'survived']


def test_failed_later_write_did_not_overwrite_the_successful_write():
    actions = _actions([_write(), _write(status='error', error='permission denied')])
    assert [a.effect_label for a in actions] == ['survived', 'failed']


def test_later_observed_write_keeps_genuine_reversion_despite_intermediate_unknown():
    actions = _actions([_write(), _write(observed=False), _write()])
    assert [a.effect_label for a in actions] == ['reverted', 'unknown', 'survived']


def test_error_only_completed_result_uses_shared_failure_evidence():
    action, = _actions([_write(error='permission denied')])
    assert action.effect_label == 'failed'
    assert action.effect_detail['reason'] == 'tool_result_error'


def test_summary_does_not_print_undefined_anchor_precision_as_zero(capsys):
    run = tv.Trajectory.from_steps([{'role': 'assistant', 'text_preview': 'Review the task'}])
    report = build_comparison_report_from_steps(run.raw, run.raw, run.steps, run.steps,
                                               anchor_files={'example.py'})
    _print_summary(report)
    summary = capsys.readouterr().out
    assert 'Write precision: n/a' in summary
    assert 'Off-patch ratio: n/a' in summary
    assert 'Write recall:    0.0%' in summary


def test_text_only_history_needs_no_tool_matching():
    from unittest.mock import patch
    run = tv.Trajectory.from_steps([{'role': 'assistant', 'text_preview': 'Consider the task'}] * 2500)
    actions = canonicalize_steps(run.steps)
    with patch('agent_trajviz.converge.canonical._find_own_tool_call',
               side_effect=AssertionError('text-only actions need no tool lookup')):
        assign_effect_labels(actions, run.steps)
    assert len(actions) == 2500
    assert all(action.effect_label == 'unknown' for action in actions)
