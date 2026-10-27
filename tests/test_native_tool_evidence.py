"""Native transcript mirrors must not turn absent outcomes into success."""
import json
import agent_trajviz as tv


def run_with_calls(*calls):
    return tv.Trajectory.from_steps([{'role': 'assistant', 'tool_calls': list(calls)}])


def test_native_unknown_outcomes_are_unavailable_in_metrics_health_and_chart():
    run = run_with_calls(
        {'tool_name': 'Read', 'status': 'unknown', 'metadata': {'_outcome_observed': False}},
        {'tool_name': 'Bash', 'status': 'completed'},
    )
    assert run.metrics['tool_call_count'] == 2
    assert run.metrics['tool_success'] == 1
    assert run.metrics['tool_unknown_outcomes'] == 1
    assert run.metrics['tool_success_rate'] is None
    assert run.metrics['tool_system_failure_rate'] is None
    assert [c['ok'] for c in run.view('tool_outcomes').data['calls']] == [None, True]
    for metric in ['Tool Success', 'Errors']:
        assert next(v for v in run.health if v['metric'] == metric)['status'] == 'unavailable'
    copy = tv.Trajectory.from_dict(json.loads(json.dumps(run.to_dict())))
    assert copy.metrics == run.metrics


def test_native_observed_failures_remain_observed_with_unknown_calls():
    run = run_with_calls(
        {'tool_name': 'Bash', 'status': 'error', 'error': 'exit 1'},
        {'tool_name': 'Read', 'status': 'unknown', 'metadata': {'_outcome_observed': False}},
    )
    assert run.metrics['tool_fail'] == 1
    assert run.metrics['tool_success_rate'] is None
    assert [c['ok'] for c in run.view('tool_outcomes').data['calls']] == [False, None]


def test_existing_native_observed_outcomes_keep_their_rate():
    run = run_with_calls(
        {'tool_name': 'Read', 'status': 'completed'},
        {'tool_name': 'Bash', 'status': 'error', 'error': 'exit 1'},
    )
    assert run.metrics['tool_success_rate'] == 50


def test_unknown_edit_cannot_establish_precision_or_recovery():
    from agent_trajviz.insight.tool_failure import step_has_success

    run = run_with_calls({'tool_name': 'Edit', 'status': 'unknown',
                          'metadata': {'_outcome_observed': False}})
    assert run.view('diagnostics').data['edit_precision'] is None
    assert run.view('diagnostics').data['edit_success'] == 0
    assert not step_has_success(run.steps[0])
