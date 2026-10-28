"""Batch identities and failures survive CLI and intervention reporting."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from agent_trajviz.converge.batch import BatchResult, parse_manifest
from agent_trajviz.converge.intervention import build_intervention_report, pair_tasks

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / 'tests/fixtures/codearts_minimal.json'


def _manifest(tmp_path, name='batch.json', ids=('ok',), invalid=False):
    bad = tmp_path / 'unrecognized.json'
    bad.write_text('{"not":"a trajectory"}', encoding='utf-8')
    entries = [{'task_id': identity, 'reference': str(FIXTURE),
                'compared': str(bad if invalid and index else FIXTURE)}
               for index, identity in enumerate(ids)]
    path = tmp_path / name
    path.write_text(json.dumps(entries), encoding='utf-8')
    return path


def _cli(*args):
    return subprocess.run([sys.executable, '-m', 'agent_trajviz.converge.cli', *map(str, args)],
                          cwd=REPO, capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('task_id', [None, False, 0, [], {}, '', ' \t'])
def test_manifest_refuses_non_string_or_blank_task_ids(tmp_path, task_id):
    with pytest.raises(ValueError, match='task_id must be a non-empty string'):
        parse_manifest(_manifest(tmp_path, ids=(task_id,)))


def test_duplicate_ids_are_rejected_before_intervention_can_discard_a_task(tmp_path):
    manifest = _manifest(tmp_path, ids=('same', 'same'))
    with pytest.raises(ValueError, match='duplicate task_id'):
        parse_manifest(manifest)
    result = _cli('--before', manifest, '--after', manifest)
    assert result.returncode == 1
    assert 'duplicate task_id' in result.stderr
    assert 'Traceback' not in result.stderr
    assert not result.stdout


def test_task_identifiers_keep_exact_unicode_and_order(tmp_path):
    ids = ('task, "quoted"', '第二任务', ' task with spaces ')
    assert [e.task_id for e in parse_manifest(_manifest(tmp_path, ids=ids))] == list(ids)


@pytest.mark.parametrize('mode', ['batch', 'intervention'])
def test_partial_positional_cannot_be_ignored_by_another_mode(tmp_path, mode):
    manifest = _manifest(tmp_path)
    args = ['--batch', manifest] if mode == 'batch' else ['--before', manifest, '--after', manifest]
    result = _cli('ignored.json', *args)
    assert result.returncode == 2
    assert 'Use one mode' in result.stderr
    assert not result.stdout


def test_incomplete_pair_is_a_usage_error():
    result = _cli(FIXTURE)
    assert result.returncode == 2
    assert 'Provide both' in result.stderr
    assert not result.stdout


def test_partial_batch_summary_reports_input_error_instead_of_fabricated_zero_metrics(tmp_path):
    manifest = _manifest(tmp_path, ids=('ok', 'bad'), invalid=True)
    result = _cli('--batch', manifest, '--output', 'summary')
    assert result.returncode == 0, result.stderr
    assert '1 compared, 1 could not be compared' in result.stdout
    failed_line = next(line for line in result.stdout.splitlines() if 'bad:' in line)
    assert 'ERROR' in failed_line and 'R=' not in failed_line
    report = json.loads(_cli('--batch', manifest).stdout)
    assert report['per_task'][1]['status'] == 'error'
    assert report['per_task'][1]['error'] in failed_line
    assert 'alignment_summary' not in report['per_task'][1]


@pytest.mark.parametrize('side', ['before', 'after'])
def test_pairing_rejects_duplicate_ids_for_in_memory_callers(side):
    duplicate = [BatchResult('same', report={}), BatchResult('same', error='unavailable')]
    with pytest.raises(ValueError, match='duplicate task_id'):
        pair_tasks(duplicate if side == 'before' else [], duplicate if side == 'after' else [])


def test_failed_before_after_entries_are_retained_and_do_not_enter_statistics(tmp_path):
    before = _manifest(tmp_path, 'before.json', ids=('ok', 'bad'), invalid=True)
    after = _manifest(tmp_path, 'after.json', ids=('ok', 'bad'))
    result = _cli('--before', before, '--after', after)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['paired_tasks'] == 1
    assert (report['unpaired_before'], report['unpaired_after']) == (1, 1)
    assert report['input_errors'][0]['side'] == 'before'
    assert report['input_errors'][0]['task_id'] == 'bad'
    assert report['intervention_effect']['alignment_f1']['before_mean'] == 1
    assert report['intervention_effect']['alignment_f1']['after_mean'] == 1
    summary = _cli('--before', before, '--after', after, '--output', 'summary')
    assert 'Before bad: ERROR' in summary.stdout


def test_all_failed_intervention_preserves_both_input_errors():
    paired, before_only, after_only = pair_tasks([BatchResult('same', error='before error')],
                                                [BatchResult('same', error='after error')])
    report = build_intervention_report('before', 'after', paired, before_only, after_only, {}, {}, [])
    assert report['paired_tasks'] == 0
    assert 'unavailable' in report['recommendation']
    assert 'No significant changes' not in report['recommendation']
    assert (report['unpaired_before'], report['unpaired_after']) == (1, 1)
    assert report['input_errors'] == [
        {'side': 'before', 'task_id': 'same', 'error': 'before error'},
        {'side': 'after', 'task_id': 'same', 'error': 'after error'},
    ]


def test_disjoint_batches_have_no_basis_for_intervention_recommendations():
    paired, before_only, after_only = pair_tasks([BatchResult('before-task', report={})],
                                                [BatchResult('after-task', report={})])
    report = build_intervention_report('before', 'after', paired, before_only, after_only, {}, {}, [])
    assert report['paired_tasks'] == 0
    assert 'unavailable' in report['recommendation']
    assert report['input_errors'] == []


def test_empty_metric_observations_are_not_a_no_change_verdict():
    from agent_trajviz.converge.intervention import generate_recommendation
    assert generate_recommendation({}, []) == 'No comparable metric observations are available.'
