"""The labeler's own sidecars must satisfy every public label-view schema."""
from pathlib import Path
import json
from unittest.mock import patch

import jsonschema

import agent_trajviz as tv
from agent_trajviz.figures import figures
from agent_trajviz.labeling.labeler import label_trajectory


FIXTURE = Path(__file__).parent / 'fixtures/codearts_minimal.json'


def _strict(value):
    if isinstance(value, dict):
        value = {key: _strict(item) for key, item in value.items()}
        if 'properties' in value and 'additionalProperties' not in value:
            value['additionalProperties'] = False
    elif isinstance(value, list):
        value = [_strict(item) for item in value]
    return value


def test_labeled_trajectory_and_comparison_exports_preserve_typed_provenance(tmp_path):
    output = tmp_path / 'labels.json'
    with patch('agent_trajviz.labeling.llm.call_llm',
               return_value='{"phase":"understand","action":"code_reading"}'):
        labeled = label_trajectory(str(FIXTURE), str(output), base_url='https://example.invalid/v1',
                                   api_key='synthetic-key', model='test-model')
    run = tv.load(FIXTURE)
    families = [tv.label_views(output), tv.label_comparison_views(output, output),
                tv.comparison_views(tv.compare(run, run)),
                tv.run_group_views(tv.compare_runs([run, run], labels=['reference', 'compared']))]
    assert sum(len(views) for views in families) == 17
    timeline = families[0]['label_timeline'].data['steps']
    assert [row['raw_index'] for row in timeline] == [step['raw_index'] for step in run.steps]
    assert [row['label_source'] for row in timeline] == [step['label_source'] for step in labeled['steps']]
    for index, views in enumerate(families):
        for name, view in views.items():
            jsonschema.Draft202012Validator(_strict(tv.view_schema(name))).validate(json.loads(view.to_json()))
        for chart in figures(views).values():
            json.loads(chart.to_json())
        paths = tv.export_json(views, tmp_path / f'family-{index}')
        assert len(paths) == len(views) + 1
