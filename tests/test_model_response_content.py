"""Missing model text must not become a fabricated string answer."""
from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

import agent_trajviz as tv
from agent_trajviz import llm
from agent_trajviz.labeling import llm as labeling_llm


CONFIG = llm.LLMConfig(base_url='https://example.invalid/v1', api_key='synthetic-key', model='test',
                       provider='openai', temperature=None, max_tokens=256, timeout=1, source='analyze')


class Response:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {'choices': [{'message': {'content': self.content}}]}


def _run():
    return tv.Trajectory.from_steps([{'role': 'user', 'text_preview': 'Inspect example.py'}])


def test_null_analysis_content_uses_empty_reply_feedback():
    with patch.object(requests, 'post', return_value=Response(None)):
        answer = llm.ask(_run(), 'Explain.', config=CONFIG)[-1]['content']
    assert 'empty reply' in answer
    assert answer != 'None'


def test_null_label_response_has_no_fabricated_text():
    with patch.object(requests, 'post', return_value=Response(None)):
        result = labeling_llm.call_llm(CONFIG.base_url, CONFIG.api_key, CONFIG.model, 'system', 'user')
    assert result == ''


@pytest.mark.parametrize('content', [{'private': 'response'}, ['response'], 0, False])
def test_non_text_analysis_is_a_safe_request_error(content):
    with patch.object(requests, 'post', return_value=Response(content)):
        answer = llm.ask(_run(), 'Explain.', config=CONFIG)[-1]['content']
    assert 'request failed: ValueError' in answer
    assert 'private' not in answer


@pytest.mark.parametrize('content', [{'private': 'response'}, ['response'], 0, False])
def test_non_text_label_response_is_rejected(content):
    with (patch.object(requests, 'post', return_value=Response(content)),
          pytest.raises(ValueError, match='not text')):
        labeling_llm.call_llm(CONFIG.base_url, CONFIG.api_key, CONFIG.model, 'system', 'user')
