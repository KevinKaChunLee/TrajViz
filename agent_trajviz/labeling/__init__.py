"""LLM step labeling: tag each trajectory step with a phase and an action.

``agent-trajviz-label`` (``step_labeler_v2``) writes a label sidecar with one
record per parsed step; ``python -m agent_trajviz.labeling.step_labeler`` is the
older assistant-only CLI. Both read the bundled ``taxonomy.md`` unless
``--taxonomy`` names another file, and need the ``llm`` extra
(``pip install "agent-trajviz[llm]"``) plus ``LABEL_BASE_URL``,
``LABEL_API_KEY`` and ``LABEL_MODEL``.
"""
