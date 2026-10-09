"""LLM step labeling: tag each trajectory step with a phase and an action.

``agent-trajviz-label`` (``python -m agent_trajviz.labeling``) writes a sidecar
JSON with one record per parsed step; ``--assistant-only`` keeps just the
assistant steps. It reads the bundled ``taxonomy.md`` unless ``--taxonomy``
names another file, and needs the ``llm`` extra
(``pip install "agent-trajviz[llm]"``) plus ``LABEL_BASE_URL``,
``LABEL_API_KEY`` and ``LABEL_MODEL``.

* :mod:`.taxonomy` — the taxonomy, the prompts and reading the model's answer;
* :mod:`.llm` — configuration and the HTTP calls;
* :mod:`.labeler` — labeling a whole trajectory, and the command.
"""
