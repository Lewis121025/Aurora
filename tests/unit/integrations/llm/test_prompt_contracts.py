from __future__ import annotations

from aurora.integrations.llm.Prompt import (
    PLOT_EXTRACTION_SYSTEM_PROMPT,
    PLOT_EXTRACTION_USER_PROMPT,
)


def test_plot_extraction_prompt_uses_exact_schema_keys():
    assert "never merge multiple fields into one key" in PLOT_EXTRACTION_SYSTEM_PROMPT
    assert "action: short concrete string" in PLOT_EXTRACTION_USER_PROMPT
    assert "context: short concrete string" in PLOT_EXTRACTION_USER_PROMPT
    assert "outcome: short concrete string" in PLOT_EXTRACTION_USER_PROMPT
    assert "goal: short string" in PLOT_EXTRACTION_USER_PROMPT
    assert "decision: short string" in PLOT_EXTRACTION_USER_PROMPT
    assert "Do not invent combined keys like `action/context/outcome`" in PLOT_EXTRACTION_USER_PROMPT
