from __future__ import annotations

from aurora.integrations.llm.Prompt.plot_extraction_prompt import (
    PLOT_EXTRACTION_SYSTEM_PROMPT,
    PLOT_EXTRACTION_USER_PROMPT,
)


def test_plot_extraction_prompt_uses_exact_schema_keys():
    assert "严禁将多个字段合并为一个键" in PLOT_EXTRACTION_SYSTEM_PROMPT
    assert "必须精确使用请求的 Schema 键名" in PLOT_EXTRACTION_SYSTEM_PROMPT
    assert "action: 简短具体的动作描述字符串" in PLOT_EXTRACTION_USER_PROMPT
    assert "context: 简短具体的背景/语境描述字符串" in PLOT_EXTRACTION_USER_PROMPT
    assert "outcome: 简短具体的结果描述字符串" in PLOT_EXTRACTION_USER_PROMPT
    assert "goal: 简短的目标描述字符串" in PLOT_EXTRACTION_USER_PROMPT
    assert "decision: 简短的决策描述字符串" in PLOT_EXTRACTION_USER_PROMPT
    assert "不要发明像 `action/context/outcome` 这样的组合键" in PLOT_EXTRACTION_USER_PROMPT
