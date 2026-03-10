"""
aurora/integrations/llm/Prompt/theme_candidate_prompt.py
主题候选提示词模块。
用于从多个故事摘要中识别涌现出的主题，要求主题具备可证伪性和实用性。
"""

# 系统提示词：指导 LLM 如何执行主题发现
THEME_CANDIDATE_SYSTEM_PROMPT = "你从多个故事摘要中识别涌现出的主题。主题应该是可证伪的且有用的。"

# 用户提示词模板
THEME_CANDIDATE_USER_PROMPT = """{instruction}

故事摘要列表：
{story_summaries}

请返回一个 ThemeCandidate 的 JSON 数组（0 到 N 个）。只包含有具体证据支持的主题。
"""
