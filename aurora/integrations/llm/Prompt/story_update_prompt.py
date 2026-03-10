"""
aurora/integrations/llm/Prompt/story_update_prompt.py
故事更新提示词模块。
用于根据新的情节更新故事弧摘要，确保叙事的连贯性和简洁性。
"""

# 系统提示词：指导 LLM 如何执行故事更新
STORY_UPDATE_SYSTEM_PROMPT = "你根据新的情节更新故事弧摘要。请保持叙述连贯且紧凑。"

# 用户提示词模板
STORY_UPDATE_USER_PROMPT = """{instruction}

目前的故事状态：
{story_so_far}

新情节：
{new_plot}

返回 StoryUpdate JSON。
"""
