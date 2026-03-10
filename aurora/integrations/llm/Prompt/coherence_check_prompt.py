"""
aurora/integrations/llm/Prompt/coherence_check_prompt.py
一致性检查提示词模块。
用于检查两个记忆元素之间是否存在事实、时间、因果或主题上的一致性冲突。
"""

# 系统提示词：指导 LLM 如何执行一致性检查
COHERENCE_CHECK_SYSTEM_PROMPT = (
    "你检查记忆元素之间的一致性冲突。请考虑事实、时间、因果和主题的一致性。"
)

# 用户提示词模板
COHERENCE_CHECK_USER_PROMPT = """{instruction}

元素 A：
{element_a}

元素 B：
{element_b}

检查这些元素之间是否相互一致。
返回 CoherenceCheck JSON。
"""
