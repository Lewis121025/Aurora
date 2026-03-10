"""
aurora/integrations/llm/Prompt/causal_relation_prompt.py
因果关系识别提示词模块。
用于识别两个事件之间的因果关系，侧重于实际因果而非简单的相关性或时间顺序。
"""

# 系统提示词：指导 LLM 如何识别因果关系
CAUSAL_RELATION_SYSTEM_PROMPT = (
    "你识别事件之间的因果关系。关注实际的因果联系，而不仅仅是相关性或时间序列。请保持保守：只有在有明确证据时才主张因果关系。"
)

# 用户提示词模板
CAUSAL_RELATION_USER_PROMPT = """{instruction}

事件 A：
{event_a}

事件 B：
{event_b}

上下文：
{context}

分析 A 和 B 之间是否存在因果关系。
返回 CausalRelation JSON。
"""
