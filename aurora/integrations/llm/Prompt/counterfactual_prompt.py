"""
aurora/integrations/llm/Prompt/counterfactual_prompt.py
反事实推理提示词模块。
用于进行反事实推理：如果某事当初有所不同，会发生什么。基于因果结构而非表面相似性进行推理。
"""

# 系统提示词：指导 LLM 如何执行反事实推理
COUNTERFACTUAL_SYSTEM_PROMPT = (
    "你进行反事实推理：如果某事当初有所不同，会发生什么。请基于因果结构而非表面相似性进行推理。"
)

# 用户提示词模板
COUNTERFACTUAL_USER_PROMPT = """{instruction}

事实情况：
{factual}

反事实问题：
如果 {antecedent}，那么 {query} 会发生什么？

相关上下文：
{context}

返回包含你的推理过程的 CounterfactualQuery JSON。
"""
