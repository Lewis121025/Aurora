"""
aurora/integrations/llm/Prompt/causal_chain_prompt.py
因果链提取提示词模块。
用于从事件序列中提取因果链，识别根本原因、中间效应和最终结果。
"""

# 系统提示词：指导 LLM 如何执行因果链提取
CAUSAL_CHAIN_SYSTEM_PROMPT = "你从事件序列中提取因果链。请识别根本原因、中间影响和最终结果。"

# 用户提示词模板
CAUSAL_CHAIN_USER_PROMPT = """{instruction}

事件（按时间顺序）：
{events}

提取因果链。返回 CausalChainExtraction JSON。
"""
