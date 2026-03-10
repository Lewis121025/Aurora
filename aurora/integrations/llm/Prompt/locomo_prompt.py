"""
aurora/integrations/llm/Prompt/locomo_prompt.py
Locomo 评测提示词模块。
用于评估问答准确性和事件摘要质量，主要用于系统的基准测试（Benchmarking）。
"""

# Locomo 问答评估系统提示词
LOCOMO_QA_EVALUATION_SYSTEM_PROMPT = "你正在评估问答的准确性。"

# Locomo 问答评估用户提示词模板
LOCOMO_QA_EVALUATION_USER_PROMPT = """你正在为一个对话记忆系统评估问答任务。

对话上下文已被摄入记忆。基于检索到的信息，系统生成了一个问题的答案。

问题：{question}
标准答案：{ground_truth}
系统答案：{prediction}

评估系统的回答是否正确。请考虑：
1. 答案是否包含标准答案中的关键信息？
2. 答案在事实是否与标准答案一致？
3. 如果意思保持不变，细微的措辞差异是可以接受的。

请回复你的评估结果。"""

# Locomo 摘要评估系统提示词
LOCOMO_SUMMARIZATION_EVALUATION_SYSTEM_PROMPT = "你正在评估事件摘要的质量。"

# Locomo 摘要评估用户提示词模板
LOCOMO_SUMMARIZATION_EVALUATION_USER_PROMPT = """你正在为一个对话记忆系统评估事件摘要任务。

系统根据对话生成了事件摘要。

标准摘要：{ground_truth}
系统摘要：{prediction}

从以下维度评估摘要：
1. 连贯性：摘要是否组织良好且易读？
2. 覆盖范围：它是否涵盖了标准摘要中的关键事件？
3. 准确性：事实是否正确？

列出标准摘要中任何在系统摘要中缺失的关键事件。

请回复你的评估结果。"""
