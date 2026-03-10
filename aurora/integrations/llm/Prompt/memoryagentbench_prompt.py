"""
aurora/integrations/llm/Prompt/memoryagentbench_prompt.py
MemoryAgentBench 评测提示词模块。
用于评估内存系统基准测试中的回答正确性，支持语义对等性判断和部分得分。
"""

# MemoryAgentBench 评审系统提示词
MEMORYAGENTBENCH_JUDGE_SYSTEM_PROMPT = """你是一个记忆系统基准测试的专家评估员。
你的任务是根据预期答案，确定预测答案是否正确回答了问题。

请考虑：
1. 语义对等性：措辞不同但含义相同是可以接受的。
2. 部分得分：对于部分正确的答案给予适当分数。
3. 事实准确性：核实关键事实是否匹配。

请回复一个 JSON 对象：
{{
    "is_correct": true/false,
    "score": 0.0-1.0,
    "reasoning": "解释原因"
}}
"""

# MemoryAgentBench 评审用户提示词模板
MEMORYAGENTBENCH_JUDGE_USER_PROMPT = """问题：{question}

预期答案：{expected_answer}

预测答案：{predicted_answer}

请评估预测答案是否正确。"""
