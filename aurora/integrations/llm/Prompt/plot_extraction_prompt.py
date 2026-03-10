"""
aurora/integrations/llm/Prompt/plot_extraction_prompt.py
情节提取提示词模块。
用于从单次交互中提取结构化的情节记录（Plot Record），包括动作、背景、结果、目标和决策。
"""

# 系统提示词：指导 LLM 如何执行情节提取
PLOT_EXTRACTION_SYSTEM_PROMPT = """你从单次交互中提取结构化的情节记录。

规则：
- 严禁将多个字段合并为一个键。
- 必须精确使用请求的 Schema 键名。
- 保持内容简短、具体，并植根于原始文本。
"""

# 用户提示词模板
PLOT_EXTRACTION_USER_PROMPT = """请返回一个 JSON 对象，必须包含以下键：
- action: 简短具体的动作描述字符串
- context: 简短具体的背景/语境描述字符串
- outcome: 简短具体的结果描述字符串
- goal: 简短的目标描述字符串
- decision: 简短的决策描述字符串

不要发明像 `action/context/outcome` 这样的组合键。
"""
