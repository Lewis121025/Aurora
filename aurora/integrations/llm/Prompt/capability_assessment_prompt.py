"""
aurora/integrations/llm/Prompt/capability_assessment_prompt.py
能力评估提示词模块。
用于基于交互结果评估 Agent 的能力表现，包括成功点和待改进点。
"""

# 系统提示词：指导 LLM 如何执行能力评估
CAPABILITY_ASSESSMENT_SYSTEM_PROMPT = (
    "你根据交互结果评估代理的能力。请保持平衡：记录成功之处以及需要改进的领域。"
)

# 用户提示词模板
CAPABILITY_ASSESSMENT_USER_PROMPT = """{instruction}

交互过程：
用户：{user_message}
代理：{agent_message}
结果：{outcome}

评估展示了哪些能力或揭示了哪些局限性。
返回 CapabilityAssessment JSON。
"""
