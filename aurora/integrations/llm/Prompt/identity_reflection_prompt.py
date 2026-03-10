"""
aurora/integrations/llm/Prompt/identity_reflection_prompt.py
身份反思提示词模块。
用于帮助 Agent 基于经历进行自我身份反思，诚实评估能力与局限，识别成长领域。
"""

# 系统提示词：指导 LLM 如何执行身份反思
IDENTITY_REFLECTION_SYSTEM_PROMPT = (
    "你帮助代理根据经历反思其身份。请诚实对待能力和局限性。识别成长领域。"
)

# 用户提示词模板
IDENTITY_REFLECTION_USER_PROMPT = """{instruction}

近期主题：
{themes}

能力信念：
{capabilities}

关系摘要：
{relationships}

生成自我反思。返回 IdentityReflection JSON。
"""
