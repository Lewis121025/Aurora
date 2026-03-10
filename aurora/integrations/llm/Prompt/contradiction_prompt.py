"""
aurora/integrations/llm/Prompt/contradiction_prompt.py
矛盾判定提示词模块。
用于判断两个事实主张之间是否存在矛盾，如果两者在不同条件下均可成立，则提供调和提示。
"""

# 系统提示词：指导 LLM 如何执行矛盾判定
CONTRADICTION_SYSTEM_PROMPT = (
    "你判断两个主张是否相互矛盾。如果它们在不同条件下都可以成立，请视为非严格矛盾，并提供调和提示（reconciliation_hint）。"
)

# 用户提示词模板
CONTRADICTION_USER_PROMPT = """{instruction}

主张 A：{claim_a}
主张 B：{claim_b}

返回 ContradictionJudgement JSON。
"""
