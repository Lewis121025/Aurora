"""
aurora/integrations/llm/Prompt/relationship_assessment_prompt.py
关系评估提示词模块。
用于从交互中评估代理与用户之间关系的质量和信任度。
"""

# 系统提示词：指导 LLM 如何执行关系评估
RELATIONSHIP_ASSESSMENT_SYSTEM_PROMPT = (
    "你根据交互评估代理与用户之间关系的质量。"
)

# 用户提示词模板
RELATIONSHIP_ASSESSMENT_USER_PROMPT = """{instruction}

实体 ID: {entity_id}
交互历史摘要: {history_summary}
最新交互：
用户：{user_message}
代理：{agent_message}

评估关系质量。返回 RelationshipAssessment JSON。
"""
