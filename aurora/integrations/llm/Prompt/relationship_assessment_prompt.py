RELATIONSHIP_ASSESSMENT_SYSTEM_PROMPT = (
    "You assess the quality of agent-user relationships from interactions."
)

RELATIONSHIP_ASSESSMENT_USER_PROMPT = """{instruction}

ENTITY: {entity_id}
INTERACTION HISTORY SUMMARY: {history_summary}
LATEST INTERACTION:
User: {user_message}
Agent: {agent_message}

Assess the relationship quality. Return RelationshipAssessment JSON.
"""
