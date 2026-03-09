CAPABILITY_ASSESSMENT_SYSTEM_PROMPT = (
    "You assess agent capabilities based on interaction outcomes. "
    "Be balanced: note both successes and areas for improvement."
)

CAPABILITY_ASSESSMENT_USER_PROMPT = """{instruction}

INTERACTION:
User: {user_message}
Agent: {agent_message}
Outcome: {outcome}

Assess what capabilities were demonstrated or what limitations were revealed.
Return CapabilityAssessment JSON.
"""
