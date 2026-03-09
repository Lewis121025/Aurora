MEMORY_BRIEF_COMPILATION_SYSTEM_PROMPT = (
    "You compile structured memory evidence into a clean memory brief for a dialogue model. "
    "Use only supported information. Prefer concise, current, directly answerable memory statements. "
    "Never quote raw dialogue or include USER:/AGENT: turn text."
)

MEMORY_BRIEF_COMPILATION_USER_PROMPT = """{instruction}

CURRENT_USER_MESSAGE:
{user_message}

QUERY_TYPE:
{query_type}

ABSTENTION_REASON:
{abstention_reason}

CANDIDATE_MEMORY:
{candidate_memory}

EVIDENCE_SUMMARIES:
{evidence_summaries}

Return MemoryBriefCompilation JSON with:
- known_facts
- preferences
- relationship_state
- active_narratives
- temporal_context
- cautions

Rules:
- Only use information supported by CANDIDATE_MEMORY or EVIDENCE_SUMMARIES.
- Never quote raw dialogue or include markers like USER: / AGENT:.
- Prefer current state over historical state unless the query is temporal.
- Keep each item short, factual, and directly useful for answering.
- If evidence is weak or missing, put that into cautions instead of guessing.
"""
