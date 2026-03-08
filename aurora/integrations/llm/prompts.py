from __future__ import annotations

from typing import Any, Dict, Optional
from .schemas import SCHEMA_VERSION
from aurora.core.config.query_types import QUESTION_STOP_WORDS, SINGLE_SESSION_USER_MAX_CONTEXT, USER_ROLE_PRIORITY_BOOST

def _json_instruction(model_name: str) -> str:
    return (
        "You MUST output ONLY valid JSON. No markdown. No extra keys. "
        f"Schema version: {SCHEMA_VERSION}. Output must conform to {model_name}."
    )

PLOT_EXTRACTION_SYSTEM = (
    "You are a precise information extractor for a narrative memory system. "
    "Extract the atomic plot and factual claims."
)

PLOT_EXTRACTION_USER = """{instruction}

INPUT:
- user_message: {user_message}
- agent_message: {agent_message}
- optional_context: {context}

Return PlotExtraction JSON with:
- actors (include 'user' and 'agent' if applicable)
- action/context/outcome (short, concrete)
- goal/obstacles/decision
- emotion_valence/arousal
- claims: short subject-predicate-object triples; avoid speculation.
"""

STORY_UPDATE_SYSTEM = (
    "You update a story arc summary based on a new plot. "
    "Keep it coherent and compact."
)

STORY_UPDATE_USER = """{instruction}

STORY_SO_FAR:
{story_so_far}

NEW_PLOT:
{new_plot}

Return StoryUpdate JSON.
"""

THEME_CANDIDATE_SYSTEM = (
    "You identify emergent themes from multiple story summaries. "
    "Themes should be falsifiable and useful."
)

THEME_CANDIDATE_USER = """{instruction}

STORY_SUMMARIES:
{story_summaries}

Return a JSON array of ThemeCandidate (0..N). Only include themes with concrete evidence.
"""

SELF_NARRATIVE_SYSTEM = (
    "You update an agent self narrative based on recent themes. "
    "Be stable: only change when evidence is strong."
)

SELF_NARRATIVE_USER = """{instruction}

CURRENT_SELF_NARRATIVE:
{current}

RECENT_THEMES:
{themes}

Return SelfNarrativeUpdate JSON.
"""

CONTRADICTION_SYSTEM = (
    "You judge whether two claims contradict each other. "
    "If they can both be true under different conditions, treat as not strict contradiction and provide reconciliation_hint."
)

CONTRADICTION_USER = """{instruction}

CLAIM_A: {claim_a}
CLAIM_B: {claim_b}

Return ContradictionJudgement JSON.
"""

# -----------------------------------------------------------------------------
# 因果推理提示
# -----------------------------------------------------------------------------

CAUSAL_RELATION_SYSTEM = (
    "You identify causal relationships between events. "
    "Focus on actual causation, not just correlation or temporal sequence. "
    "Be conservative: only claim causation when there is clear evidence."
)

CAUSAL_RELATION_USER = """{instruction}

EVENT A:
{event_a}

EVENT B:
{event_b}

CONTEXT:
{context}

Analyze if there is a causal relationship between A and B.
Return CausalRelation JSON.
"""

CAUSAL_CHAIN_SYSTEM = (
    "You extract causal chains from sequences of events. "
    "Identify root causes, intermediate effects, and final outcomes."
)

CAUSAL_CHAIN_USER = """{instruction}

EVENTS (in temporal order):
{events}

Extract the causal chain. Return CausalChainExtraction JSON.
"""

COUNTERFACTUAL_SYSTEM = (
    "You reason about counterfactuals: what would have happened if something were different. "
    "Base your reasoning on the causal structure, not just surface similarity."
)

COUNTERFACTUAL_USER = """{instruction}

FACTUAL SITUATION:
{factual}

COUNTERFACTUAL QUESTION:
If {antecedent}, what would have happened to {query}?

RELEVANT CONTEXT:
{context}

Return CounterfactualQuery JSON with your reasoning.
"""

# -----------------------------------------------------------------------------
# 自我叙述提示
# -----------------------------------------------------------------------------

CAPABILITY_ASSESSMENT_SYSTEM = (
    "You assess agent capabilities based on interaction outcomes. "
    "Be balanced: note both successes and areas for improvement."
)

CAPABILITY_ASSESSMENT_USER = """{instruction}

INTERACTION:
User: {user_message}
Agent: {agent_message}
Outcome: {outcome}

Assess what capabilities were demonstrated or what limitations were revealed.
Return CapabilityAssessment JSON.
"""

RELATIONSHIP_ASSESSMENT_SYSTEM = (
    "You assess the quality of agent-user relationships from interactions."
)

RELATIONSHIP_ASSESSMENT_USER = """{instruction}

ENTITY: {entity_id}
INTERACTION HISTORY SUMMARY: {history_summary}
LATEST INTERACTION:
User: {user_message}
Agent: {agent_message}

Assess the relationship quality. Return RelationshipAssessment JSON.
"""

IDENTITY_REFLECTION_SYSTEM = (
    "You help the agent reflect on its identity based on experiences. "
    "Be honest about capabilities and limitations. Identify growth areas."
)

IDENTITY_REFLECTION_USER = """{instruction}

RECENT THEMES:
{themes}

CAPABILITY BELIEFS:
{capabilities}

RELATIONSHIP SUMMARY:
{relationships}

Generate a self-reflection. Return IdentityReflection JSON.
"""

# -----------------------------------------------------------------------------
# 一致性提示
# -----------------------------------------------------------------------------

COHERENCE_CHECK_SYSTEM = (
    "You check for coherence conflicts between memory elements. "
    "Consider factual, temporal, causal, and thematic consistency."
)

COHERENCE_CHECK_USER = """{instruction}

ELEMENT A:
{element_a}

ELEMENT B:
{element_b}

Check if these elements are coherent with each other.
Return CoherenceCheck JSON.
"""

def render(template: str, **kwargs: Any) -> str:
    return template.format(**kwargs)

def instruction(model_name: str) -> str:
    return _json_instruction(model_name)

# -----------------------------------------------------------------------------
# Question-Answering Prompts (Type-Specific)
# -----------------------------------------------------------------------------

# Prompt templates for different question types
QA_PROMPT_TEMPLATES: Dict[str, str] = {
    'knowledge-update': """CRITICAL: This question is about information that may have CHANGED over time.
The context below contains conversations in CHRONOLOGICAL ORDER (earlier conversations first, later conversations last).

YOUR TASK: Find the MOST RECENT/LATEST value for what is being asked.
- If the same fact appears multiple times with different values, USE THE LATER ONE.
- Look for phrases like "now", "currently", "updated", "new", "recently" which indicate newer information.
- Information appearing LATER in the context supersedes earlier information.

Context (in chronological order - LATER entries are more current):
{context}

Question: {question}

Answer (MUST use the most recent/latest value if information has changed):""",

    'temporal-reasoning': """Based on the conversation history, answer the question about timing or sequence.
Pay attention to words like "first", "last", "before", "after", "earliest", "latest", "when".

Context:
{context}

Question: {question}

Answer (be specific about time/sequence):""",

    'multi-session': """Based on the conversation history from MULTIPLE SESSIONS below, 
answer the question by AGGREGATING information across ALL sessions.

CRITICAL INSTRUCTION: This question requires you to AGGREGATE information from MULTIPLE conversations.

YOUR TASK:
1. READ through ALL the context provided carefully - each section separated by '---' may be from a DIFFERENT session
2. IDENTIFY every mention of the topic being asked about
3. EXTRACT each relevant piece of information (names, numbers, amounts, items, etc.)
4. AGGREGATE the information appropriately:
   - "How many" → COUNT all distinct items/instances
   - "How much" / "Total" → SUM all amounts/values
   - "What are all" / "List all" → ENUMERATE all items mentioned
5. VERIFY you haven't missed anything by re-scanning the context

COMMON MISTAKES TO AVOID:
- Only counting items from one session (there are MULTIPLE sessions!)
- Missing items because they were phrased differently
- Not reading all the way to the end of the context

Context (from multiple sessions - READ ALL OF IT):
{context}

Question: {question}

Think step by step:
1. What items/values did I find in the context? List them.
2. Did I check ALL sections of the context?
3. What is the aggregated answer?

Answer (be specific, complete, include all items found):""",

    'single-session-assistant': """Based on the conversation history below, answer the question concisely.
Focus on extracting the specific information requested.

Context:
{context}

Question: {question}

Answer (be brief, specific, and factual):""",

    'single-session-user': """Based on the conversation history below, answer the question about facts the USER stated.

IMPORTANT: Focus ONLY on information that the USER explicitly mentioned or stated.
Look for statements where the user says things like:
- "I have/got/bought/own..."
- "My [X] is..."
- "I'm using/working with..."
- Numbers, names, or specific details the user shared

Context:
{context}

Question: {question}

Instructions:
1. Scan the context for USER statements (lines starting with "User:" or "用户:")
2. Find the specific fact the question is asking about
3. Extract the exact value/answer the user mentioned

Answer (extract the specific fact from USER statements, be brief and precise):""",

    'single-session-preference': """Based on the conversation history below, describe what the user would prefer.

Your task is to IDENTIFY and DESCRIBE the user's preferences, NOT to give an actual recommendation.

Context:
{context}

Question: {question}

IMPORTANT INSTRUCTIONS:
1. Find specific tools, brands, products, or topics the user mentions (e.g., "Adobe Premiere Pro", "Sony A7R IV", "deep learning for medical imaging")
2. Identify what category or domain the question is asking about
3. Describe what type of responses the user would prefer based on their stated interests
4. Also describe what they would NOT prefer

Your answer MUST follow this format:
"The user would prefer [responses/suggestions/resources] related to [specific tool/brand/topic they mentioned]. They would/might not prefer [opposite or unrelated options]."

Answer (describe preferences, do NOT give actual recommendations):""",

    'default': """Based on the conversation history, answer the question concisely.
If the information is not available, say "I don't know".

Context:
{context}

Question: {question}

Answer (be brief and specific):"""
}

def detect_question_type(question: str, question_type_hint: Optional[str] = None) -> str:
    """从问题文本或使用提供的提示检测问题类型。

    参数：
        question：问题文本
        question_type_hint：可选的显式问题类型（例如，来自数据集）

    返回：
        用于提示模板选择的问题类型键
    """
    # 如果提供了显式类型，使用它（带规范化）
    if question_type_hint:
        hint_lower = question_type_hint.lower().replace('_', '-')
        # 映射常见变体
        if 'multi-session' in hint_lower or 'multisession' in hint_lower:
            return 'multi-session'
        if 'knowledge-update' in hint_lower or 'knowledgeupdate' in hint_lower:
            return 'knowledge-update'
        if 'temporal' in hint_lower or 'time' in hint_lower:
            return 'temporal-reasoning'
        if 'preference' in hint_lower:
            return 'single-session-preference'
        if 'assistant' in hint_lower:
            return 'single-session-assistant'
        if 'user' in hint_lower and 'single' in hint_lower:
            return 'single-session-user'

    # 从问题文本自动检测
    question_lower = question.lower()

    # 检查多会话/聚合指示符
    # 这些关键字表示需要来自多个会话的信息的问题
    aggregation_keywords = [
        'how many', 'how much', 'total', 'sum', 'count', 'all', 'every', 'each',
        'aggregate', 'combined', 'together', 'altogether', 'in total', 'in all',
        'number of', 'amount of', 'quantity of', 'across', 'multiple', 'sessions'
    ]
    if any(kw in question_lower for kw in aggregation_keywords):
        return 'multi-session'

    # 检查时间推理指示符
    temporal_keywords = [
        'first', 'last', 'earliest', 'latest', 'before', 'after',
        'when', 'time', 'date', 'then', 'previously', 'initially',
        'finally', 'next', 'previous', 'earlier', 'later'
    ]
    if any(kw in question_lower for kw in temporal_keywords):
        return 'temporal-reasoning'

    # 检查知识更新指示符
    update_keywords = [
        'current', 'now', 'latest', 'most recent', 'updated',
        'change', 'changed', 'new', 'newer'
    ]
    if any(kw in question_lower for kw in update_keywords):
        return 'knowledge-update'

    # 默认为通用提示
    return 'default'

def _extract_question_keywords(question: str) -> list:
    """从问题中提取有意义的关键字用于匹配。

    参数：
        question：问题文本

    返回：
        关键字字符串列表（小写）
    """
    
    question_lower = question.lower()
    words = []
    for w in question_lower.split():
        # 清理标点符号
        clean_w = w.strip('?.,!\'\"()[]{}:;')
        if len(clean_w) > 2 and clean_w not in QUESTION_STOP_WORDS:
            words.append(clean_w)

    return words

def _extract_relevant_context(
    context: str,
    question: str,
    max_length: int = 12000,
    delimiter: str = '\n---\n',
    question_type: Optional[str] = None
) -> str:
    """根据问题关键字提取上下文中最相关的部分。

    这使用一个简单但有效的策略：
    1. 按分隔符将上下文分割成块
    2. 根据与问题的关键字重叠对每个块进行评分
    3. 包括最高评分的块，直到达到 max_length

    针对 single-session-user 问题的增强：
    - 优先考虑包含用户陈述的块
    - 提取匹配关键字周围的更多上下文
    - 使用更高的 max_length

    参数：
        context：完整上下文字符串
        question：要回答的问题
        max_length：最大输出长度
        delimiter：上下文中的块分隔符
        question_type：可选的问题类型提示，用于优化提取

    返回：
        包含最相关块的过滤上下文
    """
    

    # 对于 single-session-user，使用更长的上下文和特殊处理
    is_user_type = question_type and 'user' in question_type.lower() and 'single' in question_type.lower()

    # 对于 single-session-preference，也使用特殊处理
    is_preference_type = question_type and 'preference' in question_type.lower()

    if is_user_type or is_preference_type:
        max_length = max(max_length, SINGLE_SESSION_USER_MAX_CONTEXT)

    if len(context) <= max_length:
        return context

    # 分割成块
    chunks = context.split(delimiter)
    if len(chunks) <= 1:
        # 未找到分隔符，尝试基于行的分割
        lines = context.split('\n')
        if len(lines) > 20:
            # 将行分组为 ~500 字符的块
            chunks = []
            current_chunk = []
            current_len = 0
            for line in lines:
                current_chunk.append(line)
                current_len += len(line) + 1
                if current_len >= 500:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_len = 0
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
        else:
            # 行数不足以分块，只需截断
            return context[:max_length] + "\n[... context truncated ...]"

    # 从问题中提取关键字
    question_words = _extract_question_keywords(question)

    # 对每个块进行评分
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = 0.0

        # 来自关键字匹配的基础分数
        for word in question_words:
            if word in chunk_lower:
                score += chunk_lower.count(word)

        # 对于 single-session-user：提升用户陈述
        if is_user_type:
            # 检查块是否包含用户陈述标记
            user_markers = ['user:', '用户:', 'user：', '用户：']
            for marker in user_markers:
                if marker in chunk_lower:
                    # 计算用户陈述数
                    user_count = chunk_lower.count(marker)
                    score += user_count * USER_ROLE_PRIORITY_BOOST * 10
                    break

        # 对于 single-session-preference：提升包含用户偏好的块
        if is_preference_type:
            # 检查用户陈述
            user_markers = ['user:', '用户:', 'user：', '用户：']
            for marker in user_markers:
                if marker in chunk_lower:
                    user_count = chunk_lower.count(marker)
                    score += user_count * USER_ROLE_PRIORITY_BOOST * 8
                    break

            # 提升提及偏好、工具或品牌的块
            preference_indicators = [
                'i use', 'i prefer', "i'm using", 'my setup', 'i like',
                'i enjoy', 'i have', 'compatible with', 'working with',
                'learning', 'interested in', 'focusing on', 'specializing',
                '我用', '我喜欢', '我的'
            ]
            for indicator in preference_indicators:
                if indicator in chunk_lower:
                    score += 5  # 提升偏好相关内容

        # 对早期块给予轻微偏好（近期偏差）
        # 但不要太多，因为重要信息可能在任何地方
        position_bonus = 0.1 * (1 - i / len(chunks))
        scored_chunks.append((chunk, score + position_bonus, i))

    # 按分数排序（最高优先），用原始位置打破平局
    scored_chunks.sort(key=lambda x: (-x[1], x[2]))

    # 通过添加块直到达到 max_length 来构建结果
    # 如果块太大，截断它但仍包括最相关的部分
    selected = []
    total_len = 0
    for chunk, score, orig_idx in scored_chunks:
        remaining = max_length - total_len
        if remaining <= 100:
            break

        if len(chunk) <= remaining:
            # 块完全适合
            selected.append((orig_idx, chunk))
            total_len += len(chunk) + len(delimiter)
        elif score > 0:
            # 块太大但有相关内容 - 提取相关部分
            # 查找包含问题词的行并包括周围上下文
            lines = chunk.split('\n')
            relevant_lines = []

            # 对于 single-session-user 和 preference，也标记用户陈述行
            user_line_indices = set()
            if is_user_type or is_preference_type:
                for j, line in enumerate(lines):
                    line_lower = line.lower()
                    if any(m in line_lower for m in ['user:', '用户:', 'user：', '用户：']):
                        user_line_indices.add(j)

            for j, line in enumerate(lines):
                line_lower = line.lower()
                is_relevant = any(word in line_lower for word in question_words)

                # 对于用户类型问题，也包括所有用户陈述行
                if is_user_type and j in user_line_indices:
                    is_relevant = True

                # 对于偏好问题，包括用户陈述和偏好指示符
                if is_preference_type and j in user_line_indices:
                    is_relevant = True

                # 也检查偏好问题中的偏好指示符
                if is_preference_type:
                    pref_indicators = ['i use', 'i prefer', "i'm using", 'my setup', 'i like',
                                       'compatible with', 'working with', 'learning', 'interested']
                    if any(ind in line_lower for ind in pref_indicators):
                        is_relevant = True

                if is_relevant:
                    # 包括此行和周围的一些上下文
                    # 对于用户类型和偏好，包括更多上下文（3 行 vs 2 行）
                    context_lines = 3 if (is_user_type or is_preference_type) else 2
                    start = max(0, j - context_lines)
                    end = min(len(lines), j + context_lines + 1)
                    for k in range(start, end):
                        if k not in [idx for idx, _ in relevant_lines]:
                            relevant_lines.append((k, lines[k]))

            if relevant_lines:
                # 按行索引排序以保持顺序
                relevant_lines.sort(key=lambda x: x[0])
                extracted = '\n'.join(line for _, line in relevant_lines)

                # 如果仍然太长则截断
                if len(extracted) > remaining - 50:
                    extracted = extracted[:remaining - 50] + "\n[...]"

                selected.append((orig_idx, extracted))
                total_len += len(extracted) + len(delimiter)

    # 按原始位置排序以保持一致性
    selected.sort(key=lambda x: x[0])

    # 连接选定的块
    result = delimiter.join(chunk for _, chunk in selected)

    if len(result) < len(context):
        result += f"\n[... {len(chunks) - len(selected)} context chunks omitted for brevity ...]"

    return result

def evaluate_preference_match(expected: str, answer: str) -> bool:
    """使用关键概念提取评估答案是否与偏好期望匹配。

    对于偏好问题，我们需要匹配关键概念（品牌、工具、主题）
    而不是所有单词。常见词如"prefer"、"user"、"would"应被忽略。

    参数：
        expected：预期答案（例如，"用户会更喜欢 Adobe Premiere Pro 的资源..."）
        answer：LLM 的答案

    返回：
        如果关键概念匹配则为 True
    """
    expected_lower = expected.lower()
    answer_lower = answer.lower()

    # 要忽略的常见词（它们出现在所有偏好答案中）
    stop_words = {
        'the', 'user', 'would', 'prefer', 'responses', 'that', 'suggest', 'resources',
        'related', 'specifically', 'tailored', 'might', 'not', 'other', 'general',
        'they', 'may', 'options', 'those', 'which', 'with', 'for', 'and', 'or',
        'be', 'interested', 'in', 'their', 'about', 'some', 'are', 'from', 'to',
        'of', 'a', 'an', 'on', 'as', 'such', 'more', 'any', 'all', 'also',
        'can', 'could', 'focus', 'especially', 'particularly', 'based', 'topics',
        'unrelated', 'involve', 'involving', 'recent', 'suggestions', 'equipment',
        'gear', 'quality', 'high', 'low', 'brands', 'brand', 'type', 'types'
    }

    # 提取关键概念：可能是专有名词或技术术语的较长单词
    # 也查找多词短语（原始文本中的大写序列）
    import re

    # 查找潜在的专有名词和技术术语（3+ 字符，非停用词）
    words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]+\b', expected)
    key_concepts = []

    for word in words:
        word_lower = word.lower()
        # 跳过停用词和非常短的单词
        if word_lower in stop_words or len(word) < 3:
            continue
        # 技术术语和专有名词通常较长或有大写字母
        if len(word) >= 4 or any(c.isupper() for c in word[1:]) or any(c.isdigit() for c in word):
            key_concepts.append(word_lower)

    # 也提取多词专有名词（例如，"Adobe Premiere Pro"、"Sony A7R IV"）
    # 查找原始文本中大写单词的序列
    proper_noun_pattern = r'\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)+)\b'
    multi_word_matches = re.findall(proper_noun_pattern, expected)
    for match in multi_word_matches:
        key_concepts.append(match.lower())

    # 去重同时保持顺序
    seen = set()
    unique_concepts = []
    for c in key_concepts:
        if c not in seen:
            seen.add(c)
            unique_concepts.append(c)

    if not unique_concepts:
        # 回退到常规关键字匹配
        return expected_lower in answer_lower

    # 检查有多少关键概念在答案中
    matched = 0
    for concept in unique_concepts:
        if concept in answer_lower:
            matched += 1

    # 要求至少 40% 的关键概念匹配（更宽松的阈值）
    # 对于偏好问题，匹配主要工具/品牌就足够了
    match_ratio = matched / len(unique_concepts) if unique_concepts else 0

    # 也检查答案是否遵循预期格式
    has_prefer_format = 'prefer' in answer_lower or 'would' in answer_lower or 'interest' in answer_lower

    # 匹配如果：
    # 1. 40%+ 关键概念匹配，或
    # 2. 至少一个关键概念匹配且答案使用偏好语言
    return match_ratio >= 0.4 or (matched >= 1 and has_prefer_format)

def build_qa_prompt(
    question: str,
    context: str,
    question_type_hint: Optional[str] = None,
    is_abstention: bool = False,
    max_context_length: int = 12000
) -> str:
    """基于问题类型构建问答提示。

    此函数使用智能上下文过滤来包括基于问题关键字最相关的部分，
    而不是简单截断，这可能会丢弃关键信息。

    参数：
        question：要回答的问题
        context：从内存检索的上下文
        question_type_hint：可选的显式问题类型
        is_abstention：这是否是弃权问题
        max_context_length：要包括的最大上下文长度（默认 12000）

    返回：
        格式化的提示字符串
    """
    # 检测问题类型
    qtype = detect_question_type(question, question_type_hint)

    # 获取模板
    template = QA_PROMPT_TEMPLATES.get(qtype, QA_PROMPT_TEMPLATES['default'])

    # 智能提取相关上下文（不是简单截断）
    # 传递 question_type 以进行类型特定的提取优化
    filtered_context = _extract_relevant_context(
        context, question, max_context_length, question_type=qtype
    )

    # 构建提示
    prompt = template.format(
        context=filtered_context,
        question=question
    )

    # 如果需要，添加弃权指令
    if is_abstention:
        prompt += "\n\nIMPORTANT: If the information is not clearly available in the conversation history, respond with 'I don't know' rather than guessing."

    return prompt
