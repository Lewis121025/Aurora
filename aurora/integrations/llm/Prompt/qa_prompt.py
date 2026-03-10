"""
aurora/integrations/llm/Prompt/qa_prompt.py
问答（QA）提示词模板模块。
包含了针对不同查询类型（知识更新、时间推理、多会话聚合、偏好分析等）的专用 Prompt 模板。
旨在通过精细化的指令引导 LLM 从检索到的上下文（Context）中准确提取答案。
"""

from __future__ import annotations

from typing import Dict, Optional

from aurora.soul.query import QueryAnalyzer, QueryType, SINGLE_SESSION_USER_MAX_CONTEXT


# 定义针对不同场景的 QA 提示词模板
QA_PROMPT_TEMPLATES: Dict[str, str] = {
    # 场景：事实随时间发生变化，需要寻找“最新”状态
    "knowledge-update": """关键要求：此问题涉及可能随时间改变的信息。
下面的上下文按时间顺序排列（先发生的对话在前，后发生的在后）。

你的任务：找到所问内容的“最新/最近”状态或数值。
- 如果同一个事实出现了多次且数值不同，请以出现在后面的为准。
- 留意诸如“现在”、“目前”、“更新了”、“新”、“最近”等暗示信息更新的词汇。
- 出现在上下文后面的信息具有更高的优先级，会覆盖前面的信息。

上下文（按时间顺序排列，越靠后越新）：
{context}

问题：{question}

答案（如果信息有变动，必须使用最新的数值）：""",
    # 场景：涉及时间顺序、先后关系的推理
    "temporal-reasoning": """基于以下对话历史，回答关于时间点或先后顺序的问题。
注意“第一次”、“最后”、“之前”、“之后”、“最早”、“最近”、“当时”等关键词。

上下文：
{context}

问题：{question}

答案（请明确说明时间或顺序）：""",
    # 场景：跨越多个会话（Session）的信息聚合与统计
    "multi-session": """基于下面来自多个会话的对话历史，通过聚合所有会话的信息来回答问题。

关键指令：此问题要求你从多次对话中汇总信息。

你的任务：
1. 仔细阅读提供的所有上下文——每个由 '---' 分隔的部分可能来自不同的会话。
2. 识别出所有提及所问话题的地方。
3. 提取每一条相关信息（名称、数字、金额、物品等）。
4. 进行适当的聚合处理：
   - “多少个/几次” -> 统计所有不同实例的总数。
   - “总共/合计” -> 累加所有数值。
   - “有哪些/列出所有” -> 枚举所有提到的项。
5. 重新扫描上下文，确保没有遗漏。

要避免的常见错误：
- 只统计了一个会话的内容（注意：这里有多个会话！）。
- 遗漏了措辞不同但含义相同的项。
- 没有读到上下文的最末尾。

上下文（来自多个会话——请完整阅读）：
{context}

问题：{question}

请按以下步骤思考：
1. 我在上下文中找到了哪些项/数值？请列出它们。
2. 我是否检查了上下文的所有部分？
3. 聚合后的最终答案是什么？

答案（请具体、完整，包含找到的所有项）：""",
    # 场景：单次会话中的通用助手信息提取
    "single-session-assistant": """基于以下对话历史，简洁地回答问题。
侧重于提取所请求的具体信息。

上下文：
{context}

问题：{question}

答案（请保持简短、具体且符合事实）：""",
    # 场景：单次会话中关于“用户陈述的事实”提取
    "single-session-user": """基于以下对话历史，回答关于用户（USER）所陈述事实的问题。

重要提示：只关注用户明确提到或陈述的信息。
寻找类似以下表达：
- “我有/我买了/我拥有...”
- “我的 [X] 是...”
- “我正在使用/处理...”
- 用户分享的数字、名称或具体细节。

上下文：
{context}

问题：{question}

操作指南：
1. 在上下文中扫描用户陈述（以 "User:" 或 "用户:" 开头的行）。
2. 找到问题所问的具体事实。
3. 提取用户提到的准确数值或答案。

答案（仅从用户的陈述中提取事实，请保持简练准确）：""",
    # 场景：识别和描述用户的偏好
    "single-session-preference": """基于以下对话历史，描述用户的偏好。

你的任务是识别并描述用户的喜好，而不是给出实际的推荐。

上下文：
{context}

问题：{question}

答案（描述用户的偏好模式）：""",
}

QUESTION_TYPE_ALIASES: dict[str, str] = {
    "knowledge-update": "knowledge-update",
    "knowledge_update": "knowledge-update",
    "update": "knowledge-update",
    "temporal": "temporal-reasoning",
    "temporal-reasoning": "temporal-reasoning",
    "temporal_reasoning": "temporal-reasoning",
    "timeline": "temporal-reasoning",
    "multi": "multi-session",
    "multi-hop": "multi-session",
    "multi_hop": "multi-session",
    "multi-session": "multi-session",
    "multi_session": "multi-session",
    "single-session-assistant": "single-session-assistant",
    "single_session_assistant": "single-session-assistant",
    "assistant": "single-session-assistant",
    "single-session-user": "single-session-user",
    "single_session_user": "single-session-user",
    "user": "single-session-user",
    "user-fact": "single-session-user",
    "user_fact": "single-session-user",
    "single-session-preference": "single-session-preference",
    "single_session_preference": "single-session-preference",
    "preference": "single-session-preference",
}

PREFERENCE_KEYWORDS = {
    "喜欢",
    "偏好",
    "爱吃",
    "最爱",
    "热衷",
    "preference",
    "prefer",
    "favorite",
    "favourite",
    "likes",
    "like to",
}

KNOWLEDGE_UPDATE_KEYWORDS = {
    "现在",
    "目前",
    "最新",
    "最近",
    "更新",
    "changed",
    "current",
    "currently",
    "latest",
    "most recent",
    "updated",
    "now",
}

_QUERY_ANALYZER = QueryAnalyzer()


def _normalize_question_type_hint(question_type_hint: Optional[str]) -> Optional[str]:
    if question_type_hint is None:
        return None

    normalized = question_type_hint.strip().lower()
    return QUESTION_TYPE_ALIASES.get(
        normalized, normalized if normalized in QA_PROMPT_TEMPLATES else None
    )


def _truncate_context(context: str, *, max_context_length: int, question_type: str) -> str:
    if max_context_length <= 0 or len(context) <= max_context_length:
        return context

    omitted_marker = "\n\n...[context truncated]...\n\n"
    remaining = max_context_length - len(omitted_marker)
    if remaining <= 0:
        return context[-max_context_length:]

    if question_type in {"knowledge-update", "temporal-reasoning"}:
        return omitted_marker.strip() + context[-remaining:]

    head_len = remaining // 2
    tail_len = remaining - head_len
    return f"{context[:head_len]}{omitted_marker}{context[-tail_len:]}"


def detect_question_type(question: str) -> str:
    """根据问题内容选择最合适的 QA 提示词模板。"""
    question_lower = question.lower()

    if any(keyword in question_lower for keyword in KNOWLEDGE_UPDATE_KEYWORDS):
        return "knowledge-update"
    if any(keyword in question_lower for keyword in PREFERENCE_KEYWORDS):
        return "single-session-preference"

    query_type = _QUERY_ANALYZER.classify(question)
    if query_type is QueryType.TEMPORAL:
        return "temporal-reasoning"
    if query_type is QueryType.MULTI_HOP:
        return "multi-session"
    if query_type is QueryType.USER_FACT:
        return "single-session-user"
    return "single-session-assistant"


def build_qa_prompt(
    *,
    question: str,
    context: str,
    question_type_hint: Optional[str] = None,
    max_context_length: int = SINGLE_SESSION_USER_MAX_CONTEXT,
) -> str:
    """构建适用于问答抽取的最终 prompt。"""
    question_type = _normalize_question_type_hint(question_type_hint) or detect_question_type(
        question
    )
    template = QA_PROMPT_TEMPLATES.get(
        question_type, QA_PROMPT_TEMPLATES["single-session-assistant"]
    )
    rendered_context = _truncate_context(
        context,
        max_context_length=max_context_length,
        question_type=question_type,
    )
    return template.format(context=rendered_context, question=question)
