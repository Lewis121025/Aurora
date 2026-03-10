"""
aurora/integrations/llm/Prompt/meaning_extraction_prompt.py
语义/意义提取提示词模块。
用于将一段原始对话文本映射为多维度的心理学指标（V3 版本的固定维度模型）。
"""

# 系统提示词：指导 LLM 如何执行心理语义提取
MEANING_EXTRACTION_SYSTEM_PROMPT = """你在为 Aurora Soul 提取一个 EventFrame。

要求：
1. 只输出符合 schema 的 JSON。
2. `trait_evidence` 和 `belief_evidence` 的值都必须在 [-1, 1]。
3. `valence` 在 [-1, 1]，`arousal` 在 [0, 1]。
4. `tags` 只放简短标签，不要解释。
5. 不要编造文本中没有体现的高强度心理结论。
"""

# 用户提示词模板
MEANING_EXTRACTION_USER_PROMPT_TEMPLATE = """请把下面这段互动映射成 EventFrame JSON。

关注点：
- trait_evidence: attachment autonomy trust vigilance openness defensiveness assertiveness coherence
- belief_evidence: closeness_safe others_reliable boundaries_allowed independence_safe vulnerability_safe
- affect and force fields: valence arousal threat care control abandonment agency shame

Text:
{text}
"""


def build_meaning_extraction_user_prompt(*, text: str) -> str:
    """
    组装语义提取的用户提示词。
    """
    return MEANING_EXTRACTION_USER_PROMPT_TEMPLATE.format(text=text)
