"""语义亲和度模块。

通过 LLM 评估片段间的语义亲和度，用于 sleep 阶段的区域构建。
结果按 sleep 周期缓存，避免重复调用。
当 LLM 不可用时静默回退到 0.0（纯结构启发式继续工作）。
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aurora.llm.provider import LLMProvider
    from aurora.memory.fragment import Fragment

logger = logging.getLogger(__name__)

_BATCH_LIMIT = 12

_SYSTEM_PROMPT = (
    "You are a semantic similarity scorer. Given fragment pairs, "
    "rate how deeply they belong to the same narrative thread on a 0.0–1.0 scale.\n"
    "Respond with JSON only: [{\"pair\": [0, 1], \"score\": 0.7}, ...]\n"
    "Score meaning:\n"
    "- 0.0: unrelated\n"
    "- 0.3: loosely related by topic\n"
    "- 0.6: share emotional or narrative continuity\n"
    "- 0.9: deeply intertwined in the same lived thread\n"
)


class SemanticScorer:
    """语义亲和度评分器。

    封装 LLM 调用，批量评估片段对的语义亲和度，
    结果缓存在实例内（per sleep cycle）。

    Attributes:
        _llm: LLM 提供者。
        _cache: 缓存（片段 ID 对 -> 亲和度评分）。
    """

    __slots__ = ("_llm", "_cache")

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self._cache: dict[tuple[str, str], float] = {}

    def score_pairs(self, fragments: list[Fragment]) -> None:
        """批量评估片段对的语义亲和度并缓存。

        Args:
            fragments: 待评估的片段列表（取前 _BATCH_LIMIT 个）。
        """
        batch = fragments[:_BATCH_LIMIT]
        if len(batch) < 2:
            return

        pairs_desc: list[dict[str, str]] = []
        pair_keys: list[tuple[str, str]] = []
        for i in range(len(batch)):
            for j in range(i + 1, len(batch)):
                key = _cache_key(batch[i].fragment_id, batch[j].fragment_id)
                if key in self._cache:
                    continue
                pairs_desc.append({
                    "pair": f"{i},{j}",
                    "a": batch[i].surface[:80],
                    "b": batch[j].surface[:80],
                })
                pair_keys.append(key)

        if not pairs_desc:
            return

        user_content = json.dumps(pairs_desc, ensure_ascii=False)
        try:
            raw = self._llm.complete([
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ])
            results = _parse_scores(raw, len(pair_keys))
        except Exception:
            logger.debug("semantic scoring LLM call failed; falling back to structural heuristic")
            return

        for idx, key in enumerate(pair_keys):
            self._cache[key] = results[idx] if idx < len(results) else 0.0

    def get(self, left_id: str, right_id: str) -> float:
        """查询缓存的语义亲和度。

        Args:
            left_id: 左侧片段 ID。
            right_id: 右侧片段 ID。

        Returns:
            语义亲和度（0.0–1.0），未缓存时返回 0.0。
        """
        return self._cache.get(_cache_key(left_id, right_id), 0.0)


def _cache_key(left_id: str, right_id: str) -> tuple[str, str]:
    """生成有序缓存键。"""
    return (min(left_id, right_id), max(left_id, right_id))


def _parse_scores(raw: str, expected: int) -> list[float]:
    """解析 LLM 返回的评分列表。"""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    scores: list[float] = []
    for item in data[:expected]:
        if isinstance(item, dict):
            scores.append(max(0.0, min(1.0, float(item.get("score", 0.0)))))
        elif isinstance(item, (int, float)):
            scores.append(max(0.0, min(1.0, float(item))))
    return scores
