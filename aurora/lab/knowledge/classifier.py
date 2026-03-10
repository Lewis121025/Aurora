"""
AURORA Knowledge Type Classifier
================================

公共入口保留在本模块，分类规则与结果模型拆分到独立文件。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aurora.lab.knowledge.models import (
    ClassificationResult,
    ConflictAnalysis,
    ConflictResolution,
    KnowledgeType,
)
from aurora.lab.knowledge.patterns import (
    COMPLEMENTARY_TRAIT_PAIRS,
    CONTRADICTORY_PAIRS,
    KEYWORD_RULES,
    PATTERN_RULES,
    TRAIT_WORDS,
)
from aurora.utils.math_utils import cosine_sim

__all__ = [
    "ClassificationResult",
    "ConflictAnalysis",
    "ConflictResolution",
    "KnowledgeClassifier",
    "KnowledgeType",
    "classify_knowledge",
    "resolve_knowledge_conflict",
]


class KnowledgeClassifier:
    """基于第一原则的知识类型分类和冲突策略引擎。"""

    def __init__(self, seed: int = 0):
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self._classification_stats: Dict[str, int] = {
            knowledge_type.value: 0 for knowledge_type in KnowledgeType
        }

    def classify(
        self,
        text: str,
        embedding: Optional[np.ndarray] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ClassificationResult:
        del embedding, context

        text_lower = text.lower()
        scores = {knowledge_type: 0.0 for knowledge_type in KnowledgeType}
        matched_patterns = {knowledge_type: [] for knowledge_type in KnowledgeType}

        for knowledge_type, (score, patterns) in self._match_keywords(text_lower).items():
            scores[knowledge_type] += score
            matched_patterns[knowledge_type].extend(patterns)

        subject, predicate = None, None
        for knowledge_type, (
            score,
            patterns,
            matched_subject,
            matched_predicate,
        ) in self._match_patterns(text).items():
            scores[knowledge_type] += score
            matched_patterns[knowledge_type].extend(patterns)
            subject = subject or matched_subject
            predicate = predicate or matched_predicate

        trait_score = self._detect_trait_words(text_lower)
        if trait_score > 0:
            scores[KnowledgeType.IDENTITY_TRAIT] += trait_score
            matched_patterns[KnowledgeType.IDENTITY_TRAIT].append("trait_word_detected")

        if sum(scores.values()) < 0.1:
            return ClassificationResult(
                knowledge_type=KnowledgeType.UNKNOWN,
                confidence=0.3,
                matched_patterns=[],
                subject=subject,
                predicate=predicate,
            )

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
        confidence = min(1.0, 0.5 + (best_score - second_best) * 0.5)

        self._classification_stats[best_type.value] += 1
        return ClassificationResult(
            knowledge_type=best_type,
            confidence=confidence,
            matched_patterns=matched_patterns[best_type],
            subject=subject,
            predicate=predicate,
        )

    def _match_keywords(self, text_lower: str) -> Dict[KnowledgeType, Tuple[float, List[str]]]:
        results: Dict[KnowledgeType, Tuple[float, List[str]]] = {}
        for knowledge_type, (keywords, weight) in KEYWORD_RULES.items():
            matches = [keyword for keyword in keywords if keyword in text_lower]
            results[knowledge_type] = (len(matches) * weight, matches)
        return results

    def _match_patterns(
        self,
        text: str,
    ) -> Dict[KnowledgeType, Tuple[float, List[str], Optional[str], Optional[str]]]:
        results = {knowledge_type: (0.0, [], None, None) for knowledge_type in KnowledgeType}

        for rule in PATTERN_RULES:
            for pattern in rule.patterns:
                match = pattern.search(text)
                if not match:
                    continue

                subject = match.group(1).strip() if len(match.groups()) > 0 else None
                predicate = match.group(2).strip() if len(match.groups()) > 1 else None
                if rule.knowledge_type == KnowledgeType.IDENTITY_TRAIT and predicate is None:
                    predicate = match.group(1).strip()

                results[rule.knowledge_type] = (
                    rule.score,
                    [pattern.pattern[:30]],
                    subject,
                    predicate,
                )
                break

        return results

    def _detect_trait_words(self, text_lower: str) -> float:
        matches = [trait_word for trait_word in TRAIT_WORDS if trait_word in text_lower]
        return len(matches) * 0.5

    def resolve_conflict(
        self,
        type_a: KnowledgeType,
        type_b: KnowledgeType,
        time_relation: str,
        text_a: Optional[str] = None,
        text_b: Optional[str] = None,
        embedding_a: Optional[np.ndarray] = None,
        embedding_b: Optional[np.ndarray] = None,
    ) -> ConflictAnalysis:
        if (
            type_a == KnowledgeType.IDENTITY_TRAIT
            and type_b == KnowledgeType.IDENTITY_TRAIT
            and text_a
            and text_b
        ):
            return self._resolve_trait_conflict(
                time_relation=time_relation,
                text_a=text_a,
                text_b=text_b,
                embedding_a=embedding_a,
                embedding_b=embedding_b,
            )

        if type_a == type_b:
            return self._resolve_same_type_conflict(type_a)
        return self._resolve_different_type_conflict(type_a, type_b)

    def _resolve_trait_conflict(
        self,
        time_relation: str,
        text_a: str,
        text_b: str,
        embedding_a: Optional[np.ndarray],
        embedding_b: Optional[np.ndarray],
    ) -> ConflictAnalysis:
        if self.are_complementary_traits(text_a, text_b, embedding_a, embedding_b):
            return ConflictAnalysis(
                resolution=ConflictResolution.PRESERVE_BOTH,
                rationale="这两个特质是互补的而非矛盾的——它们在不同情境下激活不同的面向",
                confidence=0.85,
                knowledge_type_a=KnowledgeType.IDENTITY_TRAIT,
                knowledge_type_b=KnowledgeType.IDENTITY_TRAIT,
                is_complementary=True,
                recommended_actions=[
                    "保留两个特质",
                    "标记为情境依赖的特质",
                    "在检索时根据上下文激活",
                ],
            )

        return ConflictAnalysis(
            resolution=(
                ConflictResolution.CORRECT
                if time_relation == "sequential"
                else ConflictResolution.UPDATE
            ),
            rationale="这两个特质存在真正的矛盾，需要解决",
            confidence=0.75,
            knowledge_type_a=KnowledgeType.IDENTITY_TRAIT,
            knowledge_type_b=KnowledgeType.IDENTITY_TRAIT,
            is_complementary=False,
            requires_human_review=True,
            recommended_actions=[
                "标记为需要人工审核",
                "保留更新的特质",
                "记录变化历史",
            ],
        )

    def _resolve_same_type_conflict(self, knowledge_type: KnowledgeType) -> ConflictAnalysis:
        if knowledge_type == KnowledgeType.FACTUAL_STATE:
            return ConflictAnalysis(
                resolution=ConflictResolution.UPDATE,
                rationale="状态性事实会随时间变化，保留最新状态",
                confidence=0.9,
                knowledge_type_a=knowledge_type,
                knowledge_type_b=knowledge_type,
                recommended_actions=["用新状态替换旧状态", "保留旧状态作为历史记录（可选）"],
            )
        if knowledge_type == KnowledgeType.FACTUAL_STATIC:
            return ConflictAnalysis(
                resolution=ConflictResolution.CORRECT,
                rationale="静态事实不应该变化，旧的可能是错误的",
                confidence=0.85,
                knowledge_type_a=knowledge_type,
                knowledge_type_b=knowledge_type,
                requires_human_review=True,
                recommended_actions=[
                    "需要验证哪个是正确的",
                    "标记旧的为可能错误",
                    "考虑信息来源可靠性",
                ],
            )
        if knowledge_type == KnowledgeType.IDENTITY_TRAIT:
            return ConflictAnalysis(
                resolution=ConflictResolution.PRESERVE_BOTH,
                rationale="身份特质可以共存，它们是不同情境下的不同面向",
                confidence=0.7,
                knowledge_type_a=knowledge_type,
                knowledge_type_b=knowledge_type,
                is_complementary=True,
                recommended_actions=[
                    "保留两个特质",
                    "分析它们是否在不同情境下有效",
                    "标记为多面性身份",
                ],
            )
        if knowledge_type == KnowledgeType.IDENTITY_VALUE:
            return ConflictAnalysis(
                resolution=ConflictResolution.PRESERVE_BOTH,
                rationale="价值观是身份的核心，不同价值观可以共存",
                confidence=0.85,
                knowledge_type_a=knowledge_type,
                knowledge_type_b=knowledge_type,
                recommended_actions=[
                    "保留所有价值观",
                    "分析它们之间的关系",
                    "识别潜在的价值观冲突",
                ],
            )
        if knowledge_type == KnowledgeType.PREFERENCE:
            return ConflictAnalysis(
                resolution=ConflictResolution.EVOLVE,
                rationale="偏好会随时间演化，保留变化轨迹",
                confidence=0.8,
                knowledge_type_a=knowledge_type,
                knowledge_type_b=knowledge_type,
                recommended_actions=["保留两个偏好作为时间线", "标记时间戳", "分析偏好演化趋势"],
            )
        if knowledge_type == KnowledgeType.BEHAVIOR_PATTERN:
            return ConflictAnalysis(
                resolution=ConflictResolution.EVOLVE,
                rationale="行为模式会随时间变化，记录变化有助于理解",
                confidence=0.75,
                knowledge_type_a=knowledge_type,
                knowledge_type_b=knowledge_type,
                recommended_actions=["保留两个行为模式", "标记时间段", "分析行为变化原因"],
            )

        return ConflictAnalysis(
            resolution=ConflictResolution.NO_ACTION,
            rationale="无法确定知识类型，需要更多信息",
            confidence=0.3,
            knowledge_type_a=knowledge_type,
            knowledge_type_b=knowledge_type,
            requires_human_review=True,
            recommended_actions=["收集更多上下文", "人工审核"],
        )

    def _resolve_different_type_conflict(
        self,
        type_a: KnowledgeType,
        type_b: KnowledgeType,
    ) -> ConflictAnalysis:
        priority = {
            KnowledgeType.IDENTITY_VALUE: 5,
            KnowledgeType.IDENTITY_TRAIT: 4,
            KnowledgeType.FACTUAL_STATIC: 3,
            KnowledgeType.FACTUAL_STATE: 2,
            KnowledgeType.PREFERENCE: 1,
            KnowledgeType.BEHAVIOR_PATTERN: 0,
            KnowledgeType.UNKNOWN: -1,
        }
        higher_type = type_a if priority.get(type_a, -1) >= priority.get(type_b, -1) else type_b
        return ConflictAnalysis(
            resolution=ConflictResolution.PRESERVE_BOTH,
            rationale=f"不同类型的知识（{type_a.value} vs {type_b.value}）通常可以共存",
            confidence=0.65,
            knowledge_type_a=type_a,
            knowledge_type_b=type_b,
            recommended_actions=[
                "保留两条知识",
                f"优先参考{higher_type.value}类型",
                "分析它们之间的关系",
            ],
        )

    def are_complementary_traits(
        self,
        text_a: str,
        text_b: str,
        embedding_a: Optional[np.ndarray] = None,
        embedding_b: Optional[np.ndarray] = None,
    ) -> bool:
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()

        for set_a, set_b in COMPLEMENTARY_TRAIT_PAIRS:
            if self._pair_matches(text_a_lower, text_b_lower, set_a, set_b):
                return True

        if self._are_truly_contradictory(text_a, text_b):
            return False

        if embedding_a is not None and embedding_b is not None:
            similarity = cosine_sim(embedding_a, embedding_b)
            if similarity < -0.3:
                return False
            if 0.2 < similarity < 0.7:
                return True

        return True

    def _are_truly_contradictory(self, text_a: str, text_b: str) -> bool:
        text_a_lower = text_a.lower()
        text_b_lower = text_b.lower()
        return any(
            self._pair_matches(text_a_lower, text_b_lower, set_a, set_b)
            for set_a, set_b in CONTRADICTORY_PAIRS
        )

    def _pair_matches(
        self,
        text_a_lower: str,
        text_b_lower: str,
        set_a: set[str],
        set_b: set[str],
    ) -> bool:
        a_in_set_a = any(word in text_a_lower for word in set_a)
        a_in_set_b = any(word in text_a_lower for word in set_b)
        b_in_set_a = any(word in text_b_lower for word in set_a)
        b_in_set_b = any(word in text_b_lower for word in set_b)
        return (a_in_set_a and b_in_set_b) or (a_in_set_b and b_in_set_a)

    def classify_batch(
        self,
        texts: List[str],
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[ClassificationResult]:
        results: List[ClassificationResult] = []
        for index, text in enumerate(texts):
            embedding = embeddings[index] if embeddings and index < len(embeddings) else None
            results.append(self.classify(text, embedding=embedding))
        return results

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "seed": self._seed,
            "classification_stats": self._classification_stats,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "KnowledgeClassifier":
        classifier = cls(seed=state.get("seed", 0))
        classifier._classification_stats = state.get(
            "classification_stats",
            classifier._classification_stats,
        )
        return classifier

    def get_statistics(self) -> Dict[str, Any]:
        total = sum(self._classification_stats.values())
        return {
            "total_classifications": total,
            "by_type": self._classification_stats.copy(),
            "distribution": {
                key: value / total if total > 0 else 0
                for key, value in self._classification_stats.items()
            },
        }


def classify_knowledge(text: str, seed: int = 0) -> ClassificationResult:
    return KnowledgeClassifier(seed=seed).classify(text)


def resolve_knowledge_conflict(
    text_a: str,
    text_b: str,
    time_relation: str = "sequential",
    seed: int = 0,
) -> ConflictAnalysis:
    classifier = KnowledgeClassifier(seed=seed)
    type_a = classifier.classify(text_a)
    type_b = classifier.classify(text_b)
    return classifier.resolve_conflict(
        type_a.knowledge_type,
        type_b.knowledge_type,
        time_relation,
        text_a,
        text_b,
    )
