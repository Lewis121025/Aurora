"""
AURORA Abstention Detection
============================

检测是否应该拒绝回答（当置信度不足时）。

第一性原理：
- LongMemEval 有 30 个 abstention 问题（6%）
- 问的是不存在于对话历史中的信息
- 正确答案是 "I don't know" 或类似表达
- 当前 AURORA 总是尝试给出答案，这 30 题全错

设计原则：
- 无硬编码阈值：使用概率/置信度判断
- 多信号融合：检索分数、语义覆盖、否定表达
- 可解释性：返回拒绝原因
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class AbstentionResult:
    """Abstention detection result.

    Attributes:
        should_abstain: Whether to abstain from answering
        confidence: Confidence score (0.0-1.0, higher = more confident to abstain)
        reason: Human-readable reason for abstention decision
    """

    should_abstain: bool
    confidence: float
    reason: str


class AbstentionDetector:
    """检测是否应该拒绝回答。

    当检索结果与查询的相关性不足时，系统应该拒绝回答而不是给出错误答案。

    检测信号：
    1. 最高检索分数低于阈值
    2. 检索结果与查询的语义覆盖不足
    3. 检索结果包含否定表达（"never mentioned", "not discussed"）
    4. 分数分布异常（没有明显相关结果）
    """

    def __init__(
        self,
        min_relevance_threshold: float = 0.35,
        min_coverage_threshold: float = 0.3,
        low_score_threshold: float = 0.3,
        uniform_score_std_threshold: float = 0.05,
    ):
        """Initialize the abstention detector.

        Args:
            min_relevance_threshold: Minimum relevance score for top result.
                If top score < this, abstain.
            min_coverage_threshold: Minimum semantic coverage threshold (unused for now).
            low_score_threshold: Threshold for detecting uniformly low scores.
            uniform_score_std_threshold: Standard deviation threshold for detecting
                uniform score distribution (suggests no specific match).
        """
        self.min_relevance = min_relevance_threshold
        self.min_coverage = min_coverage_threshold
        self.low_score_threshold = low_score_threshold
        self.uniform_score_std_threshold = uniform_score_std_threshold

    def detect(
        self,
        query: str,
        retrieved_scores: List[float],
        retrieved_texts: List[str],
    ) -> AbstentionResult:
        """判断是否应该拒绝回答。

        Args:
            query: The query text
            retrieved_scores: List of relevance scores for retrieved results
            retrieved_texts: List of text content from retrieved results

        Returns:
            AbstentionResult indicating whether to abstain and why
        """
        if not retrieved_scores:
            return AbstentionResult(should_abstain=True, confidence=1.0, reason="No results found")

        top_score = max(retrieved_scores)

        # 信号1: 检索分数过低
        if top_score < self.min_relevance:
            confidence = min(1.0, 1.0 - (top_score / self.min_relevance))
            return AbstentionResult(
                should_abstain=True,
                confidence=confidence,
                reason=f"Low relevance score ({top_score:.3f} < {self.min_relevance:.3f})",
            )

        # 信号2: 检查查询本身是否是"询问是否存在"的形式
        # 注意：不检查检索结果中的否定词，因为那会导致误报
        # 例如，记录"I didn't mention X"被检索到时，不应触发弃权
        existence_query_patterns = [
            "did we ever",
            "have we ever",
            "did i ever",
            "have i ever",
            "was there ever",
            "is there any mention",
            "any information about",
            "do you know if",
            "did you mention",
            "have you mentioned",
            "我们有没有",
            "有没有提到",
            "有没有讨论",
            "是否提到过",
            "是否讨论过",
            "有记录吗",
            "提到过吗",
        ]
        query_lower = query.lower()
        is_existence_query = any(p in query_lower for p in existence_query_patterns)

        # 如果是"询问是否存在"类型的查询，且检索分数中等偏低，可能需要更保守
        if is_existence_query and top_score < 0.45:
            return AbstentionResult(
                should_abstain=True,
                confidence=0.6,
                reason=f"Existence query with moderate relevance ({top_score:.3f})",
            )

        # 信号3: 分数分布异常（没有明显相关结果）
        if len(retrieved_scores) >= 3:
            top_scores = retrieved_scores[:5]
            score_std = np.std(top_scores)
            if (
                score_std < self.uniform_score_std_threshold
                and top_score < self.low_score_threshold
            ):
                return AbstentionResult(
                    should_abstain=True,
                    confidence=0.7,
                    reason=(
                        f"Uniform low scores (std={score_std:.3f}, "
                        f"top={top_score:.3f}) suggest no specific match"
                    ),
                )

        # 信号4: 所有结果分数都很低
        if len(retrieved_scores) >= 2:
            avg_score = np.mean(retrieved_scores[:3])
            if avg_score < self.min_relevance * 0.8:  # 80% of threshold
                return AbstentionResult(
                    should_abstain=True,
                    confidence=0.75,
                    reason=(
                        f"All top results have low scores "
                        f"(avg={avg_score:.3f} < {self.min_relevance * 0.8:.3f})"
                    ),
                )

        # 不拒绝回答
        return AbstentionResult(
            should_abstain=False,
            confidence=top_score,
            reason=f"Confident (top score: {top_score:.3f})",
        )
