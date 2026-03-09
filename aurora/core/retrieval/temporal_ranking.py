"""
AURORA 时间检索重排
=======================

时间范围过滤、锚点识别后的重排，以及时间多样性选择。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from aurora.core.config.query_types import (
    TEMPORAL_DIVERSITY_BUCKETS,
    TEMPORAL_DIVERSITY_MMR_LAMBDA,
    TEMPORAL_SORT_WEIGHT,
)
from aurora.core.retrieval.query_analysis import TimeAnchor
from aurora.core.retrieval.time_filter import TimeRange


class TemporalRankingMixin:
    """时间感知的过滤与排序逻辑。"""

    def _get_timestamp(self, node_id: str) -> float:
        try:
            payload = self.graph.payload(node_id)
            return getattr(payload, "ts", getattr(payload, "created_ts", 0.0))
        except Exception:
            return 0.0

    def _matches_time_range(self, node_id: str, time_range: Optional[TimeRange]) -> bool:
        if time_range is None or time_range.relation in {"any", "span"}:
            return True

        ts = self._get_timestamp(node_id)
        if time_range.start is not None and ts < time_range.start:
            return False
        if time_range.end is not None and ts > time_range.end:
            return False
        return True

    def _apply_time_filter(
        self,
        ranked: List[Tuple[str, float, str]],
        time_range: TimeRange,
    ) -> List[Tuple[str, float, str]]:
        if time_range.relation in {"any", "span"}:
            return ranked

        filtered = [item for item in ranked if self._matches_time_range(item[0], time_range)]

        if time_range.relation == "first":
            filtered.sort(key=lambda item: self._get_timestamp(item[0]))
        elif time_range.relation == "last":
            filtered.sort(key=lambda item: -self._get_timestamp(item[0]))

        return filtered

    def _temporal_aware_rerank(
        self,
        ranked: List[Tuple[str, float, str]],
        query_text: str,
        k: int,
    ) -> List[Tuple[str, float, str]]:
        if not ranked:
            return ranked

        items_with_ts = [
            (node_id, score, kind, self._get_timestamp(node_id))
            for node_id, score, kind in ranked
        ]
        time_anchor = self._detect_time_anchor(query_text)

        if time_anchor == TimeAnchor.RECENT:
            items_with_ts.sort(key=lambda item: (item[3], item[1]), reverse=True)
            return [(node_id, score, kind) for node_id, score, kind, _ in items_with_ts[:k]]

        if time_anchor == TimeAnchor.EARLIEST:
            items_with_ts.sort(key=lambda item: (-item[3], item[1]), reverse=True)
            return [(node_id, score, kind) for node_id, score, kind, _ in items_with_ts[:k]]

        if time_anchor == TimeAnchor.SPAN:
            return self._select_temporal_diversity(items_with_ts, k)

        return self._blend_semantic_temporal(items_with_ts, k)

    def _blend_semantic_temporal(
        self,
        items_with_ts: List[Tuple[str, float, str, float]],
        k: int,
    ) -> List[Tuple[str, float, str]]:
        if not items_with_ts:
            return []

        timestamps = [ts for _, _, _, ts in items_with_ts]
        max_ts = max(timestamps)
        min_ts = min(timestamps)
        ts_range = max_ts - min_ts if max_ts > min_ts else 1.0

        reranked: List[Tuple[str, float, str]] = []
        for node_id, score, kind, ts in items_with_ts:
            normalized_ts = (ts - min_ts) / ts_range if ts_range > 0 else 0.5
            combined = (1.0 - TEMPORAL_SORT_WEIGHT) * score + TEMPORAL_SORT_WEIGHT * normalized_ts
            reranked.append((node_id, combined, kind))

        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:k]

    def _select_temporal_diversity(
        self,
        items_with_ts: List[Tuple[str, float, str, float]],
        k: int,
    ) -> List[Tuple[str, float, str]]:
        if not items_with_ts:
            return []
        if len(items_with_ts) <= k:
            return [(node_id, score, kind) for node_id, score, kind, _ in items_with_ts]

        timestamps = [ts for _, _, _, ts in items_with_ts]
        max_ts = max(timestamps)
        min_ts = min(timestamps)
        ts_range = max_ts - min_ts if max_ts > min_ts else 1.0

        def bucket_for(ts: float) -> int:
            if ts_range == 0:
                return 0
            normalized = (ts - min_ts) / ts_range
            scaled = int(normalized * TEMPORAL_DIVERSITY_BUCKETS)
            return min(scaled, TEMPORAL_DIVERSITY_BUCKETS - 1)

        remaining = [
            (node_id, score, kind, ts, bucket_for(ts))
            for node_id, score, kind, ts in items_with_ts
        ]
        selected: List[Tuple[str, float, str]] = []
        selected_buckets: List[int] = []

        while len(selected) < k and remaining:
            best_idx = -1
            best_score = float("-inf")

            for idx, (node_id, score, kind, _ts, bucket) in enumerate(remaining):
                bucket_count = selected_buckets.count(bucket)
                temporal_penalty = bucket_count / max(len(selected), 1) if selected else 0.0
                mmr_score = (
                    TEMPORAL_DIVERSITY_MMR_LAMBDA * score
                    - (1.0 - TEMPORAL_DIVERSITY_MMR_LAMBDA) * temporal_penalty
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            node_id, score, kind, _ts, bucket = remaining.pop(best_idx)
            selected.append((node_id, score, kind))
            selected_buckets.append(bucket)

        return selected
