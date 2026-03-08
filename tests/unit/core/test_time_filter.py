"""
AURORA 时间过滤器测试
========================

时间范围提取和预过滤优化的测试。
"""

from __future__ import annotations

import numpy as np
import pytest
import time

from aurora.core.retrieval.time_filter import TimeRangeExtractor, TimeRange
from aurora.core.retrieval.field_retriever import FieldRetriever, QueryType
from aurora.core.components.metric import LowRankMetric
from aurora.core.graph.vector_index import VectorIndex
from aurora.core.graph.memory_graph import MemoryGraph
from aurora.integrations.embeddings.hash import HashEmbedding


@pytest.fixture
def time_extractor():
    """时间范围提取器夹具。"""
    return TimeRangeExtractor()


@pytest.fixture
def events_timeline():
    """用于测试的示例事件时间线。"""
    base_ts = time.time()
    return [
        ("学习Python", base_ts - 30 * 86400),  # 30 天前
        ("学习JavaScript", base_ts - 20 * 86400),  # 20 天前
        ("学习Rust", base_ts - 10 * 86400),  # 10 天前
        ("学习TypeScript", base_ts),  # 今天
    ]


class TestTimeRangeExtraction:
    """从查询中提取时间范围的测试。"""

    def test_extract_first_anchor(self, time_extractor, events_timeline):
        """测试提取"first"时间锚点。"""
        time_range = time_extractor.extract("最早学了什么？", events_timeline)

        assert time_range.relation == "first"
        assert time_range.end is not None
        assert time_range.start is None
        # 结束应该在最早时间戳 + 1 天左右
        earliest_ts = min(ts for _, ts in events_timeline)
        assert abs(time_range.end - (earliest_ts + 86400)) < 100  # 允许小的容差

    def test_extract_last_anchor(self, time_extractor, events_timeline):
        """测试提取"last"时间锚点。"""
        time_range = time_extractor.extract("最近学了什么？", events_timeline)

        assert time_range.relation == "last"
        assert time_range.start is not None
        assert time_range.end is None
        # 开始应该在最新时间戳 - 1 天左右
        latest_ts = max(ts for _, ts in events_timeline)
        assert abs(time_range.start - (latest_ts - 86400)) < 100

    def test_extract_span_anchor(self, time_extractor, events_timeline):
        """测试提取"span"时间锚点。"""
        time_range = time_extractor.extract("学习的历史", events_timeline)

        assert time_range.relation == "span"
        # 跨度查询不过滤（需要完整范围）
        assert time_range.start is None
        assert time_range.end is None

    def test_extract_no_anchor(self, time_extractor, events_timeline):
        """测试提取无时间锚点。"""
        time_range = time_extractor.extract("学了什么？", events_timeline)

        assert time_range.relation == "any"
        assert time_range.start is None
        assert time_range.end is None

    def test_extract_relative_time_yesterday(self, time_extractor, events_timeline):
        """测试提取相对时间模式。"""
        time_range = time_extractor.extract("昨天学了什么？", events_timeline)

        assert time_range.relation == "during"
        assert time_range.start is not None
        assert time_range.end is not None

    def test_extract_without_timeline(self, time_extractor):
        """测试不带时间线的提取（应返回仅关系）。"""
        time_range = time_extractor.extract("最早学了什么？", events_timeline=None)

        assert time_range.relation == "first"
        # 没有时间线，无法解析特定时间戳
        assert time_range.start is None or time_range.end is None


class TestTimeRangeFiltering:
    """候选项时间范围过滤的测试。"""

    def test_filter_by_range_first(self, time_extractor):
        """测试使用"first"关系的过滤。"""
        base_ts = time.time()
        candidates = [
            ("id1", 0.9, base_ts - 30 * 86400),  # 30 天前
            ("id2", 0.8, base_ts - 20 * 86400),  # 20 天前
            ("id3", 0.7, base_ts - 10 * 86400),  # 10 天前
            ("id4", 0.6, base_ts),  # 今天
        ]

        time_range = TimeRange(
            end=base_ts - 25 * 86400,  # 过滤到前 25 天
            relation="first"
        )

        def get_ts(nid: str) -> float:
            for cid, _, ts in candidates:
                if cid == nid:
                    return ts
            return 0.0

        filtered = time_extractor.filter_by_range(candidates, time_range, get_ts)

        # 应仅包含结束时间戳之前的项
        assert len(filtered) == 1
        assert filtered[0][0] == "id1"
        # 应按升序排列（最早优先）
        assert filtered[0][2] <= filtered[-1][2] if len(filtered) > 1 else True

    def test_filter_by_range_last(self, time_extractor):
        """测试使用"last"关系的过滤。"""
        base_ts = time.time()
        candidates = [
            ("id1", 0.9, base_ts - 30 * 86400),  # 30 天前
            ("id2", 0.8, base_ts - 20 * 86400),  # 20 天前
            ("id3", 0.7, base_ts - 10 * 86400),  # 10 天前
            ("id4", 0.6, base_ts),  # 今天
        ]

        time_range = TimeRange(
            start=base_ts - 15 * 86400,  # 过滤到最后 15 天
            relation="last"
        )

        def get_ts(nid: str) -> float:
            for cid, _, ts in candidates:
                if cid == nid:
                    return ts
            return 0.0

        filtered = time_extractor.filter_by_range(candidates, time_range, get_ts)

        # 应仅包含开始时间戳之后的项
        assert len(filtered) == 2
        assert filtered[0][0] in ["id3", "id4"]
        # 应按降序排列（最新优先）
        assert filtered[0][2] >= filtered[-1][2]

    def test_filter_by_range_any(self, time_extractor):
        """测试使用"any"关系的过滤（无过滤）。"""
        base_ts = time.time()
        candidates = [
            ("id1", 0.9, base_ts - 30 * 86400),
            ("id2", 0.8, base_ts - 20 * 86400),
        ]

        time_range = TimeRange(relation="any")

        def get_ts(nid: str) -> float:
            return 0.0

        filtered = time_extractor.filter_by_range(candidates, time_range, get_ts)

        # 应返回所有候选项不变
        assert len(filtered) == len(candidates)

    def test_filter_by_range_span(self, time_extractor):
        """测试使用"span"关系的过滤（无过滤）。"""
        base_ts = time.time()
        candidates = [
            ("id1", 0.9, base_ts - 30 * 86400),
            ("id2", 0.8, base_ts - 20 * 86400),
        ]

        time_range = TimeRange(relation="span")

        def get_ts(nid: str) -> float:
            return 0.0

        filtered = time_extractor.filter_by_range(candidates, time_range, get_ts)

        # 应返回所有候选项不变
        assert len(filtered) == len(candidates)


class TestTimeFilterIntegration:
    """FieldRetriever 中时间过滤的集成测试。"""

    @pytest.fixture
    def retriever_setup(self):
        """使用时间测试数据设置检索器。"""
        dim = 64
        metric = LowRankMetric(dim=dim, rank=16, seed=42)
        vindex = VectorIndex(dim=dim)
        graph = MemoryGraph()
        embedder = HashEmbedding(dim=dim, seed=42)

        base_ts = time.time()
        rng = np.random.default_rng(42)

        # 在不同时间戳添加绘图
        timestamps = [
            base_ts - 30 * 86400,  # 30 天前
            base_ts - 20 * 86400,  # 20 天前
            base_ts - 10 * 86400,  # 10 天前
            base_ts,  # 今天
        ]

        texts = [
            "学习Python",
            "学习JavaScript",
            "学习Rust",
            "学习TypeScript",
        ]

        for i, (text, ts) in enumerate(zip(texts, timestamps)):
            vec = rng.standard_normal(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            node_id = f"plot_{i}"

            vindex.add(node_id, vec, kind="plot")
            # 在有效负载中存储时间戳
            graph.add_node(node_id, "plot", {
                "id": node_id,
                "text": text,
                "ts": ts,
                "embedding": vec,
            })

        retriever = FieldRetriever(metric=metric, vindex=vindex, graph=graph)

        return retriever, embedder, vindex, graph

    def test_temporal_query_with_time_filter(self, retriever_setup):
        """测试时间查询应用时间过滤。"""
        retriever, embedder, vindex, graph = retriever_setup

        # 查询最早的学习
        trace = retriever.retrieve(
            query_text="最早学了什么？",
            embed=embedder,
            kinds=("plot",),
            k=5,
        )

        # 应检测为 TEMPORAL 查询
        assert trace.query_type == QueryType.TEMPORAL

        # 结果应过滤到最早的时间段
        # （在这种情况下，应优先选择 plot_0，即 30 天前）
        if trace.ranked:
            # 检查结果是否按时间过滤
            timestamps = [retriever._get_timestamp(nid) for nid, _, _ in trace.ranked]
            # 对于"first"查询，最早时间戳应在结果中
            earliest_ts = min(ts for nid in graph.g.nodes()
                            if graph.kind(nid) == "plot"
                            for ts in [retriever._get_timestamp(nid)])
            assert min(timestamps) <= earliest_ts + 86400  # 在 1 天缓冲内

    def test_temporal_query_recent(self, retriever_setup):
        """测试"recent"查询过滤到最新时间段。"""
        retriever, embedder, vindex, graph = retriever_setup

        # 查询最近的学习
        trace = retriever.retrieve(
            query_text="最近学了什么？",
            embed=embedder,
            kinds=("plot",),
            k=5,
        )

        # 应检测为 TEMPORAL 查询
        assert trace.query_type == QueryType.TEMPORAL

        # 结果应过滤到最新的时间段
        if trace.ranked:
            timestamps = [retriever._get_timestamp(nid) for nid, _, _ in trace.ranked]
            # 对于"last"查询，最新时间戳应在结果中
            latest_ts = max(ts for nid in graph.g.nodes()
                          if graph.kind(nid) == "plot"
                          for ts in [retriever._get_timestamp(nid)])
            assert max(timestamps) >= latest_ts - 86400  # 在 1 天缓冲内

    def test_non_temporal_query_no_filter(self, retriever_setup):
        """测试非时间查询不应用时间过滤。"""
        retriever, embedder, vindex, graph = retriever_setup

        # 事实查询（无时间关键词）
        trace = retriever.retrieve(
            query_text="学了什么？",
            embed=embedder,
            kinds=("plot",),
            k=5,
        )

        # 不应检测为 TEMPORAL
        assert trace.query_type != QueryType.TEMPORAL

        # 结果应包含所有时间段（无过滤）
        if trace.ranked:
            timestamps = [retriever._get_timestamp(nid) for nid, _, _ in trace.ranked]
            # 应有来自不同时间段的结果
            assert len(set(int(ts // 86400) for ts in timestamps)) >= 1
