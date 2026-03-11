"""
AURORA 字段检索器测试
============================

FieldRetriever 组件的测试。

测试覆盖：
- 均值漂移吸引子追踪
- 个性化 PageRank 图扩散
- 组合检索
- 类型过滤
"""

from __future__ import annotations

import numpy as np
import pytest

from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.soul.retrieval import FieldRetriever, LowRankMetric, MemoryGraph, VectorIndex
from tests.helpers.query_router import build_test_query_analyzer


@pytest.fixture
def retriever_setup():
    """使用基本组件设置检索器。"""
    dim = 64
    metric = LowRankMetric(dim=dim, rank=16, seed=42)
    vindex = VectorIndex(dim=dim)
    graph = MemoryGraph()
    embedder = HashEmbedding(dim=dim, seed=42)

    # 添加一些测试向量
    rng = np.random.default_rng(42)

    for i in range(20):
        vec = rng.standard_normal(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        node_id = f"plot_{i}"
        kind = "plot"

        vindex.add(node_id, vec, kind=kind)
        graph.add_node(node_id, kind, {"id": node_id})

    # 添加一些边
    for i in range(19):
        graph.ensure_edge(f"plot_{i}", f"plot_{i + 1}", "temporal")
        graph.ensure_edge(f"plot_{i + 1}", f"plot_{i}", "temporal")

    retriever = FieldRetriever(
        metric=metric,
        vindex=vindex,
        graph=graph,
        query_analyzer=build_test_query_analyzer(),
    )

    return retriever, embedder, vindex, graph


class TestFieldRetrieverBasic:
    """FieldRetriever 的基本测试。"""

    def test_retrieve_returns_results(self, retriever_setup, identity_state):
        """测试 retrieve 返回结果。"""
        retriever, embedder, vindex, graph = retriever_setup

        trace = retriever.retrieve(
            query_text="测试查询",
            embedder=embedder,
            state=identity_state,
            kinds=["plot"],
            k=5,
        )

        assert trace is not None
        assert len(trace.ranked) <= 5

    def test_retrieve_with_empty_index(self, metric, identity_state):
        """测试使用空索引的 retrieve。"""
        vindex = VectorIndex(dim=64)
        graph = MemoryGraph()
        retriever = FieldRetriever(
            metric=metric,
            vindex=vindex,
            graph=graph,
            query_analyzer=build_test_query_analyzer(),
        )
        embedder = HashEmbedding(dim=64, seed=42)

        trace = retriever.retrieve(
            query_text="测试",
            embedder=embedder,
            state=identity_state,
            kinds=["plot"],
            k=5,
        )

        assert trace is not None
        assert len(trace.ranked) == 0


class TestMeanShift:
    """均值漂移吸引子追踪的测试。"""

    def test_mean_shift_convergence(self, retriever_setup):
        """测试均值漂移收敛到吸引子。"""
        retriever, embedder, vindex, graph = retriever_setup

        # 创建查询嵌入
        query = embedder.embed("测试查询")

        # 从 vindex 创建候选项
        candidates = []
        for pid in ["plot_0", "plot_1", "plot_2"]:
            emb = embedder.embed(f"测试文本{pid}")
            candidates.append((pid, emb, 1.0))

        # 运行均值漂移
        path = retriever._mean_shift(query, candidates, steps=8)

        # 路径应包含嵌入
        assert len(path) > 0
        assert path[-1] is not None
        assert len(path[-1]) == 64

    def test_mean_shift_path_recorded(self, retriever_setup):
        """测试均值漂移记录路径。"""
        retriever, embedder, vindex, graph = retriever_setup

        query = embedder.embed("测试查询")

        # 创建候选项
        candidates = [(f"plot_{i}", embedder.embed(f"测试{i}"), 1.0) for i in range(3)]

        path = retriever._mean_shift(query, candidates, steps=8)

        # 路径应记录步骤
        assert isinstance(path, list)
        assert len(path) == 9  # 初始 + 8 步


class TestPageRank:
    """个性化 PageRank 的测试。"""

    def test_pagerank_with_seeds(self, retriever_setup):
        """测试带有种子节点的 PageRank。"""
        retriever, embedder, vindex, graph = retriever_setup

        # 使用一些节点作为种子
        seeds = {"plot_0": 0.5, "plot_1": 0.5}

        scores = retriever._pagerank(seeds, damping=0.85, max_iter=60)

        assert scores is not None
        assert isinstance(scores, dict)
        # 如果图没有这些节点，可能为空

    def test_pagerank_empty_seeds(self, retriever_setup):
        """测试带有空种子的 PageRank。"""
        retriever, embedder, vindex, graph = retriever_setup

        scores = retriever._pagerank({}, damping=0.85, max_iter=60)

        assert scores is not None
        assert len(scores) == 0

    def test_pagerank_ignores_negative_edges(self, retriever_setup):
        """测试负边不会直接进入扩散权重。"""
        retriever, embedder, vindex, graph = retriever_setup

        graph.ensure_edge(
            "plot_0",
            "plot_10",
            "contradicts",
            sign=-1,
            weight=10.0,
            confidence=1.0,
            provenance="test",
        )
        scores = retriever._pagerank({"plot_0": 1.0}, damping=0.85, max_iter=60)

        assert scores["plot_10"] < 0.2

    def test_negative_edges_apply_bounded_post_retrieval_inhibition(
        self, retriever_setup, identity_state
    ):
        retriever, embedder, vindex, graph = retriever_setup

        query = embedder.embed("图记忆冲突测试")
        idx0 = vindex.ids.index("plot_0")
        idx1 = vindex.ids.index("plot_1")
        vindex.vecs[idx0] = query.astype(np.float32)
        vindex.vecs[idx1] = ((query + 0.02).astype(np.float32)) / np.linalg.norm(query + 0.02)
        graph.add_node("anchor_core", "anchor", {"id": "anchor_core"})
        graph.ensure_edge(
            "anchor_core",
            "plot_0",
            "contradicts_self",
            sign=-1,
            weight=1.0,
            confidence=1.0,
            provenance="test",
        )
        graph.ensure_edge(
            "plot_0",
            "anchor_core",
            "contradicts_self",
            sign=-1,
            weight=1.0,
            confidence=1.0,
            provenance="test",
        )

        trace = retriever.retrieve(
            query_text="图记忆冲突测试",
            embedder=embedder,
            state=identity_state,
            kinds=["plot"],
            k=3,
        )

        assert trace.ranked
        assert trace.ranked[0][0] == "plot_1"


class TestKindFiltering:
    """检索中类型过滤的测试。"""

    def test_filter_by_kind(self, retriever_setup, identity_state):
        """测试结果按类型过滤。"""
        retriever, embedder, vindex, graph = retriever_setup

        trace = retriever.retrieve(
            query_text="测试",
            embedder=embedder,
            state=identity_state,
            kinds=["plot"],
            k=5,
        )

        # 所有结果应为请求的类型
        for node_id, score, kind in trace.ranked:
            assert kind == "plot"

    def test_multiple_kinds(self, retriever_setup, identity_state):
        """测试使用多种类型的检索。"""
        retriever, embedder, vindex, graph = retriever_setup

        # 添加一个故事节点
        vec = np.random.randn(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vindex, graph = retriever_setup[2], retriever_setup[3]
        vindex.add("story_1", vec, kind="story")
        graph.add_node("story_1", "story", {"id": "story_1"})

        trace = retriever.retrieve(
            query_text="测试",
            embedder=embedder,
            state=identity_state,
            kinds=["plot", "story"],
            k=10,
        )

        # 结果应包含两种类型（如果可用）
        kinds_in_results = {kind for _, _, kind in trace.ranked}
        assert "plot" in kinds_in_results or "story" in kinds_in_results

    def test_non_plot_payload_without_embedded_vector_still_retrieves(
        self, retriever_setup, identity_state
    ):
        """测试 story/theme 等节点即使 payload 未携带向量，也不会在重排阶段崩溃。"""
        retriever, embedder, vindex, graph = retriever_setup

        vec = np.random.randn(64).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vindex.add("story_raw", vec, kind="story")
        graph.add_node(
            "story_raw",
            "story",
            {
                "id": "story_raw",
                "name": "raw story payload",
                "updated_ts": 1.0,
            },
        )

        trace = retriever.retrieve(
            query_text="story raw",
            embedder=embedder,
            state=identity_state,
            kinds=["story"],
            k=5,
        )

        assert trace is not None
        assert trace.ranked
        assert trace.ranked[0][2] == "story"


class TestRetrievalTrace:
    """RetrievalTrace 结构的测试。"""

    def test_trace_has_query_info(self, retriever_setup, identity_state):
        """测试追踪包含查询信息。"""
        retriever, embedder, vindex, graph = retriever_setup

        trace = retriever.retrieve(
            query_text="测试查询文本",
            embedder=embedder,
            state=identity_state,
            kinds=["plot"],
            k=5,
        )

        assert trace.query == "测试查询文本"
        assert trace.query_emb is not None

    def test_trace_has_attractor_path(self, retriever_setup, identity_state):
        """测试追踪包含吸引子路径。"""
        retriever, embedder, vindex, graph = retriever_setup

        trace = retriever.retrieve(
            query_text="测试",
            embedder=embedder,
            state=identity_state,
            kinds=["plot"],
            k=5,
        )

        assert hasattr(trace, "attractor_path")
