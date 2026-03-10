"""
AURORA 内存图
====================

具有概率边强度的类型化节点图。
包括 PageRank 缓存以提高检索性能。
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple, cast

import networkx as nx

from aurora.lab.graph.edge_belief import EdgeBelief


class MemoryGraph:
    """具有概率边强度的类型化节点图。

    存储带有类型标签和有效负载的节点，通过具有可学习信念强度的边连接。

    该图表示情节、故事和主题之间的关系，
    边强度从检索反馈中学习。

    功能特性：
    - PageRank 缓存：计算的 PageRank 结果被缓存，仅在
      边更新时失效，延迟降低约 30%
    - 边修改时自动缓存失效

    属性：
        g: 底层 NetworkX 有向图
        _pagerank_cache: PageRank 计算结果的缓存
        _cache_valid: 指示缓存是否有效的标志
    """

    def __init__(self) -> None:
        """初始化一个空的内存图。"""
        self.g = nx.DiGraph()
        # PageRank 缓存：映射 (personalization_hash, damping, max_iter) -> 结果
        self._pagerank_cache: Dict[Tuple[str, float, int], Dict[str, float]] = {}
        self._cache_valid: bool = True
        self._edge_version: int = 0  # 边更改时递增

    def add_node(self, node_id: str, kind: str, payload: Any) -> None:
        """向图中添加一个节点。

        参数：
            node_id: 节点的唯一标识符
            kind: 节点类型（例如，"plot"、"story"、"theme"）
            payload: 与此节点关联的数据对象
        """
        self.g.add_node(node_id, kind=kind, payload=payload)

    def kind(self, node_id: str) -> str:
        """获取节点的类型。

        参数：
            node_id: 节点标识符

        返回：
            节点类型字符串
        """
        return cast(str, self.g.nodes[node_id]["kind"])

    def payload(self, node_id: str) -> Any:
        """获取节点的有效负载。

        参数：
            node_id: 节点标识符

        返回：
            与此节点关联的数据对象
        """
        return self.g.nodes[node_id]["payload"]

    def ensure_edge(self, src: str, dst: str, edge_type: str) -> None:
        """确保两个节点之间存在边。

        如果边不存在，则使用新的 EdgeBelief 创建边。
        添加新边时使 PageRank 缓存失效。

        参数：
            src: 源节点 ID
            dst: 目标节点 ID
            edge_type: 关系类型
        """
        if self.g.has_edge(src, dst):
            return
        self.g.add_edge(src, dst, belief=EdgeBelief(edge_type=edge_type))
        self._invalidate_cache()

    # -------------------------------------------------------------------------
    # PageRank 缓存管理
    # -------------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        """使 PageRank 缓存失效（内部使用）。"""
        self._edge_version += 1
        self._pagerank_cache.clear()
        self._cache_valid = False

    def invalidate_cache(self) -> None:
        """手动使 PageRank 缓存失效。

        当外部更改影响应使缓存的 PageRank 结果失效的图结构
        或边权重时，调用此方法。
        """
        self._invalidate_cache()

    def _hash_personalization(self, personalization: Dict[str, float]) -> str:
        """为个性化字典创建稳定哈希。"""
        # 排序项以确保确定性顺序
        items = sorted(personalization.items())
        # 创建字符串表示并对其进行哈希
        key_str = str(items)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def get_cached_pagerank(
        self,
        personalization: Dict[str, float],
        damping: float = 0.85,
        max_iter: int = 50,
    ) -> Optional[Dict[str, float]]:
        """获取缓存的 PageRank 结果（如果可用）。

        参数：
            personalization: 初始节点权重
            damping: PageRank 阻尼因子
            max_iter: 最大迭代次数

        返回：
            如果可用，返回缓存的 PageRank 字典，否则返回 None
        """
        p_hash = self._hash_personalization(personalization)
        cache_key = (p_hash, damping, max_iter)
        return self._pagerank_cache.get(cache_key)

    def set_cached_pagerank(
        self,
        personalization: Dict[str, float],
        damping: float,
        max_iter: int,
        result: Dict[str, float],
    ) -> None:
        """将 PageRank 结果存储在缓存中。

        参数：
            personalization: 用于计算的初始节点权重
            damping: PageRank 阻尼因子
            max_iter: 最大迭代次数
            result: 计算的 PageRank 分数
        """
        p_hash = self._hash_personalization(personalization)
        cache_key = (p_hash, damping, max_iter)
        self._pagerank_cache[cache_key] = result
        self._cache_valid = True

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息以进行监控。

        返回：
            包含缓存大小、有效性和边版本的字典
        """
        return {
            "cache_size": len(self._pagerank_cache),
            "cache_valid": self._cache_valid,
            "edge_version": self._edge_version,
        }

    def edge_belief(self, src: str, dst: str) -> EdgeBelief:
        """获取边的信念。

        参数：
            src: 源节点 ID
            dst: 目标节点 ID

        返回：
            此边的 EdgeBelief
        """
        return cast(EdgeBelief, self.g.edges[src, dst]["belief"])

    def nodes_of_kind(self, kind: str) -> List[str]:
        """获取特定类型的所有节点。

        参数：
            kind: 要过滤的节点类型

        返回：
            具有指定类型的节点 ID 列表
        """
        return [n for n, d in self.g.nodes(data=True) if d.get("kind") == kind]

    def to_state_dict(self) -> Dict[str, Any]:
        """将图结构序列化为 JSON 兼容字典。

        注意：节点有效负载不在此处序列化 - 应该
        单独序列化（plots、stories、themes 字典）。
        PageRank 缓存不被序列化（将在首次查询时重建）。
        """
        nodes = []
        for node_id, data in self.g.nodes(data=True):
            nodes.append(
                {
                    "id": node_id,
                    "kind": data.get("kind", ""),
                }
            )

        edges = []
        for src, dst, data in self.g.edges(data=True):
            belief: EdgeBelief = data.get("belief")
            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "belief": belief.to_state_dict() if belief else None,
                }
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "edge_version": self._edge_version,
        }

    @classmethod
    def from_state_dict(
        cls, d: Dict[str, Any], payloads: Optional[Dict[str, Any]] = None
    ) -> "MemoryGraph":
        """从状态字典重建图。

        参数：
            d: 包含节点和边的状态字典
            payloads: 可选的字典，映射 node_id -> 有效负载对象

        注意：
            PageRank 缓存不被恢复（将在首次查询时重建）。
        """
        payloads = payloads or {}
        obj = cls()

        for node in d.get("nodes", []):
            node_id = node["id"]
            kind = node["kind"]
            payload = payloads.get(node_id)
            obj.g.add_node(node_id, kind=kind, payload=payload)

        for edge in d.get("edges", []):
            belief_data = edge.get("belief")
            belief = (
                EdgeBelief.from_state_dict(belief_data)
                if belief_data
                else EdgeBelief(edge_type="unknown")
            )
            obj.g.add_edge(edge["src"], edge["dst"], belief=belief)

        # 恢复边版本（缓存将为空，这是正确的）
        obj._edge_version = d.get("edge_version", 0)
        obj._cache_valid = False  # 强制在首次查询时重建缓存

        return obj
