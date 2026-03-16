"""记忆存储模块。

实现内存级图数据库（MemoryStore），管理：
- Node: 记忆节点（原子事实）
- Edge: 节点间的关联边

极简设计：退化为纯图数据库，不做任何认知模拟。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class Node:
    """记忆节点。

    Attributes:
        node_id: 节点唯一 ID。
        relation_id: 所属关系 ID。
        content: 节点内容（原子事实文本）。
        created_at: 创建时间戳。
        last_touched_at: 最后触碰时间戳。
    """

    node_id: str
    relation_id: str
    content: str
    created_at: float
    last_touched_at: float


@dataclass(frozen=True, slots=True)
class Edge:
    """记忆边。

    Attributes:
        edge_id: 边唯一 ID。
        src_node_id: 源节点 ID。
        dst_node_id: 目标节点 ID。
        weight: 权重（0.0-1.0）。
        created_at: 创建时间戳。
    """

    edge_id: str
    src_node_id: str
    dst_node_id: str
    weight: float
    created_at: float


class MemoryStore:
    """内存图数据库。

    极简存储：仅包含 Node 和 Edge，维护索引关系。

    Attributes:
        nodes: 节点字典。
        edges: 边字典。
        relation_nodes: 关系 -> 节点索引。
        node_edges: 节点 -> 边索引。
    """

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, Edge] = {}
        self.relation_nodes: dict[str, list[str]] = defaultdict(list)
        self.node_edges: dict[str, list[str]] = defaultdict(list)

        self._dirty_nodes: set[str] = set()
        self._dirty_edges: set[str] = set()
        self._deleted_nodes: set[str] = set()
        self._deleted_edges: set[str] = set()

    def create_node(
        self,
        relation_id: str,
        content: str,
        now_ts: float,
    ) -> Node:
        """创建节点。

        Args:
            relation_id: 关系 ID。
            content: 节点内容。
            now_ts: 当前时间戳。

        Returns:
            新节点。
        """
        node = Node(
            node_id=f"node_{uuid4().hex[:12]}",
            relation_id=relation_id,
            content=content,
            created_at=now_ts,
            last_touched_at=now_ts,
        )
        self.add_node(node)
        return node

    def add_node(self, node: Node) -> None:
        """添加节点。

        Args:
            node: 节点对象。
        """
        self.nodes[node.node_id] = node
        self._dirty_nodes.add(node.node_id)
        if node.node_id not in self.relation_nodes[node.relation_id]:
            self.relation_nodes[node.relation_id].append(node.node_id)

    def link_nodes(
        self,
        src_node_id: str,
        dst_node_id: str,
        weight: float,
        now_ts: float,
    ) -> Edge:
        """创建边连接两个节点。

        Args:
            src_node_id: 源节点 ID。
            dst_node_id: 目标节点 ID。
            weight: 权重。
            now_ts: 当前时间戳。

        Returns:
            新边。
        """
        edge = Edge(
            edge_id=f"edge_{uuid4().hex[:12]}",
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            weight=weight,
            created_at=now_ts,
        )
        self.add_edge(edge)
        return edge

    def add_edge(self, edge: Edge) -> None:
        """添加边。

        Args:
            edge: 边对象。
        """
        self.edges[edge.edge_id] = edge
        self._dirty_edges.add(edge.edge_id)
        self.node_edges[edge.src_node_id].append(edge.edge_id)
        self.node_edges[edge.dst_node_id].append(edge.edge_id)

    def nodes_for_relation(self, relation_id: str) -> tuple[Node, ...]:
        """获取关系的节点列表。

        Args:
            relation_id: 关系 ID。

        Returns:
            节点元组。
        """
        return tuple(
            self.nodes[node_id]
            for node_id in self.relation_nodes.get(relation_id, ())
            if node_id in self.nodes
        )

    def edges_for_node(self, node_id: str) -> tuple[Edge, ...]:
        """获取节点的边列表。

        Args:
            node_id: 节点 ID。

        Returns:
            边元组。
        """
        return tuple(
            self.edges[edge_id]
            for edge_id in self.node_edges.get(node_id, ())
            if edge_id in self.edges
        )

    def touch_node(self, node_id: str, now_ts: float) -> None:
        """触碰节点。

        Args:
            node_id: 节点 ID。
            now_ts: 当前时间戳。
        """
        node = self.nodes.get(node_id)
        if node is None:
            return
        from dataclasses import replace

        self.nodes[node_id] = replace(node, last_touched_at=now_ts)
        self._dirty_nodes.add(node_id)

    def remove_node(self, node_id: str) -> None:
        """移除节点。

        Args:
            node_id: 节点 ID。
        """
        node = self.nodes.pop(node_id, None)
        if node is None:
            return
        self._deleted_nodes.add(node_id)
        self._dirty_nodes.discard(node_id)
        rel_list = self.relation_nodes.get(node.relation_id)
        if rel_list and node_id in rel_list:
            rel_list.remove(node_id)
        self.node_edges.pop(node_id, None)

    def remove_edge(self, edge_id: str) -> None:
        """移除边。

        Args:
            edge_id: 边 ID。
        """
        edge = self.edges.pop(edge_id, None)
        if edge is None:
            return
        self._deleted_edges.add(edge_id)
        self._dirty_edges.discard(edge_id)
        for node_id in (edge.src_node_id, edge.dst_node_id):
            edges = self.node_edges.get(node_id)
            if edges and edge_id in edges:
                edges.remove(edge_id)

    def clear_dirty(self) -> None:
        """清空脏标记。

        在持久化完成后调用。
        """
        self._dirty_nodes.clear()
        self._dirty_edges.clear()
        self._deleted_nodes.clear()
        self._deleted_edges.clear()
