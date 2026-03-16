"""状态投影测试。"""

import pytest

from aurora.memory.store import MemoryStore, Node
from aurora.relation.state import RelationalState
from aurora.relation.tension import TensionQueue


class TestMemoryStore:
    def test_create_node(self):
        store = MemoryStore()
        node = store.create_node("rel1", "test content", 1000.0)

        assert node.relation_id == "rel1"
        assert node.content == "test content"

    def test_nodes_for_relation(self):
        store = MemoryStore()
        store.create_node("rel1", "content1", 1000.0)
        store.create_node("rel1", "content2", 1001.0)
        store.create_node("rel2", "content3", 1002.0)

        nodes = store.nodes_for_relation("rel1")
        assert len(nodes) == 2

    def test_link_nodes(self):
        store = MemoryStore()
        n1 = store.create_node("rel1", "node1", 1000.0)
        n2 = store.create_node("rel1", "node2", 1001.0)

        edge = store.link_nodes(n1.node_id, n2.node_id, 0.8, 1002.0)
        assert edge.weight == 0.8

    def test_touch_node(self):
        store = MemoryStore()
        node = store.create_node("rel1", "content", 1000.0)

        store.touch_node(node.node_id, 2000.0)
        updated = store.nodes[node.node_id]
        assert updated.last_touched_at == 2000.0
