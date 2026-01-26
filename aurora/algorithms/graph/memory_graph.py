"""
AURORA Memory Graph
====================

Typed node graph with probabilistic edge strengths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from aurora.algorithms.graph.edge_belief import EdgeBelief


class MemoryGraph:
    """Typed node graph with probabilistic edge strengths.

    Stores nodes with kind labels and payloads, connected by edges
    with learnable belief strengths.

    The graph represents relationships between plots, stories, and themes,
    with edge strengths learned from retrieval feedback.

    Attributes:
        g: The underlying NetworkX directed graph
    """

    def __init__(self):
        """Initialize an empty memory graph."""
        self.g = nx.DiGraph()

    def add_node(self, node_id: str, kind: str, payload: Any) -> None:
        """Add a node to the graph.

        Args:
            node_id: Unique identifier for the node
            kind: Node type (e.g., "plot", "story", "theme")
            payload: The data object associated with this node
        """
        self.g.add_node(node_id, kind=kind, payload=payload)

    def kind(self, node_id: str) -> str:
        """Get the kind of a node.

        Args:
            node_id: Node identifier

        Returns:
            Node kind string
        """
        return self.g.nodes[node_id]["kind"]

    def payload(self, node_id: str) -> Any:
        """Get the payload of a node.

        Args:
            node_id: Node identifier

        Returns:
            The data object associated with this node
        """
        return self.g.nodes[node_id]["payload"]

    def ensure_edge(self, src: str, dst: str, edge_type: str) -> None:
        """Ensure an edge exists between two nodes.

        Creates the edge with a new EdgeBelief if it doesn't exist.

        Args:
            src: Source node ID
            dst: Destination node ID
            edge_type: Type of relationship
        """
        if self.g.has_edge(src, dst):
            return
        self.g.add_edge(src, dst, belief=EdgeBelief(edge_type=edge_type))

    def edge_belief(self, src: str, dst: str) -> EdgeBelief:
        """Get the belief for an edge.

        Args:
            src: Source node ID
            dst: Destination node ID

        Returns:
            The EdgeBelief for this edge
        """
        return self.g.edges[src, dst]["belief"]

    def nodes_of_kind(self, kind: str) -> List[str]:
        """Get all nodes of a specific kind.

        Args:
            kind: Node kind to filter by

        Returns:
            List of node IDs with the specified kind
        """
        return [n for n, d in self.g.nodes(data=True) if d.get("kind") == kind]

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize graph structure to JSON-compatible dict.

        Note: Node payloads are NOT serialized here - they should be
        serialized separately (plots, stories, themes dicts).
        """
        nodes = []
        for node_id, data in self.g.nodes(data=True):
            nodes.append({
                "id": node_id,
                "kind": data.get("kind", ""),
            })

        edges = []
        for src, dst, data in self.g.edges(data=True):
            belief: EdgeBelief = data.get("belief")
            edges.append({
                "src": src,
                "dst": dst,
                "belief": belief.to_state_dict() if belief else None,
            })

        return {
            "nodes": nodes,
            "edges": edges,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any], payloads: Optional[Dict[str, Any]] = None) -> "MemoryGraph":
        """Reconstruct graph from state dict.

        Args:
            d: State dict with nodes and edges
            payloads: Optional dict mapping node_id -> payload object
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
            belief = EdgeBelief.from_state_dict(belief_data) if belief_data else EdgeBelief(edge_type="unknown")
            obj.g.add_edge(edge["src"], edge["dst"], belief=belief)

        return obj
