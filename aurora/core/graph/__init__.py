"""
AURORA Graph Structures
========================

Graph-based memory organization and edge beliefs.
"""

from aurora.core.graph.edge_belief import EdgeBelief
from aurora.core.graph.memory_graph import MemoryGraph
from aurora.core.graph.vector_index import VectorIndex

try:
    from aurora.core.graph.faiss_index import FAISSVectorIndex, FAISS_AVAILABLE
except ImportError:
    FAISSVectorIndex = None
    FAISS_AVAILABLE = False

__all__ = [
    "EdgeBelief",
    "MemoryGraph",
    "VectorIndex",
    "FAISSVectorIndex",
    "FAISS_AVAILABLE",
]
