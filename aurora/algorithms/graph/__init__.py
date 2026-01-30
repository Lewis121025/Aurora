"""
AURORA Graph Structures
========================

Graph-based memory organization and edge beliefs.
"""

from aurora.algorithms.graph.edge_belief import EdgeBelief
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.graph.vector_index import VectorIndex

try:
    from aurora.algorithms.graph.faiss_index import FAISSVectorIndex, FAISS_AVAILABLE
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
