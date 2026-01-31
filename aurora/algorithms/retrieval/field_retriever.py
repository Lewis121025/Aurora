"""
AURORA Field Retrieval
=======================

Two-stage retrieval with attractor tracing and graph diffusion.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from aurora.utils.math_utils import l2_normalize, softmax
from aurora.algorithms.models.trace import RetrievalTrace
from aurora.algorithms.components.metric import LowRankMetric
from aurora.embeddings.hash import HashEmbedding
from aurora.algorithms.graph.edge_belief import EdgeBelief
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.graph.vector_index import VectorIndex


class FieldRetriever:
    """Two-stage retrieval with attractor tracing and graph diffusion.

    Stage 1: Continuous-space attractor tracing (mean-shift in learned metric space).
        This yields a context-adaptive "mode" representing what the query is pulling from memory.

    Stage 2: Discrete graph diffusion (personalized PageRank) seeded by vector hits around the attractor.
        Edge probabilities are learned Beta posteriors.

    Attributes:
        metric: Learned low-rank metric for distance computation
        vindex: Vector index for similarity search
        graph: Memory graph with nodes and edges
    """

    def __init__(self, metric: LowRankMetric, vindex: VectorIndex, graph: MemoryGraph):
        """Initialize the field retriever.

        Args:
            metric: Learned metric for distance computation
            vindex: Vector index for initial candidates
            graph: Memory graph for diffusion
        """
        self.metric = metric
        self.vindex = vindex
        self.graph = graph

    def _mean_shift(
        self,
        x0: np.ndarray,
        candidates: List[Tuple[str, np.ndarray, float]],
        steps: int = 8,
    ) -> List[np.ndarray]:
        """Perform mean-shift to find attractor.

        Args:
            x0: Initial query embedding
            candidates: List of (id, vec, mass) tuples
            steps: Number of mean-shift iterations

        Returns:
            List of embeddings along the mean-shift path
        """
        if not candidates:
            return [x0]
        x = x0.copy()
        path = [x.copy()]
        # Dynamic bandwidth: median distance to candidates in current metric
        for _ in range(steps):
            d2s = [self.metric.d2(x, v) for _, v, _ in candidates]
            # Bandwidth as robust scale
            sigma2 = float(np.median(d2s)) + 1e-6
            logits = [-(d2 / (2.0 * sigma2)) + m for d2, (_, _, m) in zip(d2s, candidates)]
            w = softmax(logits)
            new_x = np.zeros_like(x)
            for wi, (_, v, _) in zip(w, candidates):
                new_x += wi * v
            x = l2_normalize(new_x)
            path.append(x.copy())
        return path

    def _pagerank(
        self,
        personalization: Dict[str, float],
        damping: float = 0.85,
        max_iter: int = 50,
    ) -> Dict[str, float]:
        """Compute personalized PageRank on the memory graph.

        Args:
            personalization: Initial node weights
            damping: PageRank damping factor
            max_iter: Maximum iterations

        Returns:
            Dict mapping node IDs to PageRank scores
        """
        G = self.graph.g
        personalization = {n: v for n, v in personalization.items() if n in G}
        if not personalization:
            return {}
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        for u, v, data in G.edges(data=True):
            belief: EdgeBelief = data["belief"]
            H.add_edge(u, v, w=max(1e-6, belief.mean()))
        return nx.pagerank(H, alpha=damping, personalization=personalization, weight="w", max_iter=max_iter)

    def retrieve(
        self,
        query_text: str,
        embed: HashEmbedding,
        kinds: Tuple[str, ...],
        k: int = 5,
    ) -> RetrievalTrace:
        """Retrieve relevant memory items for a query.

        Args:
            query_text: The query text
            embed: Embedding model to use
            kinds: Tuple of kinds to retrieve ("plot", "story", "theme")
            k: Number of results to return

        Returns:
            RetrievalTrace with ranked results
        """
        q = embed.embed(query_text)

        # 1) Seed candidates from vector index (plots + stories + themes)
        candidates: List[Tuple[str, np.ndarray, float]] = []
        seed_scores: Dict[str, float] = {}
        for kind in kinds:
            for _id, sim in self.vindex.search(q, k=50, kind=kind):
                if _id not in self.graph.g:
                    continue
                payload = self.graph.payload(_id)
                vec = getattr(payload, "embedding", getattr(payload, "centroid", getattr(payload, "prototype", None)))
                if vec is None:
                    continue
                mass = float(getattr(payload, "mass")()) if hasattr(payload, "mass") else 0.0
                candidates.append((_id, vec, mass))
                seed_scores[_id] = max(seed_scores.get(_id, 0.0), sim)

        # 2) Continuous attractor tracing
        path = self._mean_shift(q, candidates, steps=8)
        attractor = path[-1]

        # 3) Reseed around attractor and diffuse on graph
        personalization: Dict[str, float] = {}
        for kind in kinds:
            for _id, sim in self.vindex.search(attractor, k=60, kind=kind):
                personalization[_id] = max(personalization.get(_id, 0.0), sim)

        pr = self._pagerank(personalization, damping=0.85, max_iter=60)

        # 4) Rank with emergent masses (no fixed weights; only small scale for tie-breaking)
        ranked: List[Tuple[str, float, str]] = []
        for nid, score in pr.items():
            kind = self.graph.kind(nid)
            if kind not in kinds:
                continue
            payload = self.graph.payload(nid)
            bonus = float(getattr(payload, "mass")()) if hasattr(payload, "mass") else 0.0
            ranked.append((nid, float(score) + 1e-3 * bonus, kind))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return RetrievalTrace(query=query_text, query_emb=q, attractor_path=path, ranked=ranked[:k])
