from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from aurora.core.soul_memory.models import (
    EdgeBelief,
    IdentityState,
    Plot,
    RetrievalTrace,
    StoryArc,
    Theme,
    l2_normalize,
    softmax,
)
from aurora.integrations.embeddings.base import EmbeddingProvider
from aurora.utils.math_utils import cosine_sim


class OnlineKDE:
    def __init__(self, dim: int, reservoir: int = 4096, k_sigma: int = 25, seed: int = 0):
        self.dim = dim
        self.reservoir = reservoir
        self.k_sigma = k_sigma
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._vecs: List[np.ndarray] = []

    def add(self, x: np.ndarray) -> None:
        vec = x.astype(np.float32)
        if len(self._vecs) < self.reservoir:
            self._vecs.append(vec)
            return
        idx = int(self.rng.integers(0, len(self._vecs) + 1))
        if idx < len(self._vecs):
            self._vecs[idx] = vec

    def _sigma(self, x: np.ndarray) -> float:
        if not self._vecs:
            return 1.0
        dists = [float(np.linalg.norm(x - v)) for v in self._vecs]
        dists.sort()
        k = min(self.k_sigma, len(dists))
        median = float(np.median(dists[:k])) if k > 0 else float(np.median(dists))
        return median + 1e-6

    def log_density(self, x: np.ndarray) -> float:
        if not self._vecs:
            return -10.0
        sigma = self._sigma(x)
        inv2 = 1.0 / (2.0 * sigma * sigma)
        vals = []
        for vec in self._vecs:
            d2 = float(np.dot(x - vec, x - vec))
            vals.append(math.exp(-d2 * inv2))
        prob = sum(vals) / len(vals)
        return math.log(prob + 1e-12)

    def surprise(self, x: np.ndarray) -> float:
        return -self.log_density(x)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "reservoir": self.reservoir,
            "k_sigma": self.k_sigma,
            "seed": self._seed,
            "vecs": [vec.astype(np.float32).tolist() for vec in self._vecs],
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "OnlineKDE":
        obj = cls(
            dim=int(data["dim"]),
            reservoir=int(data["reservoir"]),
            k_sigma=int(data.get("k_sigma", 25)),
            seed=int(data.get("seed", 0)),
        )
        obj._vecs = [np.asarray(vec, dtype=np.float32) for vec in data.get("vecs", [])]
        return obj


class LowRankMetric:
    def __init__(self, dim: int, rank: int = 64, seed: int = 0):
        self.dim = dim
        self.rank = min(rank, dim)
        self._seed = seed
        rng = np.random.default_rng(seed)
        self.L = np.eye(dim, dtype=np.float32)[: self.rank].copy()
        self.L += (0.01 * rng.normal(size=self.L.shape)).astype(np.float32)
        self.G = np.zeros_like(self.L)
        self.t = 0

    def d2(self, x: np.ndarray, y: np.ndarray) -> float:
        delta = (x - y).astype(np.float32)
        projected = self.L @ delta
        return float(np.dot(projected, projected))

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1.0 / (1.0 + self.d2(x, y))

    def update_triplet(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        margin: float = 1.0,
    ) -> float:
        self.t += 1
        ap = (anchor - positive).astype(np.float32)
        an = (anchor - negative).astype(np.float32)
        lap = self.L @ ap
        lan = self.L @ an
        dap = float(np.dot(lap, lap))
        dan = float(np.dot(lan, lan))
        loss = max(0.0, margin + dap - dan)
        if loss <= 0.0:
            return 0.0
        grad = 2.0 * (np.outer(lap, ap) - np.outer(lan, an)).astype(np.float32)
        self.G += grad * grad
        step = (1.0 / math.sqrt(self.t + 1.0)) * grad / (np.sqrt(self.G) + 1e-8)
        self.L -= step
        return float(loss)

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "rank": self.rank,
            "seed": self._seed,
            "L": self.L.tolist(),
            "G": self.G.tolist(),
            "t": self.t,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "LowRankMetric":
        obj = cls(
            dim=int(data["dim"]),
            rank=int(data["rank"]),
            seed=int(data.get("seed", 0)),
        )
        obj.L = np.asarray(data["L"], dtype=np.float32)
        obj.G = np.asarray(data["G"], dtype=np.float32)
        obj.t = int(data.get("t", 0))
        return obj


class ThompsonBernoulliGate:
    def __init__(self, feature_dim: int, seed: int = 0):
        self.d = feature_dim
        self._seed = seed
        self.rng = np.random.default_rng(seed)
        self.w_mean = np.zeros(self.d, dtype=np.float32)
        self.prec = np.ones(self.d, dtype=np.float32) * 1e-2
        self.grad2 = np.zeros(self.d, dtype=np.float32)
        self.t = 0

    def _sample_w(self) -> np.ndarray:
        std = np.sqrt(1.0 / (self.prec + 1e-9))
        return self.w_mean + self.rng.normal(size=self.d).astype(np.float32) * std

    def prob(self, x: np.ndarray) -> float:
        weight = self._sample_w()
        dot = float(np.dot(weight, x))
        if dot >= 0:
            z = math.exp(-dot)
            return 1.0 / (1.0 + z)
        z = math.exp(dot)
        return z / (1.0 + z)

    def decide(self, x: np.ndarray) -> bool:
        return bool(self.rng.random() < self.prob(x))

    def update(self, x: np.ndarray, reward: float) -> None:
        self.t += 1
        y = 1.0 if reward > 0 else 0.0
        dot = float(np.dot(self.w_mean, x))
        if dot >= 0:
            z = math.exp(-dot)
            p = 1.0 / (1.0 + z)
        else:
            z = math.exp(dot)
            p = z / (1.0 + z)
        grad = (y - p) * x
        self.grad2 = 0.99 * self.grad2 + 0.01 * (grad * grad)
        step = (1.0 / math.sqrt(self.t + 1.0)) * grad / (np.sqrt(self.grad2) + 1e-6)
        self.w_mean += step
        self.prec += grad * grad

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "d": self.d,
            "seed": self._seed,
            "w_mean": self.w_mean.tolist(),
            "prec": self.prec.tolist(),
            "grad2": self.grad2.tolist(),
            "t": self.t,
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "ThompsonBernoulliGate":
        obj = cls(feature_dim=int(data["d"]), seed=int(data.get("seed", 0)))
        obj.w_mean = np.asarray(data["w_mean"], dtype=np.float32)
        obj.prec = np.asarray(data["prec"], dtype=np.float32)
        obj.grad2 = np.asarray(data["grad2"], dtype=np.float32)
        obj.t = int(data.get("t", 0))
        return obj


class MemoryGraph:
    def __init__(self) -> None:
        self.g = nx.DiGraph()

    def add_node(self, node_id: str, kind: str, payload: Any) -> None:
        self.g.add_node(node_id, kind=kind, payload=payload)

    def kind(self, node_id: str) -> str:
        return str(self.g.nodes[node_id]["kind"])

    def payload(self, node_id: str) -> Any:
        return self.g.nodes[node_id]["payload"]

    def ensure_edge(self, src: str, dst: str, edge_type: str) -> None:
        if self.g.has_edge(src, dst):
            return
        self.g.add_edge(src, dst, belief=EdgeBelief(edge_type=edge_type))

    def edge_belief(self, src: str, dst: str) -> EdgeBelief:
        return self.g.edges[src, dst]["belief"]

    def to_state_dict(self) -> Dict[str, Any]:
        edges = []
        for src, dst, data in self.g.edges(data=True):
            belief: EdgeBelief = data["belief"]
            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "belief": belief.to_state_dict(),
                }
            )
        return {"edges": edges}

    def restore_edges(self, state: Dict[str, Any]) -> None:
        for item in state.get("edges", []):
            src = str(item["src"])
            dst = str(item["dst"])
            if src not in self.g or dst not in self.g:
                continue
            self.g.add_edge(src, dst, belief=EdgeBelief.from_state_dict(item["belief"]))


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.vecs: List[np.ndarray] = []
        self.kinds: List[str] = []

    def add(self, item_id: str, vec: np.ndarray, kind: str) -> None:
        vector = vec.astype(np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"vector dim mismatch: {vector.shape} vs {(self.dim,)}")
        self.ids.append(item_id)
        self.vecs.append(vector)
        self.kinds.append(kind)

    def remove(self, item_id: str) -> None:
        if item_id not in self.ids:
            return
        idx = self.ids.index(item_id)
        self.ids.pop(idx)
        self.vecs.pop(idx)
        self.kinds.pop(idx)

    def search(self, q: np.ndarray, k: int = 10, kind: Optional[str] = None) -> List[Tuple[str, float]]:
        if not self.vecs:
            return []
        vector = q.astype(np.float32)
        hits: List[Tuple[str, float]] = []
        for item_id, candidate, candidate_kind in zip(self.ids, self.vecs, self.kinds):
            if kind is not None and candidate_kind != kind:
                continue
            hits.append((item_id, cosine_sim(vector, candidate)))
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:k]

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "ids": list(self.ids),
            "kinds": list(self.kinds),
            "vecs": [vec.astype(np.float32).tolist() for vec in self.vecs],
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "VectorIndex":
        obj = cls(dim=int(data["dim"]))
        obj.ids = [str(item) for item in data.get("ids", [])]
        obj.kinds = [str(item) for item in data.get("kinds", [])]
        obj.vecs = [np.asarray(vec, dtype=np.float32) for vec in data.get("vecs", [])]
        return obj


class CRPAssigner:
    def __init__(self, alpha: float = 1.0, seed: int = 0):
        self.alpha = alpha
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, logps: Dict[str, float]) -> Tuple[Optional[str], Dict[str, float]]:
        logs = dict(logps)
        logs["__new__"] = math.log(self.alpha)
        keys = list(logs.keys())
        probs = softmax([logs[k] for k in keys])
        choice = str(self.rng.choice(keys, p=np.asarray(probs, dtype=np.float64)))
        post = {k: p for k, p in zip(keys, probs)}
        if choice == "__new__":
            return None, post
        return choice, post

    def to_state_dict(self) -> Dict[str, Any]:
        return {"alpha": float(self.alpha), "seed": self._seed}

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "CRPAssigner":
        return cls(alpha=float(data["alpha"]), seed=int(data.get("seed", 0)))


class StoryModel:
    def __init__(self, metric: LowRankMetric):
        self.metric = metric

    def loglik(self, plot: Plot, story: StoryArc) -> float:
        ll_sem = 0.0
        if story.centroid is not None:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            variance = max(story.dist_var(), 1e-3)
            ll_sem = -0.5 * d2 / variance

        ll_time = 0.0
        if story.plot_ids:
            gap = max(0.0, plot.ts - story.updated_ts)
            tau = story.gap_mean_safe()
            lam = 1.0 / max(tau, 1e-6)
            ll_time = math.log(lam + 1e-12) - lam * gap

        ll_actor = 0.0
        beta = 1.0
        total = sum(story.actor_counts.values())
        denom = total + beta * max(len(story.actor_counts), 1)
        for actor in plot.actors:
            ll_actor += math.log(story.actor_counts.get(actor, 0) + beta) - math.log(denom + 1e-12)

        if story.source_counts:
            total_sources = sum(story.source_counts.values())
            ll_source = math.log(story.source_counts.get(plot.source, 0) + 1.0) - math.log(
                total_sources + len(story.source_counts) + 1e-12
            )
        else:
            ll_source = 0.0

        return ll_sem + ll_time + ll_actor + ll_source


class ThemeModel:
    def __init__(self, metric: LowRankMetric):
        self.metric = metric

    def loglik(self, story: StoryArc, theme: Theme) -> float:
        if theme.prototype is None or story.centroid is None:
            return 0.0
        d2 = self.metric.d2(story.centroid, theme.prototype)
        return -0.5 * d2


class FieldRetriever:
    def __init__(self, metric: LowRankMetric, vindex: VectorIndex, graph: MemoryGraph):
        self.metric = metric
        self.vindex = vindex
        self.graph = graph

    def _payload_vec(self, payload: Any) -> Optional[np.ndarray]:
        if hasattr(payload, "embedding"):
            return payload.embedding
        if hasattr(payload, "centroid"):
            return payload.centroid
        if hasattr(payload, "prototype"):
            return payload.prototype
        return None

    def _mean_shift(
        self,
        x0: np.ndarray,
        candidates: List[Tuple[str, np.ndarray, float]],
        steps: int = 8,
    ) -> List[np.ndarray]:
        if not candidates:
            return [x0]
        x = x0.copy()
        path = [x.copy()]
        for _ in range(steps):
            d2s = [self.metric.d2(x, vec) for _, vec, _ in candidates]
            sigma2 = float(np.median(d2s)) + 1e-6
            logits = [-(d2 / (2.0 * sigma2)) + mass for d2, (_, _, mass) in zip(d2s, candidates)]
            weights = softmax(logits)
            new_x = np.zeros_like(x)
            for weight, (_, vec, _) in zip(weights, candidates):
                new_x += weight * vec
            x = l2_normalize(new_x)
            path.append(x.copy())
        return path

    def _pagerank(
        self,
        personalization: Dict[str, float],
        damping: float = 0.85,
        max_iter: int = 60,
    ) -> Dict[str, float]:
        graph = self.graph.g
        personalized = {node: value for node, value in personalization.items() if node in graph}
        if not personalized:
            return {}
        weighted = nx.DiGraph()
        weighted.add_nodes_from(graph.nodes())
        for src, dst, data in graph.edges(data=True):
            belief: EdgeBelief = data["belief"]
            weighted.add_edge(src, dst, w=max(1e-6, belief.mean()))
        return nx.pagerank(
            weighted,
            alpha=damping,
            personalization=personalized,
            weight="w",
            max_iter=max_iter,
        )

    def retrieve(
        self,
        query_text: str,
        embedder: EmbeddingProvider,
        state: IdentityState,
        kinds: Tuple[str, ...],
        k: int = 5,
    ) -> RetrievalTrace:
        query_emb = embedder.embed(query_text)
        phase_vec = state.signed_traits()

        candidates: List[Tuple[str, np.ndarray, float]] = []
        for kind in kinds:
            for item_id, sim in self.vindex.search(query_emb, k=60, kind=kind):
                if item_id not in self.graph.g:
                    continue
                payload = self.graph.payload(item_id)
                vec = self._payload_vec(payload)
                if vec is None:
                    continue
                phase_bonus = 0.0
                if isinstance(payload, Plot):
                    phase_bonus = 0.15 * float(np.dot(payload.frame.trait_vector(), phase_vec))
                mass = float(payload.mass()) if hasattr(payload, "mass") else 0.0
                candidates.append((item_id, vec, mass + phase_bonus + sim))

        path = self._mean_shift(query_emb, candidates, steps=8)
        attractor = path[-1]

        personalization: Dict[str, float] = {}
        for kind in kinds:
            for item_id, sim in self.vindex.search(attractor, k=80, kind=kind):
                personalization[item_id] = max(personalization.get(item_id, 0.0), sim)

        pr = self._pagerank(personalization, damping=0.85, max_iter=60)
        ranked: List[Tuple[str, float, str]] = []
        for item_id, score in pr.items():
            kind = self.graph.kind(item_id)
            if kind not in kinds:
                continue
            payload = self.graph.payload(item_id)
            bonus = float(payload.mass()) if hasattr(payload, "mass") else 0.0
            if isinstance(payload, Plot):
                bonus += 0.10 * float(np.dot(payload.frame.trait_vector(), phase_vec))
                bonus += 0.03 * payload.confidence
            ranked.append((item_id, float(score) + 1e-3 * bonus, kind))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return RetrievalTrace(query=query_text, query_emb=query_emb, attractor_path=path, ranked=ranked[:k])
