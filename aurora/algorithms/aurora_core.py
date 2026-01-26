"""
AURORA Memory — Adaptive Uncertainty-Reducing, Resource-Optimal, Recursive Autobiographical Memory
===============================================================================================

Reference implementation (single-file) focusing on algorithmic correctness and extensibility.

Design goal: zero hard-coded thresholds/weights for core behaviors.
Instead of:
- if score > 0.3: encode
- 0.25*actor + 0.30*theme + ...
We use:
- Bayesian / stochastic decision policies (Thompson sampling)
- Nonparametric clustering (CRP) for story/theme counts
- Learnable metric + edge beliefs updated from downstream success

Dependencies:
    numpy, networkx

This file is intended to be copied into your project and incrementally replaced with:
- real embedding model
- LLM-based extractors/summarizers
- persistent stores + ANN indexes
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import math
import random
import time
import uuid

import numpy as np
import networkx as nx

# -----------------------------------------------------------------------------
# Deterministic IDs for event-sourcing (production requirement)
# -----------------------------------------------------------------------------
AURORA_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "aurora-memory")


def det_id(kind: str, seed: str) -> str:
    """Deterministic UUID (stable across restarts/replays)."""
    return str(uuid.uuid5(AURORA_NAMESPACE, f"{kind}:{seed}"))



# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def now_ts() -> float:
    return time.time()


def stable_hash(text: str) -> int:
    """Deterministic hash across runs (unlike Python's built-in salted hash)."""
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return (v / n).astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


def sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    Z = sum(exps) + 1e-12
    return [e / Z for e in exps]


# -----------------------------------------------------------------------------
# Core data structures
# -----------------------------------------------------------------------------

@dataclass
class Plot:
    """Atomic interaction/event memory.

    In production, this should be extracted from interaction using an LLM or a structured parser.
    Here we keep it generic and embed the full interaction text.
    """
    id: str
    ts: float
    text: str
    actors: Tuple[str, ...]
    embedding: np.ndarray

    # Signals computed online (no fixed mixing weights)
    surprise: float = 0.0          # -log p(x) under OnlineKDE
    pred_error: float = 0.0        # mismatch with best story predictor
    redundancy: float = 0.0        # max similarity to existing plots
    goal_relevance: float = 0.0    # similarity to query/goal context
    tension: float = 0.0           # free-energy proxy
    # Assignment
    story_id: Optional[str] = None

    # Usage stats -> "mass" emerges
    access_count: int = 0
    last_access_ts: float = field(default_factory=now_ts)
    status: Literal["active", "absorbed", "archived"] = "active"

    def mass(self) -> float:
        """Emergent inertia: increases with access frequency, decreases with age."""
        age = max(1.0, now_ts() - self.ts)
        freshness = 1.0 / math.log1p(age)
        return freshness * math.log1p(self.access_count + 1)

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "ts": self.ts,
            "text": self.text,
            "actors": list(self.actors),
            "embedding": self.embedding.tolist(),
            "surprise": self.surprise,
            "pred_error": self.pred_error,
            "redundancy": self.redundancy,
            "goal_relevance": self.goal_relevance,
            "tension": self.tension,
            "story_id": self.story_id,
            "access_count": self.access_count,
            "last_access_ts": self.last_access_ts,
            "status": self.status,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "Plot":
        """Reconstruct from state dict."""
        return cls(
            id=d["id"],
            ts=d["ts"],
            text=d["text"],
            actors=tuple(d["actors"]),
            embedding=np.array(d["embedding"], dtype=np.float32),
            surprise=d.get("surprise", 0.0),
            pred_error=d.get("pred_error", 0.0),
            redundancy=d.get("redundancy", 0.0),
            goal_relevance=d.get("goal_relevance", 0.0),
            tension=d.get("tension", 0.0),
            story_id=d.get("story_id"),
            access_count=d.get("access_count", 0),
            last_access_ts=d.get("last_access_ts", now_ts()),
            status=d.get("status", "active"),
        )


@dataclass
class StoryArc:
    """Mesoscale narrative unit: a self-organizing cluster of plots."""
    id: str
    created_ts: float
    updated_ts: float
    plot_ids: List[str] = field(default_factory=list)

    # Online generative parameters
    centroid: Optional[np.ndarray] = None
    # Stats for semantic dispersion and temporal gaps
    dist_mean: float = 0.0
    dist_m2: float = 0.0
    dist_n: int = 0
    gap_mean: float = 0.0
    gap_m2: float = 0.0
    gap_n: int = 0

    actor_counts: Dict[str, int] = field(default_factory=dict)
    tension_curve: List[float] = field(default_factory=list)

    status: Literal["developing", "resolved", "abandoned"] = "developing"
    reference_count: int = 0

    # --- running stats helpers ---
    def _update_stats(self, name: str, x: float) -> None:
        if name == "dist":
            self.dist_n += 1
            delta = x - self.dist_mean
            self.dist_mean += delta / self.dist_n
            self.dist_m2 += delta * (x - self.dist_mean)
        elif name == "gap":
            self.gap_n += 1
            delta = x - self.gap_mean
            self.gap_mean += delta / self.gap_n
            self.gap_m2 += delta * (x - self.gap_mean)
        else:
            raise ValueError(name)

    def dist_var(self) -> float:
        return self.dist_m2 / (self.dist_n - 1) if self.dist_n > 1 else 1.0

    def gap_mean_safe(self, default: float = 3600.0) -> float:
        return self.gap_mean if self.gap_n > 0 and self.gap_mean > 0 else default

    def activity_probability(self, ts: Optional[float] = None) -> float:
        """Probability the story is still active under a learned temporal hazard model.

        If a story usually gets updates every ~gap_mean seconds, then being idle >> gap_mean
        should reduce activity probability smoothly (not via a fixed threshold).
        """
        ts = ts or now_ts()
        idle = max(0.0, ts - self.updated_ts)
        tau = self.gap_mean_safe()
        # survival function of exponential: P(active) ~ exp(-idle/tau)
        return math.exp(-idle / max(tau, 1e-6))

    def mass(self) -> float:
        """Emergent importance at story level."""
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        size = math.log1p(len(self.plot_ids))
        return freshness * (size + math.log1p(self.reference_count + 1))

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "plot_ids": self.plot_ids,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "dist_mean": self.dist_mean,
            "dist_m2": self.dist_m2,
            "dist_n": self.dist_n,
            "gap_mean": self.gap_mean,
            "gap_m2": self.gap_m2,
            "gap_n": self.gap_n,
            "actor_counts": self.actor_counts,
            "tension_curve": self.tension_curve,
            "status": self.status,
            "reference_count": self.reference_count,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "StoryArc":
        """Reconstruct from state dict."""
        centroid = d.get("centroid")
        return cls(
            id=d["id"],
            created_ts=d["created_ts"],
            updated_ts=d["updated_ts"],
            plot_ids=d.get("plot_ids", []),
            centroid=np.array(centroid, dtype=np.float32) if centroid is not None else None,
            dist_mean=d.get("dist_mean", 0.0),
            dist_m2=d.get("dist_m2", 0.0),
            dist_n=d.get("dist_n", 0),
            gap_mean=d.get("gap_mean", 0.0),
            gap_m2=d.get("gap_m2", 0.0),
            gap_n=d.get("gap_n", 0),
            actor_counts=d.get("actor_counts", {}),
            tension_curve=d.get("tension_curve", []),
            status=d.get("status", "developing"),
            reference_count=d.get("reference_count", 0),
        )


@dataclass
class Theme:
    """Macroscale stable pattern (attractor) emerging from stories."""
    id: str
    created_ts: float
    updated_ts: float
    story_ids: List[str] = field(default_factory=list)
    prototype: Optional[np.ndarray] = None

    # Epistemic confidence as Beta posterior (evidence from applications)
    a: float = 1.0
    b: float = 1.0

    name: str = ""
    description: str = ""
    theme_type: Literal["pattern", "lesson", "preference", "causality", "capability", "limitation"] = "pattern"

    def confidence(self) -> float:
        return self.a / (self.a + self.b)

    def update_evidence(self, success: bool) -> None:
        if success:
            self.a += 1.0
        else:
            self.b += 1.0
        self.updated_ts = now_ts()

    def mass(self) -> float:
        age = max(1.0, now_ts() - self.updated_ts)
        freshness = 1.0 / math.log1p(age)
        return freshness * math.log1p(len(self.story_ids) + 1) * self.confidence()

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "story_ids": self.story_ids,
            "prototype": self.prototype.tolist() if self.prototype is not None else None,
            "a": self.a,
            "b": self.b,
            "name": self.name,
            "description": self.description,
            "theme_type": self.theme_type,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "Theme":
        """Reconstruct from state dict."""
        prototype = d.get("prototype")
        return cls(
            id=d["id"],
            created_ts=d["created_ts"],
            updated_ts=d["updated_ts"],
            story_ids=d.get("story_ids", []),
            prototype=np.array(prototype, dtype=np.float32) if prototype is not None else None,
            a=d.get("a", 1.0),
            b=d.get("b", 1.0),
            name=d.get("name", ""),
            description=d.get("description", ""),
            theme_type=d.get("theme_type", "pattern"),
        )


# -----------------------------------------------------------------------------
# Learnable components (no threshold-based behavior)
# -----------------------------------------------------------------------------

class HashEmbedding:
    """Dependency-free embedding model for local testing.

    Replace with a real embedding model in production.
    """
    def __init__(self, dim: int = 384, seed: int = 7):
        self.dim = dim
        self.seed = seed

    def embed(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(stable_hash(text) ^ self.seed)
        v = rng.normal(size=self.dim).astype(np.float32)
        return l2_normalize(v)


class OnlineKDE:
    """Kernel density estimator in embedding space.

    Used to compute surprise = -log p(x) without any similarity thresholds.
    """
    def __init__(self, dim: int, reservoir: int = 4096, k_sigma: int = 25, seed: int = 0):
        self.dim = dim
        self.reservoir = reservoir
        self.k_sigma = k_sigma
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._vecs: List[np.ndarray] = []

    def add(self, x: np.ndarray) -> None:
        x = x.astype(np.float32)
        if len(self._vecs) < self.reservoir:
            self._vecs.append(x)
        else:
            # reservoir sampling (capacity-limited memory)
            j = int(self.rng.integers(0, len(self._vecs) + 1))
            if j < len(self._vecs):
                self._vecs[j] = x

    def _sigma(self, x: np.ndarray) -> float:
        if not self._vecs:
            return 1.0
        dists = [float(np.linalg.norm(x - v)) for v in self._vecs]
        dists.sort()
        k = min(self.k_sigma, len(dists))
        med = float(np.median(dists[:k])) if k > 0 else float(np.median(dists))
        return med + 1e-6

    def log_density(self, x: np.ndarray) -> float:
        if not self._vecs:
            # weak prior: very low density
            return -10.0
        sigma = self._sigma(x)
        inv2 = 1.0 / (2.0 * sigma * sigma)
        vals = []
        for v in self._vecs:
            d2 = float(np.dot(x - v, x - v))
            vals.append(math.exp(-d2 * inv2))
        p = sum(vals) / len(vals)
        return math.log(p + 1e-12)

    def surprise(self, x: np.ndarray) -> float:
        return -self.log_density(x)

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "dim": self.dim,
            "reservoir": self.reservoir,
            "k_sigma": self.k_sigma,
            "seed": self._seed,
            "vecs": [v.tolist() for v in self._vecs],
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "OnlineKDE":
        """Reconstruct from state dict."""
        obj = cls(
            dim=d["dim"],
            reservoir=d["reservoir"],
            k_sigma=d["k_sigma"],
            seed=d["seed"],
        )
        obj._vecs = [np.array(v, dtype=np.float32) for v in d.get("vecs", [])]
        return obj


class LowRankMetric:
    """Low-rank Mahalanobis metric: d(x,y)^2 = ||L(x-y)||^2.

    Interpretable as learning a task- and user-adapted "information geometry" metric
    from retrieval feedback.
    
    Parameter stability:
        - window_size: Sliding window for Adagrad accumulator reset.
          Every `window_size` updates, the accumulator G is rescaled to prevent
          the learning rate from decaying to near-zero over time.
        - decay_factor: Applied to G during periodic recomputation (default 0.5).
    """
    def __init__(self, dim: int, rank: int = 64, seed: int = 0, window_size: int = 10000, decay_factor: float = 0.5):
        self.dim = dim
        self.rank = min(rank, dim)
        self._seed = seed
        self.window_size = window_size
        self.decay_factor = decay_factor
        
        rng = np.random.default_rng(seed)
        self.L = np.eye(dim, dtype=np.float32)[: self.rank].copy()
        self.L += (0.01 * rng.normal(size=self.L.shape)).astype(np.float32)

        self.G = np.zeros_like(self.L)  # Adagrad accumulator
        self.t = 0
        
        # Statistics for monitoring
        self._total_loss = 0.0
        self._update_count = 0

    def d2(self, x: np.ndarray, y: np.ndarray) -> float:
        z = (x - y).astype(np.float32)
        p = self.L @ z
        return float(np.dot(p, p))

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1.0 / (1.0 + self.d2(x, y))

    def update_triplet(self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 1.0) -> float:
        """Online OASIS-like update with Adagrad.

        margin is not a threshold on similarity; it's a geometric separation unit.
        
        Includes sliding window mechanism: periodically rescales the Adagrad
        accumulator to prevent learning rate from vanishing.
        """
        self.t += 1
        ap = (anchor - positive).astype(np.float32)
        an = (anchor - negative).astype(np.float32)
        Lap = self.L @ ap
        Lan = self.L @ an
        dap = float(np.dot(Lap, Lap))
        dan = float(np.dot(Lan, Lan))
        loss = max(0.0, margin + dap - dan)
        if loss <= 0:
            return 0.0

        grad = 2.0 * (np.outer(Lap, ap) - np.outer(Lan, an)).astype(np.float32)
        self.G += grad * grad
        # self-tuning learning rate: decays with t automatically
        base = 1.0 / math.sqrt(self.t + 1.0)
        step = base * grad / (np.sqrt(self.G) + 1e-8)
        self.L -= step
        
        # Track statistics
        self._total_loss += loss
        self._update_count += 1
        
        # Sliding window: periodically rescale G to maintain plasticity
        # This prevents the Adagrad accumulator from growing unboundedly
        # which would cause the learning rate to approach zero
        if self.t > 0 and self.t % self.window_size == 0:
            self._rescale_accumulator()
        
        return float(loss)

    def _rescale_accumulator(self) -> None:
        """Rescale the Adagrad accumulator to maintain learning capacity.
        
        This implements a "soft reset" that preserves learned structure
        while preventing the accumulator from growing too large.
        """
        self.G *= self.decay_factor

    def average_loss(self) -> float:
        """Return average triplet loss over all updates."""
        return self._total_loss / self._update_count if self._update_count > 0 else 0.0

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "dim": self.dim,
            "rank": self.rank,
            "seed": self._seed,
            "window_size": self.window_size,
            "decay_factor": self.decay_factor,
            "L": self.L.tolist(),
            "G": self.G.tolist(),
            "t": self.t,
            "total_loss": self._total_loss,
            "update_count": self._update_count,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "LowRankMetric":
        """Reconstruct from state dict."""
        obj = cls(
            dim=d["dim"],
            rank=d["rank"],
            seed=d.get("seed", 0),
            window_size=d.get("window_size", 10000),
            decay_factor=d.get("decay_factor", 0.5),
        )
        obj.L = np.array(d["L"], dtype=np.float32)
        obj.G = np.array(d["G"], dtype=np.float32)
        obj.t = d["t"]
        obj._total_loss = d.get("total_loss", 0.0)
        obj._update_count = d.get("update_count", 0)
        return obj


class ThompsonBernoulliGate:
    """Stochastic encode policy with Thompson sampling.

    We do not decide by "score > threshold". We sample a parameter vector w and
    encode with probability sigmoid(w·x). The mapping is learned from delayed rewards.
    
    Parameter stability:
        - forgetting_factor (lambda): Prevents precision from infinite accumulation.
          prec = lambda * prec + grad * grad, where lambda=0.99 ensures ~1% decay per update.
        - This keeps the system "plastic" and able to adapt to distribution shifts.
    """
    def __init__(self, feature_dim: int, seed: int = 0, forgetting_factor: float = 0.99):
        self.d = feature_dim
        self._seed = seed
        self.lambda_ = forgetting_factor  # Forgetting factor for precision
        self.rng = np.random.default_rng(seed)

        self.w_mean = np.zeros(self.d, dtype=np.float32)
        self.prec = np.ones(self.d, dtype=np.float32) * 1e-2  # weak precision
        self.grad2 = np.zeros(self.d, dtype=np.float32)  # RMS

        self.t = 0
        
        # Statistics tracking for monitoring
        self._encode_count = 0
        self._skip_count = 0

    def _sample_w(self) -> np.ndarray:
        std = np.sqrt(1.0 / (self.prec + 1e-9))
        return self.w_mean + self.rng.normal(size=self.d).astype(np.float32) * std

    def prob(self, x: np.ndarray) -> float:
        w = self._sample_w()
        return sigmoid(float(np.dot(w, x)))

    def decide(self, x: np.ndarray) -> bool:
        result = bool(self.rng.random() < self.prob(x))
        if result:
            self._encode_count += 1
        else:
            self._skip_count += 1
        return result

    def update(self, x: np.ndarray, reward: float) -> None:
        """Bandit update: reward in [-1, 1] from downstream task success.
        
        Uses forgetting factor to prevent precision from accumulating indefinitely,
        which would cause variance to approach zero and freeze the policy.
        """
        self.t += 1
        y = 1.0 if reward > 0 else 0.0
        p = sigmoid(float(np.dot(self.w_mean, x)))
        grad = (y - p) * x  # ascent

        # RMS with self-tuning step size
        self.grad2 = 0.99 * self.grad2 + 0.01 * (grad * grad)
        base = 1.0 / math.sqrt(self.t + 1.0)
        step = base * grad / (np.sqrt(self.grad2) + 1e-6)
        self.w_mean += step
        
        # Apply forgetting factor to prevent precision from growing unboundedly
        # This ensures the system remains "plastic" and can adapt to distribution shifts
        self.prec = self.lambda_ * self.prec + grad * grad

    def pass_rate(self) -> float:
        """Return the gate pass rate (encode / total decisions)."""
        total = self._encode_count + self._skip_count
        return self._encode_count / total if total > 0 else 0.5

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "d": self.d,
            "seed": self._seed,
            "lambda": self.lambda_,
            "w_mean": self.w_mean.tolist(),
            "prec": self.prec.tolist(),
            "grad2": self.grad2.tolist(),
            "t": self.t,
            "encode_count": self._encode_count,
            "skip_count": self._skip_count,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "ThompsonBernoulliGate":
        """Reconstruct from state dict."""
        obj = cls(
            feature_dim=d["d"],
            seed=d.get("seed", 0),
            forgetting_factor=d.get("lambda", 0.99),
        )
        obj.w_mean = np.array(d["w_mean"], dtype=np.float32)
        obj.prec = np.array(d["prec"], dtype=np.float32)
        obj.grad2 = np.array(d["grad2"], dtype=np.float32)
        obj.t = d["t"]
        obj._encode_count = d.get("encode_count", 0)
        obj._skip_count = d.get("skip_count", 0)
        return obj


# -----------------------------------------------------------------------------
# Graph memory primitives
# -----------------------------------------------------------------------------

@dataclass
class EdgeBelief:
    """Edge helpfulness posterior: Beta(a,b)."""
    edge_type: str
    a: float = 1.0
    b: float = 1.0
    use_count: int = 0
    last_used_ts: float = field(default_factory=now_ts)

    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def update(self, success: bool) -> None:
        self.use_count += 1
        self.last_used_ts = now_ts()
        if success:
            self.a += 1.0
        else:
            self.b += 1.0

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "edge_type": self.edge_type,
            "a": self.a,
            "b": self.b,
            "use_count": self.use_count,
            "last_used_ts": self.last_used_ts,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "EdgeBelief":
        """Reconstruct from state dict."""
        return cls(
            edge_type=d["edge_type"],
            a=d["a"],
            b=d["b"],
            use_count=d["use_count"],
            last_used_ts=d["last_used_ts"],
        )


class MemoryGraph:
    """Typed node graph with probabilistic edge strengths."""
    def __init__(self):
        self.g = nx.DiGraph()

    def add_node(self, node_id: str, kind: str, payload: Any) -> None:
        self.g.add_node(node_id, kind=kind, payload=payload)

    def kind(self, node_id: str) -> str:
        return self.g.nodes[node_id]["kind"]

    def payload(self, node_id: str) -> Any:
        return self.g.nodes[node_id]["payload"]

    def ensure_edge(self, src: str, dst: str, edge_type: str) -> None:
        if self.g.has_edge(src, dst):
            return
        self.g.add_edge(src, dst, belief=EdgeBelief(edge_type=edge_type))

    def edge_belief(self, src: str, dst: str) -> EdgeBelief:
        return self.g.edges[src, dst]["belief"]

    def nodes_of_kind(self, kind: str) -> List[str]:
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
    def from_state_dict(cls, d: Dict[str, Any], payloads: Dict[str, Any] = None) -> "MemoryGraph":
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


class VectorIndex:
    """Brute-force vector index with kind filtering.

    Replace with FAISS/pgvector for production.
    
    DEPRECATED: Use aurora.storage.vector_store.VectorStore instead for production.
    This class is kept for backward compatibility and testing.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.vecs: List[np.ndarray] = []
        self.kinds: List[str] = []

    def add(self, _id: str, vec: np.ndarray, kind: str) -> None:
        vec = vec.astype(np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"vector dim mismatch: {vec.shape} vs {(self.dim,)}")
        self.ids.append(_id)
        self.vecs.append(vec)
        self.kinds.append(kind)

    def remove(self, _id: str) -> None:
        if _id not in self.ids:
            return
        i = self.ids.index(_id)
        self.ids.pop(i)
        self.vecs.pop(i)
        self.kinds.pop(i)

    def search(self, q: np.ndarray, k: int = 10, kind: Optional[str] = None) -> List[Tuple[str, float]]:
        if not self.vecs:
            return []
        q = q.astype(np.float32)
        hits: List[Tuple[str, float]] = []
        for _id, v, kd in zip(self.ids, self.vecs, self.kinds):
            if kind is not None and kd != kind:
                continue
            hits.append((_id, cosine_sim(q, v)))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits[:k]

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "dim": self.dim,
            "ids": self.ids,
            "vecs": [v.tolist() for v in self.vecs],
            "kinds": self.kinds,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "VectorIndex":
        """Reconstruct from state dict."""
        obj = cls(dim=d["dim"])
        obj.ids = d["ids"]
        obj.vecs = [np.array(v, dtype=np.float32) for v in d["vecs"]]
        obj.kinds = d["kinds"]
        return obj


# -----------------------------------------------------------------------------
# Nonparametric hierarchical assignment (CRP)
# -----------------------------------------------------------------------------

class CRPAssigner:
    """Generic CRP-like assigner for (item -> cluster) with probabilistic sampling."""

    def __init__(self, alpha: float = 1.0, seed: int = 0):
        self.alpha = alpha
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def sample(self, logps: Dict[str, float]) -> Tuple[Optional[str], Dict[str, float]]:
        """Return chosen existing cluster id OR None meaning 'new cluster'."""
        # Add new cluster option
        logs = dict(logps)
        logs["__new__"] = math.log(self.alpha)
        keys = list(logs.keys())
        probs = softmax([logs[k] for k in keys])
        choice = self.rng.choice(keys, p=np.array(probs, dtype=np.float64))
        post = {k: p for k, p in zip(keys, probs)}
        if choice == "__new__":
            return None, post
        return choice, post

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "alpha": self.alpha,
            "seed": self._seed,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "CRPAssigner":
        """Reconstruct from state dict."""
        return cls(alpha=d["alpha"], seed=d.get("seed", 0))


class StoryModel:
    """Likelihood model p(plot | story) with interpretable factors.

    No fixed weights: all components are generative log-likelihood terms.
    """
    def __init__(self, metric: LowRankMetric):
        self.metric = metric

    def loglik(self, plot: Plot, story: StoryArc) -> float:
        # semantic likelihood: Gaussian in metric space using story's dispersion
        ll_sem = 0.0
        if story.centroid is not None:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            var = max(story.dist_var(), 1e-3)
            ll_sem = -0.5 * d2 / var

        # temporal likelihood: exponential around learned typical gap
        ll_time = 0.0
        if story.plot_ids:
            gap = max(0.0, plot.ts - story.updated_ts)
            tau = story.gap_mean_safe()
            lam = 1.0 / max(tau, 1e-6)
            ll_time = math.log(lam + 1e-12) - lam * gap

        # actor likelihood: Dirichlet-multinomial predictive
        ll_actor = 0.0
        beta = 1.0
        total = sum(story.actor_counts.values())
        denom = total + beta * max(len(story.actor_counts), 1)
        for a in plot.actors:
            ll_actor += math.log(story.actor_counts.get(a, 0) + beta) - math.log(denom + 1e-12)

        return ll_sem + ll_time + ll_actor


class ThemeModel:
    """Likelihood model p(story | theme)."""
    def __init__(self, metric: LowRankMetric):
        self.metric = metric

    def loglik(self, story: StoryArc, theme: Theme) -> float:
        if theme.prototype is None or story.centroid is None:
            return 0.0
        d2 = self.metric.d2(story.centroid, theme.prototype)
        # theme dispersion is not stored; we use a robust default scale = 1
        return -0.5 * d2


# -----------------------------------------------------------------------------
# Field retrieval: attractor tracing + graph diffusion
# -----------------------------------------------------------------------------

@dataclass
class RetrievalTrace:
    query: str
    query_emb: np.ndarray
    attractor_path: List[np.ndarray]
    ranked: List[Tuple[str, float, str]]  # (id, score, kind)


@dataclass
class EvolutionSnapshot:
    """Read-only snapshot of memory state for background evolution.
    
    Captures everything needed to compute evolution without modifying state.
    """
    # Story data
    story_ids: List[str]
    story_statuses: Dict[str, str]
    story_centroids: Dict[str, Optional[np.ndarray]]
    story_tension_curves: Dict[str, List[float]]
    story_updated_ts: Dict[str, float]
    story_gap_means: Dict[str, float]
    
    # Theme data
    theme_ids: List[str]
    theme_story_counts: Dict[str, int]
    theme_prototypes: Dict[str, Optional[np.ndarray]]
    
    # CRP parameters
    crp_theme_alpha: float
    
    # RNG state for reproducibility
    rng_state: Dict[str, Any]


@dataclass
class EvolutionPatch:
    """Computed changes from evolution, to be applied atomically.
    
    Represents a diff that can be applied to the memory state.
    """
    # Story status changes: story_id -> new_status
    status_changes: Dict[str, str]
    
    # Theme assignments: [(story_id, theme_id)]
    theme_assignments: List[Tuple[str, str]]
    
    # New themes to create: [(theme_id, prototype)]
    new_themes: List[Tuple[str, np.ndarray]]


class FieldRetriever:
    """Two-stage retrieval:

    1) Continuous-space attractor tracing (mean-shift in learned metric space).
       This yields a context-adaptive "mode" representing what the query is pulling from memory.

    2) Discrete graph diffusion (personalized PageRank) seeded by vector hits around the attractor.
       Edge probabilities are learned Beta posteriors.
    """

    def __init__(self, metric: LowRankMetric, vindex: VectorIndex, graph: MemoryGraph):
        self.metric = metric
        self.vindex = vindex
        self.graph = graph

    def _mean_shift(self, x0: np.ndarray, candidates: List[Tuple[str, np.ndarray, float]], steps: int = 8) -> List[np.ndarray]:
        """Mean shift path. candidates: list of (id, vec, mass)."""
        if not candidates:
            return [x0]
        x = x0.copy()
        path = [x.copy()]
        # dynamic bandwidth: median distance to candidates in current metric
        for _ in range(steps):
            d2s = [self.metric.d2(x, v) for _, v, _ in candidates]
            # bandwidth as robust scale
            sigma2 = float(np.median(d2s)) + 1e-6
            logits = [-(d2 / (2.0 * sigma2)) + m for d2, (_, _, m) in zip(d2s, candidates)]
            w = softmax(logits)
            new_x = np.zeros_like(x)
            for wi, (_, v, _) in zip(w, candidates):
                new_x += wi * v
            x = l2_normalize(new_x)
            path.append(x.copy())
        return path

    def _pagerank(self, personalization: Dict[str, float], damping: float = 0.85, max_iter: int = 50) -> Dict[str, float]:
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

    def retrieve(self, query_text: str, embed: HashEmbedding, kinds: Tuple[str, ...], k: int = 5) -> RetrievalTrace:
        q = embed.embed(query_text)
        # 1) seed candidates from vector index (plots + stories + themes)
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

        # 2) continuous attractor tracing
        path = self._mean_shift(q, candidates, steps=8)
        attractor = path[-1]

        # 3) reseed around attractor and diffuse on graph
        personalization: Dict[str, float] = {}
        for kind in kinds:
            for _id, sim in self.vindex.search(attractor, k=60, kind=kind):
                personalization[_id] = max(personalization.get(_id, 0.0), sim)

        pr = self._pagerank(personalization, damping=0.85, max_iter=60)

        # 4) rank with emergent masses (no fixed weights; only small scale for tie-breaking)
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


# -----------------------------------------------------------------------------
# Main system: AURORA Memory
# -----------------------------------------------------------------------------

@dataclass
class MemoryConfig:
    dim: int = 384
    metric_rank: int = 64

    # bounded memory pressures (resource constraints are first-principles reality)
    max_plots: int = 5000
    kde_reservoir: int = 4096

    # CRP concentration priors
    story_alpha: float = 1.0
    theme_alpha: float = 0.5

    # encode gate feature dimension
    gate_feature_dim: int = 6

    # retrieval preferences
    retrieval_kinds: Tuple[str, ...] = ("theme", "story", "plot")

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "dim": self.dim,
            "metric_rank": self.metric_rank,
            "max_plots": self.max_plots,
            "kde_reservoir": self.kde_reservoir,
            "story_alpha": self.story_alpha,
            "theme_alpha": self.theme_alpha,
            "gate_feature_dim": self.gate_feature_dim,
            "retrieval_kinds": list(self.retrieval_kinds),
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "MemoryConfig":
        """Reconstruct from state dict."""
        return cls(
            dim=d.get("dim", 384),
            metric_rank=d.get("metric_rank", 64),
            max_plots=d.get("max_plots", 5000),
            kde_reservoir=d.get("kde_reservoir", 4096),
            story_alpha=d.get("story_alpha", 1.0),
            theme_alpha=d.get("theme_alpha", 0.5),
            gate_feature_dim=d.get("gate_feature_dim", 6),
            retrieval_kinds=tuple(d.get("retrieval_kinds", ["theme", "story", "plot"])),
        )


class AuroraMemory:
    """AURORA Memory: emergent narrative memory from first principles.

    Key APIs:
        ingest(interaction_text, actors, context_text) -> Plot (may or may not be stored)
        query(text, k) -> RetrievalTrace
        feedback_retrieval(query_text, chosen_id, success) -> update beliefs
        evolve() -> consolidate plots->stories->themes, manage pressure, update statuses
    """

    def __init__(self, cfg: MemoryConfig = MemoryConfig(), seed: int = 0):
        self.cfg = cfg
        self._seed = seed
        self.rng = np.random.default_rng(seed)

        # learnable primitives
        self.embedder = HashEmbedding(dim=cfg.dim, seed=seed)
        self.kde = OnlineKDE(dim=cfg.dim, reservoir=cfg.kde_reservoir, seed=seed)
        self.metric = LowRankMetric(dim=cfg.dim, rank=cfg.metric_rank, seed=seed)
        self.gate = ThompsonBernoulliGate(feature_dim=cfg.gate_feature_dim, seed=seed)

        # memory stores
        self.graph = MemoryGraph()
        self.vindex = VectorIndex(dim=cfg.dim)

        self.plots: Dict[str, Plot] = {}
        self.stories: Dict[str, StoryArc] = {}
        self.themes: Dict[str, Theme] = {}

        # nonparametric assignment
        self.crp_story = CRPAssigner(alpha=cfg.story_alpha, seed=seed)
        self.story_model = StoryModel(metric=self.metric)

        self.crp_theme = CRPAssigner(alpha=cfg.theme_alpha, seed=seed + 1)
        self.theme_model = ThemeModel(metric=self.metric)

        self.retriever = FieldRetriever(metric=self.metric, vindex=self.vindex, graph=self.graph)

        # bookkeeping for delayed credit assignment
        self._recent_encoded_plot_ids: List[str] = []  # sliding window

    # -------------------------------------------------------------------------
    # Feature computation for value-of-information (VOI) encoding
    # -------------------------------------------------------------------------

    def _redundancy(self, emb: np.ndarray) -> float:
        hits = self.vindex.search(emb, k=8, kind="plot")
        return max((s for _, s in hits), default=0.0)

    def _goal_relevance(self, emb: np.ndarray, context_emb: Optional[np.ndarray]) -> float:
        return cosine_sim(emb, context_emb) if context_emb is not None else 0.0

    def _pred_error(self, emb: np.ndarray) -> float:
        # predictive error vs best-matching story centroid under current metric
        best = None
        best_sim = -1.0
        for s in self.stories.values():
            if s.centroid is None:
                continue
            sim = self.metric.sim(emb, s.centroid)
            if sim > best_sim:
                best_sim = sim
                best = s
        if best is None:
            return 1.0
        return 1.0 - best_sim

    def _voi_features(self, plot: Plot) -> np.ndarray:
        # Features are observables, not manually weighted rules.
        return np.array([
            plot.surprise,
            plot.pred_error,
            1.0 - plot.redundancy,
            plot.goal_relevance,
            math.tanh(len(plot.text) / 512.0),
            1.0,
        ], dtype=np.float32)

    # -------------------------------------------------------------------------
    # Ingest
    # -------------------------------------------------------------------------

    def ingest(
        self,
        interaction_text: str,
        actors: Optional[Sequence[str]] = None,
        context_text: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> Plot:
        """Ingest an interaction/event.

        This method:
        1) embeds the interaction
        2) computes surprise + predictive error + redundancy + relevance
        3) decides to store stochastically via Thompson-sampled encode gate
        4) if stored, assigns it into a story (CRP) and updates graph edges
        """
        actors = tuple(actors) if actors else ("user", "agent")
        emb = self.embedder.embed(interaction_text)

        plot = Plot(
            id=det_id("plot", event_id) if event_id else str(uuid.uuid4()),
            ts=now_ts(),
            text=interaction_text,
            actors=tuple(actors),
            embedding=emb,
        )

        # Update global density regardless of storage decision (calibration)
        self.kde.add(emb)

        context_emb = self.embedder.embed(context_text) if context_text else None
        plot.surprise = float(self.kde.surprise(emb))
        plot.pred_error = float(self._pred_error(emb))
        plot.redundancy = float(self._redundancy(emb))
        plot.goal_relevance = float(self._goal_relevance(emb, context_emb))

        # Free-energy proxy: surprise plus belief update magnitude (here pred_error)
        plot.tension = plot.surprise * (1.0 + plot.pred_error)

        # Decide encode stochastically (no threshold)
        x = self._voi_features(plot)
        encode = self.gate.decide(x)

        if encode:
            self._store_plot(plot)
            self._recent_encoded_plot_ids.append(plot.id)
            # keep sliding window bounded
            if len(self._recent_encoded_plot_ids) > 200:
                self._recent_encoded_plot_ids = self._recent_encoded_plot_ids[-200:]
        # else: dropped (but still influenced KDE as calibration)

        # Pressure management: consolidate/absorb when exceeding capacity (probabilistic)
        self._pressure_manage()
        return plot

    # -------------------------------------------------------------------------
    # Store plot into story + graph weaving
    # -------------------------------------------------------------------------

    def _store_plot(self, plot: Plot) -> None:
        # 1) choose story assignment probabilities via CRP posterior
        logps: Dict[str, float] = {}
        for sid, s in self.stories.items():
            # prior proportional to story size
            prior = math.log(len(s.plot_ids) + 1e-6)
            logps[sid] = prior + self.story_model.loglik(plot, s)

        chosen, post = self.crp_story.sample(logps)
        if chosen is None:
            story = StoryArc(id=det_id("story", plot.id), created_ts=now_ts(), updated_ts=now_ts())
            self.stories[story.id] = story
            self.graph.add_node(story.id, "story", story)
            self.vindex.add(story.id, plot.embedding, kind="story")
            chosen = story.id

        story = self.stories[chosen]

        # 2) update story statistics and centroid
        if story.centroid is not None:
            d2 = self.metric.d2(plot.embedding, story.centroid)
            story._update_stats("dist", d2)
            gap = max(0.0, plot.ts - story.updated_ts)
            story._update_stats("gap", gap)

        story.plot_ids.append(plot.id)
        story.updated_ts = plot.ts
        story.actor_counts = {a: story.actor_counts.get(a, 0) + 1 for a in plot.actors}
        story.tension_curve.append(plot.tension)

        if story.centroid is None:
            story.centroid = plot.embedding.copy()
        else:
            n = len(story.plot_ids)
            story.centroid = l2_normalize(story.centroid * ((n - 1) / n) + plot.embedding * (1.0 / n))

        # 3) store plot
        plot.story_id = story.id
        self.plots[plot.id] = plot
        self.graph.add_node(plot.id, "plot", plot)
        self.vindex.add(plot.id, plot.embedding, kind="plot")

        # 4) weave edges (typed relations; strength learned by Beta posteriors)
        self.graph.ensure_edge(plot.id, story.id, "belongs_to")
        self.graph.ensure_edge(story.id, plot.id, "contains")

        # temporal to previous plot in story
        if len(story.plot_ids) > 1:
            prev = story.plot_ids[-2]
            self.graph.ensure_edge(prev, plot.id, "temporal")

        # semantic links to nearest neighbors (no threshold; just top-k)
        for pid, _sim in self.vindex.search(plot.embedding, k=8, kind="plot"):
            if pid == plot.id:
                continue
            self.graph.ensure_edge(plot.id, pid, "semantic")
            self.graph.ensure_edge(pid, plot.id, "semantic")

    # -------------------------------------------------------------------------
    # Query / retrieval
    # -------------------------------------------------------------------------

    def query(self, text: str, k: int = 5) -> RetrievalTrace:
        trace = self.retriever.retrieve(query_text=text, embed=self.embedder, kinds=self.cfg.retrieval_kinds, k=k)

        # update access/mass
        for nid, _score, kind in trace.ranked:
            if kind == "plot":
                p: Plot = self.graph.payload(nid)
                p.access_count += 1
                p.last_access_ts = now_ts()
            elif kind == "story":
                s: StoryArc = self.graph.payload(nid)
                s.reference_count += 1
        return trace

    # -------------------------------------------------------------------------
    # Feedback: credit assignment and learning
    # -------------------------------------------------------------------------

    def feedback_retrieval(self, query_text: str, chosen_id: str, success: bool) -> None:
        """Delayed reward signal.

        Updates:
        - edge Beta posteriors along paths from query seeds to chosen item
        - metric using a triplet (query, chosen, negative)
        - encode gate using aggregate reward attributed to recent encoded plots
        """
        q = self.embedder.embed(query_text)

        # 1) update edges on (a few) shortest paths from seeds
        seeds = [i for i, _ in self.vindex.search(q, k=10)]
        G = self.graph.g
        if chosen_id in G:
            for s in seeds:
                if s not in G:
                    continue
                try:
                    path = nx.shortest_path(G, source=s, target=chosen_id)
                except nx.NetworkXNoPath:
                    continue
                for u, v in zip(path[:-1], path[1:]):
                    self.graph.edge_belief(u, v).update(success)

        # 2) metric triplet update
        if chosen_id in G:
            chosen = self.graph.payload(chosen_id)
            pos_emb = getattr(chosen, "embedding", getattr(chosen, "centroid", getattr(chosen, "prototype", None)))
            if pos_emb is not None:
                # pick negative among high-sim but not chosen
                cands = [i for i, _ in self.vindex.search(q, k=30) if i != chosen_id and i in G]
                if cands:
                    neg_id = random.choice(cands)
                    neg = self.graph.payload(neg_id)
                    neg_emb = getattr(neg, "embedding", getattr(neg, "centroid", getattr(neg, "prototype", None)))
                    if neg_emb is not None:
                        self.metric.update_triplet(anchor=q, positive=pos_emb, negative=neg_emb)

        # 3) encode gate update: attribute reward to recently encoded plots (soft credit)
        reward = 1.0 if success else -1.0
        # Use a light heuristic: distribute reward uniformly to last N encoded plots
        recent = self._recent_encoded_plot_ids[-20:]
        for pid in recent:
            p = self.plots.get(pid)
            if p is None:
                continue
            x = self._voi_features(p)
            self.gate.update(x, reward)

        # 4) theme evidence update if chosen was a theme
        if chosen_id in self.themes:
            self.themes[chosen_id].update_evidence(success)

    # -------------------------------------------------------------------------
    # Evolution: consolidation & theme emergence (Plot -> Story -> Theme)
    # -------------------------------------------------------------------------

    def evolve(self) -> None:
        """Offline-ish evolution step (can be triggered periodically or on idle).

        This method:
        - updates story statuses stochastically based on activity probabilities
        - assigns stories into themes via CRP (hierarchical emergence)
        - absorbs low-mass plots into their story to save space (pressure-driven)
        """
        # 1) update story statuses (probabilistic, no idle threshold)
        for s in self.stories.values():
            if s.status != "developing":
                continue
            p_active = s.activity_probability()
            # If story is still active, keep; else sample resolution vs abandonment
            if self.rng.random() < p_active:
                continue
            # resolve vs abandon depends on tension curve (if it has a decay, more likely resolved)
            if len(s.tension_curve) >= 3:
                # crude slope: last - first
                slope = s.tension_curve[-1] - s.tension_curve[0]
                p_resolve = sigmoid(-slope)  # if tension decreased, p_resolve high
            else:
                p_resolve = 0.5
            s.status = "resolved" if (self.rng.random() < p_resolve) else "abandoned"

        # 2) theme emergence: assign each resolved story into a theme
        for sid, s in list(self.stories.items()):
            if s.status != "resolved":
                continue
            if s.centroid is None:
                continue
            logps: Dict[str, float] = {}
            for tid, t in self.themes.items():
                prior = math.log(len(t.story_ids) + 1e-6)
                logps[tid] = prior + self.theme_model.loglik(s, t)
            chosen, post = self.crp_theme.sample(logps)
            if chosen is None:
                theme = Theme(id=det_id("theme", sid), created_ts=now_ts(), updated_ts=now_ts())
                theme.prototype = s.centroid.copy()
                self.themes[theme.id] = theme
                self.graph.add_node(theme.id, "theme", theme)
                self.vindex.add(theme.id, theme.prototype, kind="theme")
                chosen = theme.id
            theme = self.themes[chosen]
            theme.story_ids.append(sid)
            theme.updated_ts = now_ts()
            # prototype update (online mean)
            if theme.prototype is None:
                theme.prototype = s.centroid.copy()
            else:
                n = len(theme.story_ids)
                theme.prototype = l2_normalize(theme.prototype * ((n - 1) / n) + s.centroid * (1.0 / n))

            # weave edges
            self.graph.ensure_edge(sid, theme.id, "thematizes")
            self.graph.ensure_edge(theme.id, sid, "exemplified_by")

        # 3) absorb low-mass plots within stories to satisfy capacity (pressure-driven)
        self._pressure_manage()

    # -------------------------------------------------------------------------
    # Async Evolution: Copy-on-Write for non-blocking processing
    # -------------------------------------------------------------------------

    def create_evolution_snapshot(self) -> "EvolutionSnapshot":
        """Create a read-only snapshot for evolution processing.
        
        This captures the current state without holding locks, allowing
        evolution to proceed in a background thread while new ingests continue.
        """
        return EvolutionSnapshot(
            story_ids=list(self.stories.keys()),
            story_statuses={sid: s.status for sid, s in self.stories.items()},
            story_centroids={sid: s.centroid.copy() if s.centroid is not None else None 
                           for sid, s in self.stories.items()},
            story_tension_curves={sid: list(s.tension_curve) for sid, s in self.stories.items()},
            story_updated_ts={sid: s.updated_ts for sid, s in self.stories.items()},
            story_gap_means={sid: s.gap_mean_safe() for sid, s in self.stories.items()},
            theme_ids=list(self.themes.keys()),
            theme_story_counts={tid: len(t.story_ids) for tid, t in self.themes.items()},
            theme_prototypes={tid: t.prototype.copy() if t.prototype is not None else None
                            for tid, t in self.themes.items()},
            crp_theme_alpha=self.crp_theme.alpha,
            rng_state=self.rng.bit_generator.state,
        )

    def compute_evolution_patch(self, snapshot: "EvolutionSnapshot") -> "EvolutionPatch":
        """Compute evolution changes from snapshot (pure function, no side effects).
        
        This can run in a background thread without locks.
        """
        # Use a local RNG to avoid state mutation
        rng = np.random.default_rng()
        rng.bit_generator.state = snapshot.rng_state
        
        # 1) Determine story status changes
        status_changes: Dict[str, str] = {}
        for sid in snapshot.story_ids:
            if snapshot.story_statuses[sid] != "developing":
                continue
            
            # Activity probability
            ts = now_ts()
            updated = snapshot.story_updated_ts[sid]
            tau = snapshot.story_gap_means[sid]
            idle = max(0.0, ts - updated)
            p_active = math.exp(-idle / max(tau, 1e-6))
            
            if rng.random() < p_active:
                continue
            
            # Resolve vs abandon
            curve = snapshot.story_tension_curves[sid]
            if len(curve) >= 3:
                slope = curve[-1] - curve[0]
                p_resolve = sigmoid(-slope)
            else:
                p_resolve = 0.5
            
            new_status = "resolved" if rng.random() < p_resolve else "abandoned"
            status_changes[sid] = new_status
        
        # 2) Compute theme assignments for newly resolved stories
        theme_assignments: List[Tuple[str, Optional[str]]] = []  # (story_id, theme_id or None for new)
        new_themes: List[Tuple[str, np.ndarray]] = []  # (theme_id, prototype)
        
        # Build updated theme counts (accounting for new assignments)
        current_theme_counts = dict(snapshot.theme_story_counts)
        
        for sid, new_status in status_changes.items():
            if new_status != "resolved":
                continue
            
            centroid = snapshot.story_centroids[sid]
            if centroid is None:
                continue
            
            # Compute log probabilities for existing themes
            logps: Dict[str, float] = {}
            for tid in snapshot.theme_ids:
                prior = math.log(current_theme_counts.get(tid, 0) + 1e-6)
                prototype = snapshot.theme_prototypes.get(tid)
                if prototype is not None:
                    # Simple distance-based likelihood
                    d2 = float(np.dot(centroid - prototype, centroid - prototype))
                    logps[tid] = prior - 0.5 * d2
                else:
                    logps[tid] = prior
            
            # Add new theme option (CRP)
            logps["__new__"] = math.log(snapshot.crp_theme_alpha)
            
            # Sample
            keys = list(logps.keys())
            probs = softmax([logps[k] for k in keys])
            choice = rng.choice(keys, p=np.array(probs, dtype=np.float64))
            
            if choice == "__new__":
                # Create new theme
                new_theme_id = det_id("theme", sid)
                new_themes.append((new_theme_id, centroid.copy()))
                theme_assignments.append((sid, new_theme_id))
                current_theme_counts[new_theme_id] = 1
            else:
                theme_assignments.append((sid, choice))
                current_theme_counts[choice] = current_theme_counts.get(choice, 0) + 1
        
        return EvolutionPatch(
            status_changes=status_changes,
            theme_assignments=theme_assignments,
            new_themes=new_themes,
        )

    def apply_evolution_patch(self, patch: "EvolutionPatch") -> None:
        """Apply computed evolution changes atomically.
        
        Should be called with appropriate locking if needed.
        """
        # 1) Apply story status changes
        for sid, new_status in patch.status_changes.items():
            if sid in self.stories:
                self.stories[sid].status = new_status
        
        # 2) Create new themes
        for theme_id, prototype in patch.new_themes:
            theme = Theme(id=theme_id, created_ts=now_ts(), updated_ts=now_ts())
            theme.prototype = prototype
            self.themes[theme_id] = theme
            self.graph.add_node(theme_id, "theme", theme)
            self.vindex.add(theme_id, prototype, kind="theme")
        
        # 3) Apply theme assignments and weave edges
        for sid, tid in patch.theme_assignments:
            if sid not in self.stories or tid not in self.themes:
                continue
            
            story = self.stories[sid]
            theme = self.themes[tid]
            
            theme.story_ids.append(sid)
            theme.updated_ts = now_ts()
            
            # Update prototype (online mean)
            if story.centroid is not None:
                if theme.prototype is None:
                    theme.prototype = story.centroid.copy()
                else:
                    n = len(theme.story_ids)
                    theme.prototype = l2_normalize(
                        theme.prototype * ((n - 1) / n) + story.centroid * (1.0 / n)
                    )
            
            # Weave edges
            self.graph.ensure_edge(sid, tid, "thematizes")
            self.graph.ensure_edge(tid, sid, "exemplified_by")
        
        # 4) Pressure management
        self._pressure_manage()

    # -------------------------------------------------------------------------
    # Pressure management (resource constraints -> compression)
    # -------------------------------------------------------------------------

    def _pressure_manage(self) -> None:
        """Keep plot count under max_plots by probabilistic absorption/archival.

        No rule like 'if age>30 days' or 'if importance<0.2'. Instead:
        - compute a soft selection distribution over plots with low mass
        - sample plots to absorb/archive until within capacity
        """
        max_plots = self.cfg.max_plots
        if len(self.plots) <= max_plots:
            return

        # candidate plots that are active and already assigned to a story
        cands = [p for p in self.plots.values() if p.status == "active" and p.story_id is not None]
        if not cands:
            return

        # Build logits favoring low-mass plots
        masses = np.array([p.mass() for p in cands], dtype=np.float32)
        # lower mass -> higher probability to absorb
        logits = (-masses).tolist()
        probs = np.array(softmax(logits), dtype=np.float64)

        # number to remove
        excess = len(self.plots) - max_plots
        remove_ids = set(self.rng.choice([p.id for p in cands], size=excess, replace=False, p=probs))
        for pid in remove_ids:
            p = self.plots.get(pid)
            if p is None:
                continue
            self._absorb_plot(pid)

    def _absorb_plot(self, plot_id: str) -> None:
        """Absorb plot into its story (lose high-res detail, keep narrative)."""
        p = self.plots.get(plot_id)
        if p is None or p.story_id is None:
            return
        p.status = "absorbed"

        # Remove from vector index to reduce retrieval noise (keeps story centroid)
        self.vindex.remove(plot_id)
        # Keep node in graph for traceability, but you may remove it in production
        # self.graph.g.remove_node(plot_id)

    # -------------------------------------------------------------------------
    # Convenience: inspect
    # -------------------------------------------------------------------------

    def get_story(self, story_id: str) -> StoryArc:
        return self.stories[story_id]

    def get_plot(self, plot_id: str) -> Plot:
        return self.plots[plot_id]

    def get_theme(self, theme_id: str) -> Theme:
        return self.themes[theme_id]

    # -------------------------------------------------------------------------
    # State serialization (JSON-compatible, replaces pickle)
    # -------------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """Serialize entire AuroraMemory state to JSON-compatible dict.
        
        This replaces pickle-based serialization with structured JSON,
        enabling:
        - Human-readable state inspection
        - Cross-version compatibility
        - Partial state recovery
        - State diffing and debugging
        """
        return {
            "version": 2,  # State format version for forward compatibility
            "cfg": self.cfg.to_state_dict(),
            "seed": self._seed,
            
            # Learnable components
            "kde": self.kde.to_state_dict(),
            "metric": self.metric.to_state_dict(),
            "gate": self.gate.to_state_dict(),
            
            # Nonparametric assignment
            "crp_story": self.crp_story.to_state_dict(),
            "crp_theme": self.crp_theme.to_state_dict(),
            
            # Memory data
            "plots": {pid: p.to_state_dict() for pid, p in self.plots.items()},
            "stories": {sid: s.to_state_dict() for sid, s in self.stories.items()},
            "themes": {tid: t.to_state_dict() for tid, t in self.themes.items()},
            
            # Graph structure (payloads reference plots/stories/themes)
            "graph": self.graph.to_state_dict(),
            
            # Vector index (deprecated in production, use VectorStore)
            "vindex": self.vindex.to_state_dict(),
            
            # Bookkeeping
            "recent_encoded_plot_ids": self._recent_encoded_plot_ids,
        }

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "AuroraMemory":
        """Reconstruct AuroraMemory from state dict.
        
        Handles version migration if needed.
        """
        version = d.get("version", 1)
        
        # Reconstruct config
        cfg = MemoryConfig.from_state_dict(d["cfg"])
        seed = d.get("seed", 0)
        
        # Create new instance with config
        obj = cls(cfg=cfg, seed=seed)
        
        # Restore learnable components
        obj.kde = OnlineKDE.from_state_dict(d["kde"])
        obj.metric = LowRankMetric.from_state_dict(d["metric"])
        obj.gate = ThompsonBernoulliGate.from_state_dict(d["gate"])
        
        # Restore CRP assigners
        obj.crp_story = CRPAssigner.from_state_dict(d["crp_story"])
        obj.crp_theme = CRPAssigner.from_state_dict(d["crp_theme"])
        
        # Restore memory data
        obj.plots = {pid: Plot.from_state_dict(pd) for pid, pd in d.get("plots", {}).items()}
        obj.stories = {sid: StoryArc.from_state_dict(sd) for sid, sd in d.get("stories", {}).items()}
        obj.themes = {tid: Theme.from_state_dict(td) for tid, td in d.get("themes", {}).items()}
        
        # Build payload lookup for graph reconstruction
        payloads: Dict[str, Any] = {}
        payloads.update(obj.plots)
        payloads.update(obj.stories)
        payloads.update(obj.themes)
        
        # Restore graph
        obj.graph = MemoryGraph.from_state_dict(d["graph"], payloads=payloads)
        
        # Restore vector index
        obj.vindex = VectorIndex.from_state_dict(d["vindex"])
        
        # Rebuild models with restored metric
        obj.story_model = StoryModel(metric=obj.metric)
        obj.theme_model = ThemeModel(metric=obj.metric)
        obj.retriever = FieldRetriever(metric=obj.metric, vindex=obj.vindex, graph=obj.graph)
        
        # Restore bookkeeping
        obj._recent_encoded_plot_ids = d.get("recent_encoded_plot_ids", [])
        
        return obj


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mem = AuroraMemory(cfg=MemoryConfig(dim=96, metric_rank=32, max_plots=2000), seed=42)

    # Ingest a few interactions
    mem.ingest("用户：我想做一个记忆系统。助理：好的，我们从第一性原理开始。", context_text="memory algorithm")
    mem.ingest("用户：不要硬编码阈值。助理：可以用贝叶斯决策和随机策略。", context_text="memory algorithm")
    mem.ingest("用户：检索要能讲故事。助理：可以用故事弧 + 主题涌现。", context_text="narrative memory")
    mem.ingest("用户：给我一个可运行的实现。助理：我会给你一份python参考实现。", context_text="implementation")

    trace = mem.query("如何避免硬编码阈值并实现叙事检索？", k=5)
    print("Top results:", trace.ranked)

    # Provide feedback
    if trace.ranked:
        chosen_id = trace.ranked[0][0]
        mem.feedback_retrieval("如何避免硬编码阈值并实现叙事检索？", chosen_id=chosen_id, success=True)

    # Run evolution
    mem.evolve()
    print("stories:", len(mem.stories), "themes:", len(mem.themes), "plots:", len(mem.plots))
