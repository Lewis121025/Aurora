# AURORA 叙事记忆算法（第一性原理版）参考实现

> [!WARNING]
> 本文档在 Aurora Soul canonical migration 之后已过时。当前可运行主线已经收敛到 `aurora.soul`，文中的旧命名和示例代码未完全同步。

> 目标：用**可学习的概率机制**替代“阈值/固定权重/规则硬编码”，在资源约束下，让记忆结构 **Plot→Story→Theme** 自组织涌现，并且能被反馈持续塑形。  
> 参考：你上传的两份文档分别强调了“叙事层级/涌现结构”与“自由能、场论、信息几何”等方向，我这里在不被原思路束缚的前提下，把这些动机从第一性原理重构成一个可运行的算法骨架。  
> 文档引用：fileciteturn0file0 fileciteturn0file1

---

## 1. 第一性原理出发的设计约束（为什么这么做）

**记忆的本质不是存储，而是：在有限资源下最大化未来决策质量。**

我们把记忆系统视为一个在线最优化器，面对每次交互 `o_t` 要解决三件事：

1. **要不要记？**  
   记忆的“价值”来自它降低未来不确定性/提高任务成功率；“成本”来自存储、检索噪声、演化开销。  
   → 用 **Thompson Sampling 的随机门控**学出来，而不是 `if score > 0.3`。

2. **记到哪里？（结构归属）**  
   交互不是孤点，应被归入正在发展的故事弧，或创建新故事弧。故事弧再汇聚成主题。  
   → 用 **CRP/Dirichlet Process** 做 Plot→Story→Theme 的非参数聚类：不预设故事/主题数量，也不需要继续阈值。

3. **怎么用？（检索）**  
   检索不仅是“最近邻”，而是“在记忆场里被上下文势能吸引到某个吸引子”。  
   → 用 **连续空间的 attractor tracing（mean shift）** + **离散图扩散（PageRank）** 的二阶段检索。

---

## 2. 这份实现的“算法升级点”（相对常见方案）

- **无阈值编码**：编码决策是一个可学习的概率策略 `p(encode|x)`，由反馈塑形。
- **无固定权重融合**：故事归属、主题归属使用**生成式对数似然**拼合（语义/时间/角色都是概率项），而不是手写权重。
- **检索=场论近似实现**：mean-shift 在 learned metric 空间寻找“模式/吸引子”，再在图上扩散得到多跳证据链。
- **边权重=贝叶斯后验**：每条边维护 Beta 后验 `Beta(a,b)` 表示“这条关系在过去是否帮助成功”，由反馈更新。
- **信息几何可学习**：用低秩 Mahalanobis `d(x,y)=||L(x-y)||` 作为“任务相关度量”，通过 triplet 反馈在线更新。
- **资源约束驱动的压缩**：不是“超过30天就归档”，而是在容量压力下对低质量 plot 做概率吸收（soft selection）。

---

## 3. 完整参考代码（单文件，可直接运行）

> 说明：  
> - 代码里使用 `HashEmbedding` 作为可运行的“假 embedding”；生产请替换成真实 embedding 模型。  
> - 代码不依赖训练大模型即可跑通；如果接入 LLM，可替换 Plot/Theme 的抽取与命名模块。  
> - 只依赖 `numpy` 和 `networkx`。

```python
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


class LowRankMetric:
    """Low-rank Mahalanobis metric: d(x,y)^2 = ||L(x-y)||^2.

    Interpretable as learning a task- and user-adapted "information geometry" metric
    from retrieval feedback.
    """
    def __init__(self, dim: int, rank: int = 64, seed: int = 0):
        self.dim = dim
        self.rank = min(rank, dim)
        rng = np.random.default_rng(seed)
        self.L = np.eye(dim, dtype=np.float32)[: self.rank].copy()
        self.L += (0.01 * rng.normal(size=self.L.shape)).astype(np.float32)

        self.G = np.zeros_like(self.L)  # Adagrad accumulator
        self.t = 0

    def d2(self, x: np.ndarray, y: np.ndarray) -> float:
        z = (x - y).astype(np.float32)
        p = self.L @ z
        return float(np.dot(p, p))

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        return 1.0 / (1.0 + self.d2(x, y))

    def update_triplet(self, anchor: np.ndarray, positive: np.ndarray, negative: np.ndarray, margin: float = 1.0) -> float:
        """Online OASIS-like update with Adagrad.

        margin is not a threshold on similarity; it's a geometric separation unit.
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
        return float(loss)


class ThompsonBernoulliGate:
    """Stochastic encode policy with Thompson sampling.

    We do not decide by "score > threshold". We sample a parameter vector w and
    encode with probability sigmoid(w·x). The mapping is learned from delayed rewards.
    """
    def __init__(self, feature_dim: int, seed: int = 0):
        self.d = feature_dim
        self.rng = np.random.default_rng(seed)

        self.w_mean = np.zeros(self.d, dtype=np.float32)
        self.prec = np.ones(self.d, dtype=np.float32) * 1e-2  # weak precision
        self.grad2 = np.zeros(self.d, dtype=np.float32)  # RMS

        self.t = 0

    def _sample_w(self) -> np.ndarray:
        std = np.sqrt(1.0 / (self.prec + 1e-9))
        return self.w_mean + self.rng.normal(size=self.d).astype(np.float32) * std

    def prob(self, x: np.ndarray) -> float:
        w = self._sample_w()
        return sigmoid(float(np.dot(w, x)))

    def decide(self, x: np.ndarray) -> bool:
        return bool(self.rng.random() < self.prob(x))

    def update(self, x: np.ndarray, reward: float) -> None:
        """Bandit update: reward in [-1, 1] from downstream task success."""
        self.t += 1
        y = 1.0 if reward > 0 else 0.0
        p = sigmoid(float(np.dot(self.w_mean, x)))
        grad = (y - p) * x  # ascent

        # RMS with self-tuning step size
        self.grad2 = 0.99 * self.grad2 + 0.01 * (grad * grad)
        base = 1.0 / math.sqrt(self.t + 1.0)
        step = base * grad / (np.sqrt(self.grad2) + 1e-6)
        self.w_mean += step
        self.prec += grad * grad


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


class VectorIndex:
    """Brute-force vector index with kind filtering.

    Replace with FAISS for larger local corpora when needed.
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


# -----------------------------------------------------------------------------
# Nonparametric hierarchical assignment (CRP)
# -----------------------------------------------------------------------------

class CRPAssigner:
    """Generic CRP-like assigner for (item -> cluster) with probabilistic sampling."""

    def __init__(self, alpha: float = 1.0, seed: int = 0):
        self.alpha = alpha
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
class SoulConfig:
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


class AuroraSoul:
    """AURORA Soul: emergent narrative memory from first principles.

    Key APIs:
        ingest(interaction_text, actors, context_text) -> Plot (may or may not be stored)
        query(text, k) -> RetrievalTrace
        feedback_retrieval(query_text, chosen_id, success) -> update beliefs
        evolve() -> consolidate plots->stories->themes, manage pressure, update statuses
    """

    def __init__(self, cfg: SoulConfig = SoulConfig(), seed: int = 0):
        self.cfg = cfg
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
            id=str(uuid.uuid4()),
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
            story = StoryArc(id=str(uuid.uuid4()), created_ts=now_ts(), updated_ts=now_ts())
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
                theme = Theme(id=str(uuid.uuid4()), created_ts=now_ts(), updated_ts=now_ts())
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


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    mem = AuroraSoul(cfg=SoulConfig(dim=96, metric_rank=32, max_plots=2000), seed=42)

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
```

---

## 4. 你接下来可以怎么把它变成“生产级”

1. **替换 HashEmbedding**：接入你们实际 embedding 服务（OpenAI / BGE / text-embedding-3-large 等）。
2. **Plot 结构化抽取**：用 LLM 抽取 actors/action/context/outcome（这会显著提升因果与一致性模块的上限）。
3. **向量索引**：把 `VectorIndex` 换成 FAISS/HNSW。
4. **一致性守护**：在 Plot/Story/Theme 上抽取 claim triples + contradiction classifier，把 `contradiction` 边纳入 Beta 后验更新。
5. **元学习层**：对 `story_alpha / theme_alpha / mean_shift_steps / pagerank_damping` 用 bandit 做自动调参，而不是写死。

---

如果你希望我把“LLM 抽取 Plot/Theme/SelfNarrative 的 prompts + JSON schema + 训练无关的对齐策略”也一起写出来，我也可以基于这套算法接口继续补齐。
