"""
aurora/soul/retrieval.py
本模块实现了 Aurora V4 的“场论检索”系统。
它结合了高维向量搜索、Mean-shift 吸引子追踪、图算法 (PageRank) 以及基于身份状态的偏置。
它不仅寻找“语义相似”的记忆，还寻找“心理连贯”和“叙事相关”的记忆。
"""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, cast

import hnswlib
import numpy as np

from aurora.soul.query import (
    BaseQueryAnalyzer,
    FACT_KEY_BOOST_MAX,
    FACTUAL_ATTRACTOR_WEIGHT,
    FACTUAL_PLOT_PRIORITY_BOOST,
    FACTUAL_SEMANTIC_WEIGHT,
    KEYWORD_MATCH_BOOST,
    KEYWORD_MATCH_MIN_RATIO,
    MULTI_HOP_EXTRA_PAGERANK_ITER,
    QueryAnalysis,
    QueryType,
    SINGLE_SESSION_USER_K_MULTIPLIER,
    TimeRange,
    TimeRangeExtractor,
    USER_ROLE_PRIORITY_BOOST,
)
from aurora.soul.facts import FactExtractor
from aurora.soul.models import (
    EdgeBelief,
    IdentityState,
    Plot,
    RetrievalTrace,
    Summary,
    Theme,
    l2_normalize,
)
from aurora.soul.tuning import RetrievalWeights
from aurora.utils.jsonx import dumps, loads
from aurora.utils.math_utils import cosine_sim, softmax


class OnlineKDE:
    """
    在线核密度估计 (Kernel Density Estimation)：
    用于评估新输入的“惊讶度 (Surprise)”。如果一个输入落在记忆密度较低的区域，
    则其惊讶度较高，更容易引发系统的注意。
    """

    def __init__(self, dim: int, reservoir: int = 4096, k_sigma: int = 25, seed: int = 0):
        self.dim = dim
        self.reservoir = reservoir
        self.k_sigma = k_sigma
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._vecs: List[np.ndarray] = []

    def add(self, x: np.ndarray) -> None:
        """添加向量到蓄水池"""
        vec = x.astype(np.float32)
        if len(self._vecs) < self.reservoir:
            self._vecs.append(vec)
            return
        idx = int(self.rng.integers(0, len(self._vecs) + 1))
        if idx < len(self._vecs):
            self._vecs[idx] = vec

    def _sigma(self, x: np.ndarray) -> float:
        """动态计算带宽 Sigma：基于 K 近邻距离的观测值"""
        if not self._vecs:
            return 1.0
        dists = [float(np.linalg.norm(x - v)) for v in self._vecs]
        dists.sort()
        k = min(self.k_sigma, len(dists))
        median = float(np.median(dists[:k])) if k > 0 else float(np.median(dists))
        return median + 1e-6

    def log_density(self, x: np.ndarray) -> float:
        """计算对数密度：衡量向量 x 与现有向量池的重合程度"""
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
        """惊讶度 = 负对数密度"""
        return -self.log_density(x)

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "dim": self.dim,
            "reservoir": self.reservoir,
            "k_sigma": self.k_sigma,
            "seed": self._seed,
            "vecs": [vec.astype(np.float32).tolist() for vec in self._vecs],
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, Any]) -> "OnlineKDE":
        """反序列化"""
        obj = cls(
            dim=int(data["dim"]),
            reservoir=int(data["reservoir"]),
            k_sigma=int(data.get("k_sigma", 25)),
            seed=int(data.get("seed", 0)),
        )
        obj._vecs = [np.asarray(vec, dtype=np.float32) for vec in data.get("vecs", [])]
        return obj


class LowRankMetric:
    """
    低秩度量学习 (Low-Rank Metric Learning)：
    用于在线微调向量空间的距离计算。通过用户的反馈（Triplet Loss），
    逐渐学会在该 Agent 的主观视角下，哪些记忆是“相关”的。
    """

    def __init__(self, dim: int, rank: int = 64, seed: int = 0):
        self.dim = dim
        self.rank = min(rank, dim)
        self._seed = seed
        rng = np.random.default_rng(seed)
        # 初始化投影矩阵 L 为单位矩阵前 rank 行 + 噪声
        self.L = np.eye(dim, dtype=np.float32)[: self.rank].copy()
        self.L += (0.01 * rng.normal(size=self.L.shape)).astype(np.float32)
        # 梯度累积，用于 AdaGrad 更新
        self.G = np.zeros_like(self.L)
        self.t = 0

    def d2(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算平方度量距离：||L(x-y)||^2"""
        delta = (x - y).astype(np.float32)
        projected = self.L @ delta
        return float(np.dot(projected, projected))

    def sim(self, x: np.ndarray, y: np.ndarray) -> float:
        """度量相似度"""
        return 1.0 / (1.0 + self.d2(x, y))

    def update_triplet(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negative: np.ndarray,
        margin: float = 1.0,
    ) -> float:
        """
        利用三元组损失进行在线学习：
        让 anchor 靠近 positive，远离 negative。
        """
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
        # 计算梯度并执行 AdaGrad 更新
        grad = 2.0 * (np.outer(lap, ap) - np.outer(lan, an)).astype(np.float32)
        self.G += grad * grad
        step = (1.0 / math.sqrt(self.t + 1.0)) * grad / (np.sqrt(self.G) + 1e-8)
        self.L -= step
        return float(loss)

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化"""
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
        """反序列化"""
        obj = cls(
            dim=int(data["dim"]),
            rank=int(data["rank"]),
            seed=int(data.get("seed", 0)),
        )
        obj.L = np.asarray(data["L"], dtype=np.float32)
        obj.G = np.asarray(data["G"], dtype=np.float32)
        obj.t = int(data.get("t", 0))
        return obj


class MemoryGraph:
    """
    记忆图谱：以稀疏邻接表维护节点与边，避免 networkx 带来的纯 Python 热路径开销。
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._kind_index: Dict[str, set[str]] = {}
        self._out_edges: Dict[str, Dict[str, EdgeBelief]] = {}
        self._in_edges: Dict[str, set[str]] = {}
        self._pagerank_cache: Dict[Tuple[int, str, float, int, float], Dict[str, float]] = {}
        self._edge_upserts: Dict[Tuple[str, str], EdgeBelief] = {}
        self._edge_deletes: set[Tuple[str, str]] = set()
        self._edge_version = 0

    def add_node(self, node_id: str, kind: str, payload: Any) -> None:
        """添加或更新节点。"""
        previous = self._nodes.get(node_id)
        if previous is not None:
            previous_kind = str(previous["kind"])
            if previous_kind != kind:
                self._kind_index.get(previous_kind, set()).discard(node_id)
        self._nodes[node_id] = {"kind": kind, "payload": payload}
        self._kind_index.setdefault(kind, set()).add(node_id)
        self._out_edges.setdefault(node_id, {})
        self._in_edges.setdefault(node_id, set())

    def kind(self, node_id: str) -> str:
        return str(self._nodes[node_id]["kind"])

    def payload(self, node_id: str) -> Any:
        return self._nodes[node_id]["payload"]

    def __contains__(self, node_id: object) -> bool:
        return isinstance(node_id, str) and node_id in self._nodes

    @property
    def edge_version(self) -> int:
        return self._edge_version

    def node_ids(self) -> List[str]:
        return list(self._nodes.keys())

    def clear(self) -> None:
        for src, dst in self.iter_edges():
            self._edge_deletes.add((src, dst))
        self._nodes.clear()
        self._kind_index.clear()
        self._out_edges.clear()
        self._in_edges.clear()
        self._edge_version += 1
        self.clear_pagerank_cache()

    def ensure_edge(
        self,
        src: str,
        dst: str,
        edge_type: str,
        *,
        sign: int = 1,
        weight: float = 1.0,
        confidence: float = 1.0,
        provenance: str = "",
    ) -> None:
        """添加边（如果不存在），关联 EdgeBelief 进行置信度管理"""
        if src not in self._nodes or dst not in self._nodes:
            raise KeyError(f"Both nodes must exist before adding an edge: {src!r} -> {dst!r}")
        normalized_sign = 1 if sign >= 0 else -1
        existing = self._out_edges.setdefault(src, {}).get(dst)
        if existing is not None:
            belief = existing
            changed = (
                belief.edge_type != edge_type
                or belief.sign != normalized_sign
                or abs(belief.weight - float(weight)) > 1e-9
                or abs(belief.confidence - float(confidence)) > 1e-9
                or (provenance and belief.provenance != provenance)
            )
            belief.edge_type = edge_type
            belief.sign = normalized_sign
            belief.weight = float(weight)
            belief.confidence = float(confidence)
            if provenance:
                belief.provenance = provenance
            if changed:
                self._edge_version += 1
                self.clear_pagerank_cache()
                self._edge_upserts[(src, dst)] = belief
                self._edge_deletes.discard((src, dst))
            return
        belief = EdgeBelief(
            edge_type=edge_type,
            sign=normalized_sign,
            weight=float(weight),
            confidence=float(confidence),
            provenance=provenance,
        )
        self._out_edges[src][dst] = belief
        self._in_edges.setdefault(dst, set()).add(src)
        self._edge_version += 1
        self.clear_pagerank_cache()
        self._edge_upserts[(src, dst)] = belief
        self._edge_deletes.discard((src, dst))

    def edge_belief(self, src: str, dst: str) -> EdgeBelief:
        return self._out_edges[src][dst]

    def nodes_of_kind(self, kind: str) -> List[str]:
        return list(self._kind_index.get(kind, ()))

    def successors(self, node_id: str) -> List[str]:
        return list(self._out_edges.get(node_id, {}).keys())

    def predecessors(self, node_id: str) -> List[str]:
        return list(self._in_edges.get(node_id, ()))

    def has_edge(self, src: str, dst: str) -> bool:
        return dst in self._out_edges.get(src, {})

    def iter_edges(self, *, sign: Optional[int] = None) -> Iterator[Tuple[str, str]]:
        for src, neighbors in self._out_edges.items():
            for dst, belief in neighbors.items():
                if sign is not None and belief.sign != sign:
                    continue
                yield src, dst

    def iter_edge_items(
        self,
        *,
        sign: Optional[int] = None,
    ) -> Iterator[Tuple[str, str, EdgeBelief]]:
        for src, neighbors in self._out_edges.items():
            for dst, belief in neighbors.items():
                if sign is not None and belief.sign != sign:
                    continue
                yield src, dst, belief

    def remove_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            return
        kind = self.kind(node_id)
        self._kind_index.get(kind, set()).discard(node_id)
        for predecessor in list(self._in_edges.get(node_id, ())):
            self._edge_upserts.pop((predecessor, node_id), None)
            self._edge_deletes.add((predecessor, node_id))
            self._out_edges.get(predecessor, {}).pop(node_id, None)
        for successor in list(self._out_edges.get(node_id, {}).keys()):
            self._edge_upserts.pop((node_id, successor), None)
            self._edge_deletes.add((node_id, successor))
            self._in_edges.get(successor, set()).discard(node_id)
        self._out_edges.pop(node_id, None)
        self._in_edges.pop(node_id, None)
        self._nodes.pop(node_id, None)
        self._edge_version += 1
        self.clear_pagerank_cache()

    def _hash_personalization(self, personalization: Dict[str, float]) -> str:
        """为个性化向量生成哈希，用于缓存查找"""
        items = sorted((str(node_id), float(score)) for node_id, score in personalization.items())
        return hashlib.md5(repr(items).encode("utf-8")).hexdigest()[:16]

    def get_cached_pagerank(
        self,
        personalization: Dict[str, float],
        damping: float,
        max_iter: int,
        tol: float,
    ) -> Optional[Dict[str, float]]:
        """获取缓存的 PR 结果"""
        key = (
            self._edge_version,
            self._hash_personalization(personalization),
            float(damping),
            int(max_iter),
            float(tol),
        )
        return self._pagerank_cache.get(key)

    def set_cached_pagerank(
        self,
        personalization: Dict[str, float],
        damping: float,
        max_iter: int,
        tol: float,
        result: Dict[str, float],
    ) -> None:
        """存入缓存"""
        key = (
            self._edge_version,
            self._hash_personalization(personalization),
            float(damping),
            int(max_iter),
            float(tol),
        )
        self._pagerank_cache[key] = dict(result)

    def clear_pagerank_cache(self) -> None:
        self._pagerank_cache.clear()

    def clear_edge_delta(self) -> None:
        self._edge_upserts.clear()
        self._edge_deletes.clear()

    def mark_edge_dirty(self, src: str, dst: str) -> None:
        if self.has_edge(src, dst):
            self._edge_upserts[(src, dst)] = self.edge_belief(src, dst)
            self._edge_deletes.discard((src, dst))

    def consume_edge_delta(
        self,
    ) -> Tuple[List[Tuple[str, str, EdgeBelief]], List[Tuple[str, str]]]:
        upserts = [(src, dst, belief) for (src, dst), belief in self._edge_upserts.items()]
        deletes = list(self._edge_deletes)
        self.clear_edge_delta()
        return upserts, deletes

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化边关系"""
        edges = []
        for src, dst, belief in self.iter_edge_items():
            edges.append(
                {
                    "src": src,
                    "dst": dst,
                    "belief": belief.to_state_dict(),
                }
            )
        return {
            "edges": edges,
            "edge_version": int(self._edge_version),
        }

    def restore_edges(self, state: Dict[str, Any]) -> None:
        """反序列化并恢复边"""
        for item in state.get("edges", []):
            src = str(item["src"])
            dst = str(item["dst"])
            if src not in self or dst not in self:
                continue
            belief = EdgeBelief.from_state_dict(item["belief"])
            self.ensure_edge(
                src,
                dst,
                belief.edge_type,
                sign=belief.sign,
                weight=belief.weight,
                confidence=belief.confidence,
                provenance=belief.provenance,
            )
        self._edge_version = int(state.get("edge_version", len(list(self.iter_edges()))))
        self.clear_edge_delta()


@dataclass
class _KindAnnIndex:
    kind: str
    index: hnswlib.Index
    label_to_id: Dict[int, str]
    id_to_label: Dict[str, int]
    next_label: int = 0


class VectorIndex:
    """
    HNSW 向量索引，按 kind 独立建索引以保持现有检索语义。
    """

    def __init__(
        self,
        dim: int,
        *,
        m: int = 32,
        ef_construction: int = 200,
        ef_search: int = 128,
    ):
        self.dim = dim
        self.m = int(m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._vectors: Dict[str, np.ndarray] = {}
        self._kinds: Dict[str, str] = {}
        self._indices: Dict[str, _KindAnnIndex] = {}

    def _build_index(self, kind: str, *, capacity: int) -> _KindAnnIndex:
        index = hnswlib.Index(space="cosine", dim=self.dim)
        index.init_index(
            max_elements=max(16, int(capacity)),
            ef_construction=self.ef_construction,
            M=self.m,
            allow_replace_deleted=True,
        )
        index.set_ef(self.ef_search)
        return _KindAnnIndex(
            kind=kind,
            index=index,
            label_to_id={},
            id_to_label={},
            next_label=0,
        )

    def _ensure_kind_index(self, kind: str, *, extra_capacity: int = 1) -> _KindAnnIndex:
        state = self._indices.get(kind)
        if state is None:
            state = self._build_index(kind, capacity=extra_capacity)
            self._indices[kind] = state
            return state
        current_capacity = int(state.index.get_max_elements())
        # HNSW labels are monotonically allocated. After repeated delete/re-add
        # cycles we must reserve against the next label, not just active items.
        required = max(len(state.id_to_label), state.next_label) + extra_capacity
        if required > current_capacity:
            state.index.resize_index(max(required, current_capacity * 2))
        return state

    def add(self, item_id: str, vec: np.ndarray, kind: str) -> None:
        """添加向量"""
        vector = vec.astype(np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"vector dim mismatch: {vector.shape} vs {(self.dim,)}")
        if item_id in self._vectors:
            self.remove(item_id)
        state = self._ensure_kind_index(kind)
        label = state.next_label
        state.next_label += 1
        state.index.add_items(vector.reshape(1, -1), np.asarray([label], dtype=np.int64))
        state.label_to_id[label] = item_id
        state.id_to_label[item_id] = label
        self._vectors[item_id] = vector
        self._kinds[item_id] = kind

    def remove(self, item_id: str) -> None:
        """删除向量"""
        kind = self._kinds.pop(item_id, None)
        if kind is None:
            return
        state = self._indices.get(kind)
        label = None if state is None else state.id_to_label.pop(item_id, None)
        self._vectors.pop(item_id, None)
        if state is not None and label is not None:
            state.label_to_id.pop(label, None)
            state.index.mark_deleted(label)

    def get_vector(self, item_id: str) -> Optional[np.ndarray]:
        return self._vectors.get(item_id)

    def clear(self) -> None:
        self._vectors.clear()
        self._kinds.clear()
        self._indices.clear()

    def rebuild_indices(self, *, kinds: Optional[Sequence[str]] = None) -> None:
        target_kinds = (
            sorted({kind for kind in self._kinds.values()})
            if kinds is None
            else [str(kind) for kind in kinds]
        )
        retained: Dict[str, _KindAnnIndex] = (
            {} if kinds is None else {kind: state for kind, state in self._indices.items() if kind not in target_kinds}
        )
        for kind in target_kinds:
            item_ids = [item_id for item_id, item_kind in self._kinds.items() if item_kind == kind]
            if not item_ids:
                continue
            state = self._build_index(kind, capacity=max(16, len(item_ids)))
            vectors = np.asarray([self._vectors[item_id] for item_id in item_ids], dtype=np.float32)
            labels = np.arange(len(item_ids), dtype=np.int64)
            state.index.add_items(vectors, labels)
            state.label_to_id = {
                int(label): item_id for label, item_id in zip(labels.tolist(), item_ids)
            }
            state.id_to_label = {item_id: int(label) for label, item_id in zip(labels.tolist(), item_ids)}
            state.next_label = len(item_ids)
            retained[kind] = state
        self._indices = retained

    def save_sidecar(
        self,
        directory: str,
        *,
        epoch: int,
        kinds: Optional[Sequence[str]] = None,
    ) -> None:
        os.makedirs(directory, exist_ok=True)
        meta_path = os.path.join(directory, "meta.json")
        metadata: Dict[str, Any]
        if kinds is None or not os.path.exists(meta_path):
            metadata = {
                "epoch": int(epoch),
                "dim": self.dim,
                "m": self.m,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "kinds": {},
            }
            for filename in os.listdir(directory):
                if filename.endswith(".bin") or filename == "meta.json":
                    os.remove(os.path.join(directory, filename))
            target_kinds = list(self._indices.keys()) if kinds is None else list(kinds)
        else:
            with open(meta_path, "r", encoding="utf-8") as handle:
                payload = loads(handle.read())
            metadata = payload if isinstance(payload, dict) else {}
            metadata["epoch"] = int(epoch)
            metadata["dim"] = self.dim
            metadata["m"] = self.m
            metadata["ef_construction"] = self.ef_construction
            metadata["ef_search"] = self.ef_search
            metadata.setdefault("kinds", {})
            target_kinds = list(kinds)
        for kind in target_kinds:
            state = self._indices.get(kind)
            filename = f"{kind}.bin"
            path = os.path.join(directory, filename)
            if state is None or not state.id_to_label:
                if os.path.exists(path):
                    os.remove(path)
                metadata["kinds"].pop(kind, None)
                continue
            state.index.save_index(path)
            metadata["kinds"][kind] = {
                "file": filename,
                "next_label": state.next_label,
                "label_to_id": {str(label): item_id for label, item_id in state.label_to_id.items()},
            }
        with open(meta_path, "w", encoding="utf-8") as handle:
            handle.write(dumps(metadata))

    def try_load_sidecar(self, directory: str, *, epoch: int) -> bool:
        meta_path = os.path.join(directory, "meta.json")
        if not os.path.exists(meta_path):
            return False
        with open(meta_path, "r", encoding="utf-8") as handle:
            payload = loads(handle.read())
        if not isinstance(payload, dict):
            return False
        if int(payload.get("epoch", -1)) != int(epoch):
            return False
        if int(payload.get("dim", -1)) != self.dim:
            return False
        kinds_payload = payload.get("kinds", {})
        if not isinstance(kinds_payload, dict):
            return False
        loaded_indices: Dict[str, _KindAnnIndex] = {}
        for kind, item in kinds_payload.items():
            if not isinstance(item, dict):
                return False
            label_to_id_raw = item.get("label_to_id", {})
            if not isinstance(label_to_id_raw, dict):
                return False
            label_to_id = {int(label): str(item_id) for label, item_id in label_to_id_raw.items()}
            for item_id in label_to_id.values():
                if item_id not in self._vectors or self._kinds.get(item_id) != kind:
                    return False
            index_path = os.path.join(directory, str(item.get("file", "")))
            if not os.path.exists(index_path):
                return False
            index = hnswlib.Index(space="cosine", dim=self.dim)
            index.load_index(index_path, max_elements=max(len(label_to_id), 16))
            index.set_ef(self.ef_search)
            loaded_indices[str(kind)] = _KindAnnIndex(
                kind=str(kind),
                index=index,
                label_to_id=label_to_id,
                id_to_label={item_id: label for label, item_id in label_to_id.items()},
                next_label=int(item.get("next_label", len(label_to_id))),
            )
        required_kinds = {kind for kind in self._kinds.values()}
        if set(loaded_indices.keys()) != required_kinds:
            return False
        self._indices = loaded_indices
        return True

    def search(
        self, q: np.ndarray, k: int = 10, kind: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """执行 HNSW 余弦相似度搜索。"""
        if not self._vectors:
            return []
        vector = q.astype(np.float32)
        kinds = [kind] if kind is not None else list(self._indices.keys())
        hits: List[Tuple[str, float]] = []
        for target_kind in kinds:
            state = self._indices.get(target_kind)
            if state is None or not state.id_to_label:
                continue
            top_k = min(int(k), len(state.id_to_label))
            labels, distances = state.index.knn_query(vector.reshape(1, -1), k=top_k)
            for label, distance in zip(labels[0], distances[0]):
                item_id = state.label_to_id.get(int(label))
                if item_id is None:
                    continue
                hits.append((item_id, 1.0 - float(distance)))
        hits.sort(key=lambda item: item[1], reverse=True)
        return hits[:k]

    def to_state_dict(self) -> Dict[str, Any]:
        """序列化"""
        return {
            "dim": self.dim,
            "m": self.m,
            "ef_construction": self.ef_construction,
            "ef_search": self.ef_search,
            "items": [
                {
                    "id": item_id,
                    "kind": self._kinds[item_id],
                    "vec": vec.astype(np.float32).tolist(),
                }
                for item_id, vec in self._vectors.items()
            ],
        }

    @classmethod
    def from_state_dict(
        cls,
        data: Dict[str, Any],
        *,
        build_index: bool = True,
    ) -> "VectorIndex":
        """反序列化"""
        obj = cls(
            dim=int(data["dim"]),
            m=int(data.get("m", 32)),
            ef_construction=int(data.get("ef_construction", 200)),
            ef_search=int(data.get("ef_search", 128)),
        )
        for item in data.get("items", []):
            item_id = str(item["id"])
            vector = np.asarray(item["vec"], dtype=np.float32)
            kind = str(item["kind"])
            obj._vectors[item_id] = vector
            obj._kinds[item_id] = kind
        if build_index:
            obj.rebuild_indices()
        return obj


@dataclass(frozen=True)
class QueryPlan:
    """针对特定查询生成的执行计划：调整检索深度、加权系数等参数"""

    query_type: QueryType
    effective_k: int
    effective_max_iter: int
    effective_reseed_k: int
    effective_attractor_weight: float
    direct_weight: float
    query_keywords: List[str]
    aggregation_entities: List[str]
    time_range: Optional[TimeRange]
    is_aggregation: bool


USER_MARKERS = ("user:", "用户:", "user：", "用户：")


class FieldRetriever:
    """
    场论检索器：实现复杂的检索逻辑。
    算法步骤：
    1. 生成查询计划。
    2. 执行直接语义检索 + 个性化 PageRank (PR)。
    3. 执行 Mean-Shift 吸引子追踪，寻找记忆空间中的密度重心。
    4. 从吸引子出发执行“重播”式的 PR 计算。
    5. 融合各路得分，并根据 IdentityState 进行性格偏置。
    """

    def __init__(
        self,
        metric: LowRankMetric,
        vindex: VectorIndex,
        graph: MemoryGraph,
        *,
        query_analyzer: BaseQueryAnalyzer,
        ppr_damping: float = 0.85,
        ppr_max_iter: int = 60,
        ppr_tol: float = 1e-6,
        weights: Optional[RetrievalWeights] = None,
    ):
        self.metric = metric
        self.vindex = vindex
        self.graph = graph
        self.query_analyzer = query_analyzer
        self.time_extractor = TimeRangeExtractor()
        self.fact_extractor = FactExtractor()
        self.ppr_damping = float(ppr_damping)
        self.ppr_max_iter = int(ppr_max_iter)
        self.ppr_tol = float(ppr_tol)
        self.weights = weights or RetrievalWeights()

    # --- 各种辅助 Payload 提取函数 ---
    def _payload_vec(self, payload: Any) -> Optional[np.ndarray]:
        if isinstance(payload, dict):
            for key in ("embedding", "centroid", "prototype"):
                vec = payload.get(key)
                if vec is not None:
                    return np.asarray(vec, dtype=np.float32)
            return None
        if hasattr(payload, "embedding"):
            vec = getattr(payload, "embedding")
            return None if vec is None else np.asarray(vec, dtype=np.float32)
        if hasattr(payload, "centroid"):
            vec = getattr(payload, "centroid")
            return None if vec is None else np.asarray(vec, dtype=np.float32)
        if hasattr(payload, "prototype"):
            vec = getattr(payload, "prototype")
            return None if vec is None else np.asarray(vec, dtype=np.float32)
        return None

    def _node_vec(self, node_id: str, payload: Any) -> Optional[np.ndarray]:
        vec = self._payload_vec(payload)
        if vec is not None:
            return np.asarray(vec, dtype=np.float32)
        index_vec = self.vindex.get_vector(node_id)
        if index_vec is None:
            return None
        return np.asarray(index_vec, dtype=np.float32)

    def _payload_text(self, payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ("semantic_text", "name", "description", "label"):
                value = payload.get(key)
                if value:
                    return str(value)
            return ""
        if isinstance(payload, Plot):
            return str(payload.semantic_text)
        if isinstance(payload, Summary):
            return str(payload.text)
        if isinstance(payload, Theme):
            return str(payload.name or payload.description or payload.label)
        return str(getattr(payload, "name", getattr(payload, "description", "")))

    def _payload_ts(self, payload: Any) -> float:
        if isinstance(payload, dict):
            for key in ("ts", "updated_ts", "created_ts"):
                value = payload.get(key)
                if value is not None:
                    return float(value)
            return 0.0
        for key in ("ts", "updated_ts", "created_ts"):
            value = getattr(payload, key, None)
            if value is not None:
                return float(value)
        return 0.0

    def _payload_mass(self, payload: Any) -> float:
        if payload is None:
            return 0.0
        if isinstance(payload, dict):
            mass = payload.get("mass")
            return float(mass) if mass is not None else 0.0
        if hasattr(payload, "mass"):
            return float(payload.mass())
        return 0.0

    def _payload_fact_keys(self, payload: Any) -> List[str]:
        if payload is None:
            return []
        if isinstance(payload, dict):
            return [str(item) for item in payload.get("fact_keys", [])]
        return [str(item) for item in getattr(payload, "fact_keys", [])]

    def _build_query_plan(
        self,
        query_text: str,
        kinds: Sequence[str],
        k: int,
        max_iter: int,
        reseed_k: int,
        attractor_weight: float,
        query_type: Optional[QueryType],
    ) -> QueryPlan:
        """根据查询内容构建执行计划"""
        analysis = self.query_analyzer.analyze(query_text)
        detected_type = query_type if query_type is not None else analysis.query_type

        effective_k = int(k)
        effective_max_iter = int(max_iter)
        effective_reseed_k = int(reseed_k)

        # 针对不同类型的查询调整检索深度
        if detected_type in {QueryType.MULTI_HOP, QueryType.CAUSAL}:  # 多关系推理：增加上下文广度
            effective_k = max(k, int(math.ceil(k * 1.5)))
            effective_max_iter += MULTI_HOP_EXTRA_PAGERANK_ITER
            effective_reseed_k = max(reseed_k, int(math.ceil(reseed_k * 1.2)))
        elif detected_type == QueryType.USER_FACT:  # 用户事实：更广泛的搜索
            effective_k = max(k, int(math.ceil(k * SINGLE_SESSION_USER_K_MULTIPLIER)))
            effective_reseed_k = max(reseed_k, int(math.ceil(reseed_k * 1.5)))

        effective_attractor_weight = float(attractor_weight)
        if detected_type == QueryType.FACTUAL:
            effective_attractor_weight = FACTUAL_ATTRACTOR_WEIGHT
        elif detected_type == QueryType.USER_FACT:
            effective_attractor_weight = FACTUAL_ATTRACTOR_WEIGHT * 0.8

        return QueryPlan(
            query_type=detected_type,
            effective_k=effective_k,
            effective_max_iter=effective_max_iter,
            effective_reseed_k=effective_reseed_k,
            effective_attractor_weight=effective_attractor_weight,
            direct_weight=1.0 - effective_attractor_weight,
            query_keywords=(
                analysis.query_keywords if detected_type in {QueryType.USER_FACT, QueryType.FACTUAL} else []
            ),
            aggregation_entities=list(analysis.aggregation_entities),
            time_range=self._extract_time_range(query_text, kinds, analysis),
            is_aggregation=bool(analysis.is_aggregation),
        )

    def _extract_time_range(
        self,
        query_text: str,
        kinds: Sequence[str],
        analysis: QueryAnalysis,
    ) -> Optional[TimeRange]:
        """针对时间相关查询，提取时间范围筛选器"""
        if analysis.query_type != QueryType.TEMPORAL:
            return None
        return self.time_extractor.extract(
            self._build_events_timeline(kinds),
            temporal_plan=analysis.temporal_plan,
        )

    def _build_events_timeline(self, kinds: Sequence[str]) -> List[Tuple[str, float]]:
        timeline: List[Tuple[str, float]] = []
        for node_id in self.graph.node_ids():
            if self.graph.kind(node_id) not in kinds:
                continue
            payload = self.graph.payload(node_id)
            ts = self._payload_ts(payload)
            if ts <= 0.0:
                continue
            timeline.append((self._payload_text(payload), ts))
        return timeline

    def _matches_time_range(self, node_id: str, time_range: Optional[TimeRange]) -> bool:
        if time_range is None or time_range.relation in {"any", "span"}:
            return True
        payload = self.graph.payload(node_id)
        ts = self._payload_ts(payload)
        if ts <= 0.0:
            return False
        if time_range.start is not None and ts < time_range.start:
            return False
        if time_range.end is not None and ts > time_range.end:
            return False
        return True

    def _apply_time_filter(
        self,
        ranked: List[Tuple[str, float, str]],
        time_range: TimeRange,
    ) -> List[Tuple[str, float, str]]:
        """应用时间筛选和排序（如“第一件事”、“最近的...”）"""
        if time_range.relation in {"any", "span"}:
            return ranked

        filtered = [item for item in ranked if self._matches_time_range(item[0], time_range)]
        if time_range.relation == "first":
            filtered.sort(key=lambda item: self._payload_ts(self.graph.payload(item[0])))
        elif time_range.relation == "last":
            filtered.sort(
                key=lambda item: self._payload_ts(self.graph.payload(item[0])), reverse=True
            )
        return filtered

    def _compute_keyword_boost(self, node_id: str, keywords: Sequence[str]) -> float:
        """计算关键词匹配加成"""
        if not keywords:
            return 0.0
        text_lower = self._payload_text(self.graph.payload(node_id)).lower()
        if not text_lower:
            return 0.0
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches == 0:
            return 0.0
        match_ratio = matches / len(keywords)
        if match_ratio < KEYWORD_MATCH_MIN_RATIO:
            return 0.0
        return KEYWORD_MATCH_BOOST * min(1.0, match_ratio * 1.5)

    def _compute_user_role_boost(self, node_id: str) -> float:
        """针对用户角色的优先加成"""
        text_lower = self._payload_text(self.graph.payload(node_id)).lower()
        return (
            USER_ROLE_PRIORITY_BOOST
            if any(marker in text_lower for marker in USER_MARKERS)
            else 0.0
        )

    def _compute_fact_key_boost(self, node_id: str, query_text: str) -> float:
        """针对事实匹配的加成：检查提取出的核心事实键是否重合"""
        payload = self.graph.payload(node_id)
        fact_keys = self._payload_fact_keys(payload)
        if not fact_keys:
            return 0.0

        query_facts = self.fact_extractor.extract(query_text)
        if not query_facts:
            return 0.0

        matches = 0.0
        lowered_fact_keys = [item.lower() for item in fact_keys]
        for fact in query_facts:
            query_fact_text = fact.fact_text.lower()
            for plot_fact_key in lowered_fact_keys:
                if query_fact_text == plot_fact_key:
                    matches += 1.0
                    break
                if query_fact_text in plot_fact_key or plot_fact_key in query_fact_text:
                    matches += 0.5
                    break
                if fact.entities and any(
                    entity.lower() in plot_fact_key for entity in fact.entities
                ):
                    matches += 0.3
                    break

        if matches == 0.0:
            return 0.0
        return min(FACT_KEY_BOOST_MAX, FACT_KEY_BOOST_MAX * matches / max(1, len(query_facts)))

    def _keyword_search(
        self,
        keywords: Sequence[str],
        kinds: Sequence[str],
        max_results: int = 100,
    ) -> List[Tuple[str, float, str]]:
        """基础关键词暴力检索，用于兜底和聚合类查询"""
        if not keywords:
            return []

        lowered = [keyword.lower() for keyword in keywords]
        ranked: List[Tuple[str, float, str]] = []
        for node_id in self.graph.node_ids():
            kind = self.graph.kind(node_id)
            if kind not in kinds:
                continue
            text_lower = self._payload_text(self.graph.payload(node_id)).lower()
            if not text_lower:
                continue
            matches = 0.0
            for keyword in lowered:
                if keyword in text_lower:
                    matches += 1.0
                elif len(keyword) >= 4 and keyword[:4] in text_lower:
                    matches += 0.5
            if matches < 1.0:
                continue
            ranked.append((node_id, matches / max(len(lowered), 1), kind))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:max_results]

    def _mean_shift(
        self,
        x0: np.ndarray,
        candidates: List[Tuple[str, np.ndarray, float]],
        steps: int = 8,
    ) -> List[np.ndarray]:
        """
        Mean-shift 吸引子追踪算法：
        在向量空间寻找记忆密度的重心。这能帮助系统跳出“字面相似度”，
        找到记忆中最核心、最稳固的那个语义区域（吸引子）。
        """
        if not candidates:
            return [x0]
        x = x0.copy()
        path = [x.copy()]
        for _ in range(steps):
            # 计算到所有候选项的平方距离
            d2s = [self.metric.d2(x, vec) for _, vec, _ in candidates]
            # 计算局部带宽
            sigma2 = float(np.median(d2s)) + 1e-6
            # 计算加权分布，融入记忆质量 (Mass)
            logits = [-(d2 / (2.0 * sigma2)) + mass for d2, (_, _, mass) in zip(d2s, candidates)]
            weights = softmax(logits)
            # 移动位置
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
        tol: float = 1e-6,
    ) -> Dict[str, float]:
        """执行基于稀疏邻接表的个性化 PageRank。"""
        personalized = {node: value for node, value in personalization.items() if node in self.graph}
        if not personalized:
            return {}
        cached = self.graph.get_cached_pagerank(personalized, damping, max_iter, tol)
        if cached is not None:
            return cached
        node_ids = self.graph.node_ids()
        if not node_ids:
            return {}
        index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        personalization_vec = np.zeros(len(node_ids), dtype=np.float64)
        for node_id, value in personalized.items():
            personalization_vec[index[node_id]] = max(0.0, float(value))
        total = float(personalization_vec.sum())
        if total <= 0.0:
            return {}
        personalization_vec /= total

        outgoing: List[List[Tuple[int, float]]] = [[] for _ in node_ids]
        dangling = np.ones(len(node_ids), dtype=bool)
        for src, dst, belief in self.graph.iter_edge_items(sign=1):
            src_idx = index.get(src)
            dst_idx = index.get(dst)
            if src_idx is None or dst_idx is None:
                continue
            outgoing[src_idx].append((dst_idx, max(1e-6, belief.pagerank_weight())))
            dangling[src_idx] = False

        transition: List[List[Tuple[int, float]]] = []
        for row in outgoing:
            total_weight = sum(weight for _, weight in row)
            if total_weight <= 0.0:
                transition.append([])
                continue
            transition.append([(dst_idx, weight / total_weight) for dst_idx, weight in row])

        rank = personalization_vec.copy()
        for _ in range(max_iter):
            next_rank = (1.0 - damping) * personalization_vec
            dangling_mass = damping * float(rank[dangling].sum())
            if dangling_mass > 0.0:
                next_rank += dangling_mass * personalization_vec
            for src_idx, row in enumerate(transition):
                if not row:
                    continue
                source_mass = damping * rank[src_idx]
                if source_mass <= 0.0:
                    continue
                for dst_idx, weight in row:
                    next_rank[dst_idx] += source_mass * weight
            delta = float(np.abs(next_rank - rank).sum())
            rank = next_rank
            if delta <= tol:
                break

        normalized_result = {
            node_id: float(rank[index[node_id]]) for node_id in node_ids if rank[index[node_id]] > 0.0
        }
        self.graph.set_cached_pagerank(personalized, damping, max_iter, tol, normalized_result)
        return normalized_result

    def _direct_semantic_search(
        self,
        query_emb: np.ndarray,
        kinds: Sequence[str],
        k: int,
        damping: float,
        max_iter: int,
        query_type: QueryType,
    ) -> List[Tuple[str, float, str]]:
        """执行直接语义检索并进行一轮 PR 扩散"""
        semantic_weight = FACTUAL_SEMANTIC_WEIGHT if query_type == QueryType.FACTUAL else 0.7
        personalization: Dict[str, float] = {}
        semantic_scores: Dict[str, float] = {}

        for kind in kinds:
            for node_id, similarity in self.vindex.search(query_emb, k=k * 2, kind=kind):
                if node_id not in self.graph:
                    continue
                if similarity > personalization.get(node_id, 0.0):
                    personalization[node_id] = similarity
                    semantic_scores[node_id] = similarity

        if not personalization:
            return []

        pagerank_scores = self._pagerank(personalization, damping=damping, max_iter=max_iter)
        pr_values = list(pagerank_scores.values())
        pr_max = max(pr_values) if pr_values else 1.0
        pr_min = min(pr_values) if pr_values else 0.0
        pr_range = pr_max - pr_min if pr_max > pr_min else 1.0
        pagerank_weight = 1.0 - semantic_weight

        ranked: List[Tuple[str, float, str]] = []
        for node_id, pr_score in pagerank_scores.items():
            kind = self.graph.kind(node_id)
            if kind not in kinds:
                continue
            payload = self.graph.payload(node_id)
            semantic_score = semantic_scores.get(node_id, 0.0)
            normalized_pr = (pr_score - pr_min) / pr_range if pr_range > 0 else 0.5
            blended = semantic_weight * semantic_score + pagerank_weight * normalized_pr
            blended += 1e-4 * self._payload_mass(payload)
            ranked.append((node_id, blended, kind))

        ranked.sort(key=lambda item: item[1], reverse=True)
        adjusted_ranked: List[Tuple[str, float, str]] = []
        stronger_ids: List[str] = []
        for node_id, blended, kind in ranked:
            penalty = self._negative_inhibition_penalty(node_id=node_id, stronger_ids=stronger_ids[:4])
            adjusted_ranked.append((node_id, max(0.0, blended - penalty), kind))
            stronger_ids.append(node_id)

        adjusted_ranked.sort(key=lambda item: item[1], reverse=True)
        return adjusted_ranked[:k]

    def _negative_inhibition_penalty(self, *, node_id: str, stronger_ids: Sequence[str]) -> float:
        anchor_ids = self.graph.nodes_of_kind("anchor")
        anchor_penalty = sum(
            self._negative_edge_strength(anchor_id, node_id)
            + self._negative_edge_strength(node_id, anchor_id)
            for anchor_id in anchor_ids
        )
        candidate_penalty = sum(
            self._negative_edge_strength(stronger_id, node_id)
            + self._negative_edge_strength(node_id, stronger_id)
            for stronger_id in stronger_ids
        )
        return min(
            0.25,
            self.weights.negative_anchor_penalty * anchor_penalty
            + self.weights.negative_candidate_penalty * candidate_penalty,
        )

    def _negative_edge_strength(self, src: str, dst: str) -> float:
        if not self.graph.has_edge(src, dst):
            return 0.0
        belief = self.graph.edge_belief(src, dst)
        if belief.sign >= 0:
            return 0.0
        return max(0.0, belief.weight) * max(0.0, belief.confidence) * max(0.5, belief.mean())

    def retrieve(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        state: IdentityState,
        kinds: Tuple[str, ...],
        k: int = 5,
    ) -> RetrievalTrace:
        """
        核心检索入口。
        """
        query_emb = np.asarray(query_embedding, dtype=np.float32)
        self_vec = (
            state.self_vector if getattr(state, "self_vector", None) is not None else query_emb
        )
        plan = self._build_query_plan(
            query_text=query_text,
            kinds=kinds,
            k=k,
            max_iter=self.ppr_max_iter,
            reseed_k=80,
            attractor_weight=self.weights.attractor_weight,
            query_type=None,
        )

        # 1. 语义路径：直接检索相关节点
        direct_ranked = self._direct_semantic_search(
            query_emb=query_emb,
            kinds=kinds,
            k=plan.effective_k,
            damping=self.weights.direct_damping,
            max_iter=plan.effective_max_iter,
            query_type=plan.query_type,
        )
        if plan.time_range is not None:
            direct_ranked = self._apply_time_filter(direct_ranked, plan.time_range)

        # 增强直接检索分数（事实、关键词、角色）
        enhanced_direct: List[Tuple[str, float, str]] = []
        for node_id, score, kind in direct_ranked:
            enhanced_score = float(score)
            if kind == "plot":
                enhanced_score += self._compute_fact_key_boost(node_id, query_text)
                if (
                    plan.query_type in {QueryType.USER_FACT, QueryType.FACTUAL}
                    and plan.query_keywords
                ):
                    enhanced_score += self._compute_keyword_boost(node_id, plan.query_keywords)
                if plan.query_type == QueryType.USER_FACT:
                    enhanced_score += self._compute_user_role_boost(node_id)
            enhanced_direct.append((node_id, enhanced_score, kind))

        # 2. 吸引子路径：执行物理场追踪
        candidates: List[Tuple[str, np.ndarray, float]] = []
        for kind in kinds:
            for item_id, sim in self.vindex.search(query_emb, k=60, kind=kind):
                if item_id not in self.graph:
                    continue
                if not self._matches_time_range(item_id, plan.time_range):
                    continue
                payload = self.graph.payload(item_id)
                node_vec = self._node_vec(item_id, payload)
                if node_vec is None:
                    continue
                # 计算身份一致性偏置：在警觉状态下，相关的威胁记忆会被“引力”增强
                identity_bonus = 0.0
                if isinstance(payload, Plot):
                    identity_bonus = (
                        self.weights.plot_alignment_bonus
                        * payload.frame.alignment_score(state.axis_state)
                    )
                mass = self._payload_mass(payload)
                candidates.append((item_id, node_vec, mass + identity_bonus + sim))

        path = self._mean_shift(query_emb, candidates, steps=8)
        attractor = path[-1]

        # 3. 从吸引子出发执行 PR 重播，获取“重心联想结果”
        personalization: Dict[str, float] = {}
        for kind in kinds:
            for item_id, sim in self.vindex.search(attractor, k=plan.effective_reseed_k, kind=kind):
                if not self._matches_time_range(item_id, plan.time_range):
                    continue
                personalization[item_id] = max(personalization.get(item_id, 0.0), sim)

        pr = self._pagerank(
            personalization,
            damping=self.weights.pagerank_damping,
            max_iter=plan.effective_max_iter,
            tol=self.ppr_tol,
        )
        attractor_ranked: List[Tuple[str, float, str]] = []
        for item_id, score in pr.items():
            kind = self.graph.kind(item_id)
            if kind not in kinds:
                continue
            payload = self.graph.payload(item_id)
            node_vec = self._node_vec(item_id, payload)
            bonus = self._payload_mass(payload)
            if isinstance(payload, Plot):
                # 再次应用身份偏置
                bonus += (
                    self.weights.attractor_plot_alignment_bonus
                    * payload.frame.alignment_score(state.axis_state)
                )
                bonus += self.weights.attractor_plot_confidence_bonus * payload.confidence
                bonus += self._compute_fact_key_boost(item_id, query_text)
                if (
                    plan.query_type in {QueryType.USER_FACT, QueryType.FACTUAL}
                    and plan.query_keywords
                ):
                    bonus += self._compute_keyword_boost(item_id, plan.query_keywords)
                if plan.query_type == QueryType.USER_FACT:
                    bonus += self._compute_user_role_boost(item_id)
            elif node_vec is not None:
                # 给故事/主题节点根据当前身份向量进行语义加权
                bonus += self.weights.story_theme_identity_bonus * float(
                    np.dot(l2_normalize(node_vec), l2_normalize(self_vec))
                )
            attractor_ranked.append((item_id, float(score) + 1e-3 * bonus, kind))
        attractor_ranked.sort(key=lambda item: item[1], reverse=True)

        # 4. 合并与排序
        keyword_ranked: List[Tuple[str, float, str]] = []
        if plan.is_aggregation and plan.aggregation_entities:
            keyword_ranked = self._keyword_search(plan.aggregation_entities, kinds=kinds, max_results=100)

        merged_scores: Dict[str, Tuple[float, str]] = {}
        for node_id, score, kind in enhanced_direct:
            merged_scores[node_id] = (plan.direct_weight * score, kind)
        for node_id, score, kind in attractor_ranked:
            current_score, current_kind = merged_scores.get(node_id, (0.0, kind))
            merged_scores[node_id] = (
                current_score + plan.effective_attractor_weight * score,
                current_kind,
            )
        if plan.is_aggregation:
            for node_id, score, kind in keyword_ranked:
                current_score, current_kind = merged_scores.get(node_id, (0.0, kind))
                merged_scores[node_id] = (current_score + 0.6 * score, current_kind)

        # 针对事实查询，提升 Plot 节点的优先级
        if plan.query_type == QueryType.FACTUAL:
            for node_id, (score, kind) in list(merged_scores.items()):
                if kind == "plot":
                    merged_scores[node_id] = (score + FACTUAL_PLOT_PRIORITY_BOOST, kind)

        ranked = [(node_id, score, kind) for node_id, (score, kind) in merged_scores.items()]
        ranked.sort(key=lambda item: item[1], reverse=True)
        if plan.time_range is not None:
            ranked = self._apply_time_filter(ranked, plan.time_range)

        return RetrievalTrace(
            query=query_text,
            query_emb=query_emb,
            attractor_path=path,
            ranked=ranked[:k],
            query_type=plan.query_type,
            time_range=plan.time_range,
        )
