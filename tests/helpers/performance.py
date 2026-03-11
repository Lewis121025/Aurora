from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import numpy as np

from aurora.integrations.embeddings.hash import HashEmbedding
from aurora.integrations.storage.runtime_store import SQLiteRuntimeStore
from aurora.runtime.runtime import RUNTIME_SCHEMA_VERSION
from aurora.soul.engine import AuroraSoul, SOUL_MEMORY_STATE_VERSION, SoulConfig
from aurora.soul.extractors import CombinatorialNarrativeProvider, HeuristicMeaningProvider
from aurora.soul.models import HOMEOSTATIC_AXES, IdentityState, Message, TextPart
from aurora.soul.query import BaseQueryAnalyzer, QueryAnalysis, QueryType
from aurora.soul.retrieval import FieldRetriever, LowRankMetric, MemoryGraph, VectorIndex
from aurora.soul.tuning import RetrievalWeights
from aurora.utils.jsonx import dumps


class StaticQueryAnalyzer(BaseQueryAnalyzer):
    def analyze(self, query_text: str) -> QueryAnalysis:
        return QueryAnalysis(
            query_type=QueryType.FACTUAL,
            query_keywords=[token for token in query_text.split() if token][:3],
        )


def build_identity_state(dim: int) -> IdentityState:
    axis_state = {name: 0.2 for name, _, _, _ in HOMEOSTATIC_AXES}
    self_vector = np.ones(dim, dtype=np.float32)
    self_vector /= np.linalg.norm(self_vector)
    return IdentityState(
        self_vector=self_vector,
        axis_state=axis_state,
        intuition_axes={name: 0.0 for name in axis_state},
        current_mode_label="origin",
    )


@dataclass(frozen=True)
class RetrievalFixture:
    retriever: FieldRetriever
    identity: IdentityState
    vectors: Dict[str, np.ndarray]
    graph: MemoryGraph
    queries: List[np.ndarray]


def build_retrieval_fixture(
    *,
    dim: int = 64,
    node_count: int = 4000,
    query_count: int = 20,
    seed: int = 123,
) -> RetrievalFixture:
    rng = np.random.default_rng(seed)
    metric = LowRankMetric(dim=dim, rank=min(16, dim), seed=seed)
    vindex = VectorIndex(dim=dim, ef_search=128)
    graph = MemoryGraph()
    retriever = FieldRetriever(
        metric=metric,
        vindex=vindex,
        graph=graph,
        query_analyzer=StaticQueryAnalyzer(),
        weights=RetrievalWeights(),
    )

    vectors: Dict[str, np.ndarray] = {}
    for i in range(node_count):
        vector = rng.standard_normal(dim).astype(np.float32)
        vector /= np.linalg.norm(vector)
        node_id = f"plot_{i}"
        vectors[node_id] = vector
        vindex.add(node_id, vector, kind="plot")
        graph.add_node(
            node_id,
            "plot",
            {
                "id": node_id,
                "semantic_text": f"topic {i % 37} cluster {i % 11}",
                "embedding": vector,
                "mass": 1.0 + float((i % 9) * 0.05),
                "fact_keys": [f"topic:{i % 37}", f"cluster:{i % 11}"],
                "ts": float(i),
            },
        )

    for i in range(node_count):
        for step in (1, 2, 7):
            neighbor = (i + step) % node_count
            graph.ensure_edge(
                f"plot_{i}",
                f"plot_{neighbor}",
                "semantic",
                weight=0.5 + 0.1 * step,
                confidence=0.9,
                provenance="perf_fixture",
            )

    queries: List[np.ndarray] = []
    node_ids = list(vectors.keys())
    for _ in range(query_count):
        anchor_id = node_ids[int(rng.integers(0, len(node_ids)))]
        base = vectors[anchor_id]
        query = base + 0.03 * rng.standard_normal(dim).astype(np.float32)
        query /= np.linalg.norm(query)
        queries.append(query.astype(np.float32))

    return RetrievalFixture(
        retriever=retriever,
        identity=build_identity_state(dim),
        vectors=vectors,
        graph=graph,
        queries=queries,
    )


def exact_knn(
    *,
    query: np.ndarray,
    vectors: Dict[str, np.ndarray],
    k: int,
) -> List[tuple[str, float]]:
    scores = [(node_id, float(np.dot(query, vector))) for node_id, vector in vectors.items()]
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores[:k]


def legacy_exact_pagerank(
    *,
    graph: MemoryGraph,
    personalization: Dict[str, float],
    damping: float,
    max_iter: int,
    tol: float,
) -> Dict[str, float]:
    personalized = {node_id: value for node_id, value in personalization.items() if node_id in graph}
    if not personalized:
        return {}
    node_ids = graph.node_ids()
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    personalization_vec = np.zeros(len(node_ids), dtype=np.float64)
    for node_id, value in personalized.items():
        personalization_vec[index[node_id]] = max(0.0, float(value))
    total = float(personalization_vec.sum())
    if total <= 0.0:
        return {}
    personalization_vec /= total

    outgoing: List[List[tuple[int, float]]] = [[] for _ in node_ids]
    dangling = np.ones(len(node_ids), dtype=bool)
    for src, dst, belief in graph.iter_edge_items(sign=1):
        src_idx = index.get(src)
        dst_idx = index.get(dst)
        if src_idx is None or dst_idx is None:
            continue
        outgoing[src_idx].append((dst_idx, max(1e-6, belief.pagerank_weight())))
        dangling[src_idx] = False

    rank = personalization_vec.copy()
    for _ in range(max_iter):
        next_rank = (1.0 - damping) * personalization_vec
        dangling_mass = damping * float(rank[dangling].sum())
        if dangling_mass > 0.0:
            next_rank += dangling_mass * personalization_vec
        for src_idx, row in enumerate(outgoing):
            if not row:
                continue
            total_weight = sum(weight for _, weight in row)
            if total_weight <= 0.0:
                continue
            source_mass = damping * rank[src_idx]
            if source_mass <= 0.0:
                continue
            inv_total = 1.0 / total_weight
            for dst_idx, weight in row:
                next_rank[dst_idx] += source_mass * weight * inv_total
        delta = float(np.abs(next_rank - rank).sum())
        rank = next_rank
        if delta <= tol:
            break

    return {
        node_id: float(rank[index[node_id]])
        for node_id in node_ids
        if rank[index[node_id]] > 0.0
    }


def legacy_exact_retrieve(
    *,
    fixture: RetrievalFixture,
    query_embedding: np.ndarray,
    k: int,
) -> List[tuple[str, float, str]]:
    retriever = fixture.retriever
    semantic_top = exact_knn(query=query_embedding, vectors=fixture.vectors, k=k * 2)
    personalization = {node_id: score for node_id, score in semantic_top}
    semantic_scores = dict(semantic_top)
    direct_pr = legacy_exact_pagerank(
        graph=fixture.graph,
        personalization=personalization,
        damping=retriever.weights.direct_damping,
        max_iter=retriever.ppr_max_iter,
        tol=retriever.ppr_tol,
    )
    direct_scores: Dict[str, float] = {}
    if direct_pr:
        pr_values = list(direct_pr.values())
        pr_min = min(pr_values)
        pr_max = max(pr_values)
        pr_range = pr_max - pr_min if pr_max > pr_min else 1.0
        for node_id, pr_score in direct_pr.items():
            normalized_pr = (pr_score - pr_min) / pr_range if pr_range > 0.0 else 0.5
            direct_scores[node_id] = 0.7 * semantic_scores.get(node_id, 0.0) + 0.3 * normalized_pr

    candidates = []
    for node_id, similarity in exact_knn(query=query_embedding, vectors=fixture.vectors, k=60):
        payload = fixture.graph.payload(node_id)
        candidates.append(
            (
                node_id,
                payload["embedding"],
                float(payload["mass"]) + similarity,
            )
        )
    attractor = retriever._mean_shift(query_embedding, candidates, steps=8)[-1]
    attractor_personalization = {
        node_id: score for node_id, score in exact_knn(query=attractor, vectors=fixture.vectors, k=80)
    }
    attractor_pr = legacy_exact_pagerank(
        graph=fixture.graph,
        personalization=attractor_personalization,
        damping=retriever.weights.pagerank_damping,
        max_iter=retriever.ppr_max_iter,
        tol=retriever.ppr_tol,
    )

    merged: Dict[str, tuple[float, str]] = {}
    for node_id, score in direct_scores.items():
        merged[node_id] = ((1.0 - retriever.weights.attractor_weight) * score, "plot")
    for node_id, score in attractor_pr.items():
        current_score, kind = merged.get(node_id, (0.0, "plot"))
        merged[node_id] = (
            current_score + retriever.weights.attractor_weight * score,
            kind,
        )

    ranked = [(node_id, score, kind) for node_id, (score, kind) in merged.items()]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:k]


def overlap_at_k(left: Sequence[str], right: Sequence[str], k: int) -> float:
    limit = max(1, min(k, len(left), len(right)))
    return len(set(left[:limit]) & set(right[:limit])) / float(limit)


def spearman_rank_correlation(
    left: Dict[str, float],
    right: Dict[str, float],
    *,
    top_k: int = 200,
) -> float:
    keys = list({*left.keys(), *right.keys()})
    if not keys:
        return 1.0
    left_ranked = sorted(left.items(), key=lambda item: item[1], reverse=True)[:top_k]
    right_ranked = sorted(right.items(), key=lambda item: item[1], reverse=True)[:top_k]
    ranked_keys = list({key for key, _ in left_ranked} | {key for key, _ in right_ranked})
    if len(ranked_keys) < 2:
        return 1.0
    left_order = {key: idx for idx, (key, _) in enumerate(left_ranked)}
    right_order = {key: idx for idx, (key, _) in enumerate(right_ranked)}
    fallback = len(ranked_keys) + 1
    diffs = [
        float(left_order.get(key, fallback) - right_order.get(key, fallback))
        for key in ranked_keys
    ]
    n = float(len(ranked_keys))
    numerator = 6.0 * sum(diff * diff for diff in diffs)
    denominator = n * (n * n - 1.0)
    if denominator <= 0.0:
        return 1.0
    return 1.0 - numerator / denominator


def median_seconds(fn: Callable[[], object], *, repeat: int) -> float:
    samples: List[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    samples.sort()
    return float(samples[len(samples) // 2])


def build_perf_memory(*, seed: int, plot_count: int) -> AuroraSoul:
    embedder = HashEmbedding(dim=64, seed=seed)
    return AuroraSoul(
        cfg=SoulConfig(dim=64, metric_rank=16, max_plots=max(512, plot_count * 2)),
        seed=seed,
        event_embedder=embedder,
        axis_embedder=embedder,
        meaning_provider=HeuristicMeaningProvider(),
        narrator=CombinatorialNarrativeProvider(),
        query_analyzer=StaticQueryAnalyzer(),
    )


@dataclass(frozen=True)
class RestoreFixture:
    memory: AuroraSoul
    store: SQLiteRuntimeStore
    ann_dir: Path
    snapshot_path: Path
    seed: int
    last_projected_seq: int

    def restore_structured(self) -> AuroraSoul:
        derived = self.store.load_derived_state()
        if derived is None:
            raise AssertionError("structured derived state is missing")
        derived["cfg"] = self.memory.cfg.to_state_dict()
        restored = AuroraSoul.from_state_dict(
            derived,
            event_embedder=HashEmbedding(dim=64, seed=self.seed),
            axis_embedder=HashEmbedding(dim=64, seed=self.seed),
            meaning_provider=HeuristicMeaningProvider(),
            narrator=CombinatorialNarrativeProvider(),
            query_analyzer=StaticQueryAnalyzer(),
            skip_vindex_rebuild=True,
            refresh_views=False,
        )
        if not restored.vindex.try_load_sidecar(
            str(self.ann_dir),
            epoch=self.last_projected_seq,
        ):
            restored.vindex.rebuild_indices()
        return restored

    def restore_legacy_json_snapshot(self) -> AuroraSoul:
        with self.snapshot_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        return AuroraSoul.from_state_dict(
            state,
            event_embedder=HashEmbedding(dim=64, seed=self.seed),
            axis_embedder=HashEmbedding(dim=64, seed=self.seed),
            meaning_provider=HeuristicMeaningProvider(),
            narrator=CombinatorialNarrativeProvider(),
            query_analyzer=StaticQueryAnalyzer(),
        )


def build_restore_fixture(
    *,
    base_dir: Path,
    plot_count: int = 250,
    seed: int = 7,
    message_repeat: int = 120,
) -> RestoreFixture:
    memory = build_perf_memory(seed=seed, plot_count=plot_count)
    repeated_tail = " ".join(
        ["long-lived memory coherence graph retrieval persistence theme clustering"]
        * message_repeat
    )
    for index in range(plot_count):
        memory.ingest(
            [
                Message(
                    role="user",
                    parts=(
                        TextPart(
                            text=(
                                f"Turn {index}: preference {index % 17}, cue {index % 29}. "
                                f"{repeated_tail}"
                            )
                        ),
                    ),
                ),
                Message(
                    role="assistant",
                    parts=(
                        TextPart(
                            text=(
                                "Acknowledged. I will consolidate this long-term preference "
                                f"and context. {repeated_tail}"
                            )
                        ),
                    ),
                ),
            ]
        )

    store = SQLiteRuntimeStore(
        str(base_dir / "runtime.db"),
        runtime_schema_version=RUNTIME_SCHEMA_VERSION,
        memory_state_version=SOUL_MEMORY_STATE_VERSION,
    )
    store.replace_derived_state(
        mem=memory,
        last_projected_seq=plot_count,
        runtime_state=store.get_runtime_state(),
    )
    ann_dir = base_dir / "ann_index_v7"
    memory.vindex.save_sidecar(str(ann_dir), epoch=plot_count)
    snapshot_path = base_dir / "legacy_snapshot.json"
    snapshot_path.write_text(dumps(memory.to_state_dict()), encoding="utf-8")
    return RestoreFixture(
        memory=memory,
        store=store,
        ann_dir=ann_dir,
        snapshot_path=snapshot_path,
        seed=seed,
        last_projected_seq=plot_count,
    )
