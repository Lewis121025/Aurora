from __future__ import annotations

import pytest

from tests.helpers.performance import (
    build_restore_fixture,
    build_retrieval_fixture,
    exact_knn,
    legacy_exact_pagerank,
    legacy_exact_retrieve,
    median_seconds,
    overlap_at_k,
    spearman_rank_correlation,
)


@pytest.fixture(scope="module")
def retrieval_fixture():
    return build_retrieval_fixture()


def test_ann_recall_at_10_against_exact_oracle(retrieval_fixture) -> None:
    recalls = []
    for query in retrieval_fixture.queries:
        exact_ids = {
            node_id
            for node_id, _ in exact_knn(
                query=query,
                vectors=retrieval_fixture.vectors,
                k=10,
            )
        }
        ann_ids = {
            node_id
            for node_id, _ in retrieval_fixture.retriever.vindex.search(
                query,
                k=10,
                kind="plot",
            )
        }
        recalls.append(len(exact_ids & ann_ids) / 10.0)

    avg_recall = sum(recalls) / len(recalls)
    assert avg_recall >= 0.98


def test_query_hot_path_overlap_and_speedup_against_exact_oracle(retrieval_fixture) -> None:
    overlaps = []
    for query in retrieval_fixture.queries[:3]:
        current = retrieval_fixture.retriever.retrieve(
            query_text="topic query",
            query_embedding=query,
            state=retrieval_fixture.identity,
            kinds=("plot",),
            k=20,
        ).ranked
        legacy = legacy_exact_retrieve(
            fixture=retrieval_fixture,
            query_embedding=query,
            k=20,
        )
        overlaps.append(
            overlap_at_k(
                [node_id for node_id, _, _ in current],
                [node_id for node_id, _, _ in legacy],
                20,
            )
        )

    hot_query = retrieval_fixture.queries[0]
    for _ in range(3):
        retrieval_fixture.retriever.retrieve(
            query_text="topic query",
            query_embedding=hot_query,
            state=retrieval_fixture.identity,
            kinds=("plot",),
            k=10,
        )

    current_seconds = median_seconds(
        lambda: retrieval_fixture.retriever.retrieve(
            query_text="topic query",
            query_embedding=hot_query,
            state=retrieval_fixture.identity,
            kinds=("plot",),
            k=10,
        ),
        repeat=15,
    )
    legacy_seconds = median_seconds(
        lambda: legacy_exact_retrieve(
            fixture=retrieval_fixture,
            query_embedding=hot_query,
            k=10,
        ),
        repeat=5,
    )

    avg_overlap = sum(overlaps) / len(overlaps)
    assert avg_overlap >= 0.95
    assert legacy_seconds / current_seconds >= 5.0


def test_sparse_ppr_matches_legacy_oracle_and_beats_it_on_hot_path(retrieval_fixture) -> None:
    seeds = {f"plot_{index}": 1.0 for index in range(10)}
    retrieval_fixture.graph.clear_pagerank_cache()
    current = retrieval_fixture.retriever._pagerank(
        seeds,
        damping=0.85,
        max_iter=60,
        tol=1e-6,
    )
    legacy = legacy_exact_pagerank(
        graph=retrieval_fixture.graph,
        personalization=seeds,
        damping=0.85,
        max_iter=60,
        tol=1e-6,
    )

    current_spearman = spearman_rank_correlation(current, legacy, top_k=200)

    hot_current_seconds = median_seconds(
        lambda: [
            retrieval_fixture.retriever._pagerank(
                seeds,
                damping=0.85,
                max_iter=60,
                tol=1e-6,
            )
            for _ in range(50)
        ],
        repeat=7,
    ) / 50.0
    legacy_seconds = median_seconds(
        lambda: [
            legacy_exact_pagerank(
                graph=retrieval_fixture.graph,
                personalization=seeds,
                damping=0.85,
                max_iter=60,
                tol=1e-6,
            )
            for _ in range(5)
        ],
        repeat=3,
    ) / 5.0

    assert current_spearman >= 0.98
    assert legacy_seconds / hot_current_seconds >= 4.0


def test_structured_restore_outperforms_legacy_json_snapshot(tmp_path) -> None:
    fixture = build_restore_fixture(base_dir=tmp_path)
    try:
        fixture.restore_structured()
        fixture.restore_legacy_json_snapshot()

        structured_seconds = median_seconds(fixture.restore_structured, repeat=5)
        legacy_seconds = median_seconds(fixture.restore_legacy_json_snapshot, repeat=3)

        assert fixture.snapshot_path.stat().st_size > 0
        assert legacy_seconds / structured_seconds >= 1.5
    finally:
        fixture.store.close()
