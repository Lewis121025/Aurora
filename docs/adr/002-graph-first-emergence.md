# ADR 002: From Rigid Narrative Pipeline to Graph-First Emergence

## Status
Accepted

## Summary
Aurora now runs on a graph-first architecture. `MemoryGraph` is the single source of truth. `Plot` remains the only authoritative write-path memory entity. `StoryArc`, `Theme`, `Dream`, and `Repair` are derived views or graph operators. The legacy narrative write path, migration-only observability modes, and backward-compatibility shims have been removed.

## Context
Aurora's stated principle is "Emergence over Programming", but the current production path encodes emergence as a rigid pipeline: ingest, dissonance, repair, consolidation, dream, and mode transition. This makes `AuroraSoul` a highly coupled orchestration object and forces explicit maintenance of `Story`, `Theme`, `SubconsciousField`, and repair state.

The current design delivers useful behavior, but it does so by enforcing top-down psychological objects on the write path. That conflicts with the intended model of local interactions producing higher-order structure. It also makes the system harder to simplify, test, and evolve.

## Decision
1. `MemoryGraph` is the authoritative persisted memory structure.
2. Every persisted interaction writes exactly one `plot` node plus explicit local edges.
3. `Plot` remains authoritative and is not deprecated. `Plot.source` is treated as provenance, not as a pipeline stage controller.
4. The graph gains signed edges. Each edge carries `relation_type`, `sign`, `weight`, `confidence`, `provenance`, and timestamps.
5. Positive edges are used for retrieval diffusion. Negative edges are used for contradiction analysis and post-retrieval inhibition only in v1; they are not fed directly into PageRank.
6. `StoryArc` and `Theme` stop being write-path state. They become materialized read models derived from graph communities and cached for performance.
7. The initial community algorithm is Louvain. Leiden is deferred.
8. `Self` is represented by pinned core anchor nodes. `self_vector` is maintained only as a derived centroid of core anchors.
9. `Dream` becomes a background graph operator based on time-decayed random walks, long-jump novelty, and tension-biased traversal.
10. `Repair` becomes a background resolution operator that detects contradiction components touching core anchors and, when needed, writes a normal `plot` node with `source="repair"` plus bridging edges.
11. The hot write path is reduced to extraction, node write, bounded local edge write, contradiction detection on bounded neighbors, and lightweight ingest scoring.
12. Synchronous CRP clustering, synchronous repair candidate selection, synchronous dream orchestration via `SubconsciousField`, and synchronous mode transitions have been removed from production code.

## Hot-Path Defaults
1. On ingest, add temporal edges to the previous 2 persisted plots.
2. Add positive semantic edges to the top 3 neighbors above the configured similarity threshold.
3. Run contradiction detection against the top 10 semantic neighbors only.
4. Keep all expensive community building and synthetic narrative generation off the write path.

## Public Interfaces and Type Changes
- `MemoryGraph` adds signed-edge support and graph-versioned cache invalidation.
- `Plot` remains the authoritative write entity.
- `StoryArc` and `Theme` remain exposed as read models, but they are derived from graph state rather than stored as authority.
- `IdentitySnapshot` remains exposed, but its values are derived from core anchors, contradiction summaries, and materialized views.
- Aurora now exposes a single production architecture: `graph_first`.

## Consequences
- The production core becomes much smaller and less coupled.
- Higher-order narrative structure becomes genuinely emergent from graph dynamics.
- Retrieval, dreaming, and repair are easier to evolve independently.
- Current story/theme metadata used by retrieval must be rebuilt from derived communities without quality loss.
- Signed-edge support must be added before contradiction-driven repair can replace the current dissonance machinery.

## Implementation Plan
1. Phase 0: Add side-by-side instrumentation for retrieval overlap, latency, contradiction counts, and materialized-view freshness.
2. Phase 1: Upgrade `MemoryGraph` to signed edges, anchor support, and graph-versioned cache invalidation.
3. Phase 2: Refactor ingest into graph write plus bounded local linking and bounded contradiction detection.
4. Phase 3: Build Louvain-based `StoryArc` and `Theme` materialized views and make retrieval consume them as derived data.
5. Phase 4: Replace `SubconsciousField` with the graph dream operator and replace global dissonance/repair logic with contradiction-component scanning and resolution-node generation.
6. Phase 5: Remove legacy authority from CRP clustering, synchronous repair selection, synchronous dream orchestration, and synchronous mode transitions.

## Test Plan
- Ingestion always persists a graph node even when no derived story/theme exists yet.
- Local edge creation is bounded, deterministic, and mode-independent.
- The same graph produces stable materialized communities and stable derived metadata.
- Retrieval continues to work when negative edges exist and positive diffusion remains isolated.
- Contradiction penalties suppress conflicting memories without collapsing recall.
- Dream generation supports both ephemeral candidates and persisted dream plots with provenance.
- Repair generation creates resolution plots that bridge contradiction components touching core anchors.
- External query, identity, and stats APIs keep stable shapes during rollout.

## Acceptance Criteria
- `MemoryGraph` is the only authoritative persisted memory store.
- The write path no longer performs synchronous story/theme clustering or synchronous dream/repair orchestration.
- Latency remains within the existing production budget.

## Assumptions and Defaults
- Louvain is sufficient for the first rollout.
- Negative edges are analytic and inhibitory in v1, not direct diffusion weights.
- `Plot` is retained as the atomic write-path unit.
- Removed architecture modes and deprecated snapshot aliases are not supported.
