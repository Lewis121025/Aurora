from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from aurora.soul.models import Plot
from aurora.soul.retrieval import MemoryGraph
from aurora.utils.math_utils import cosine_sim
from aurora.utils.time_utils import now_ts


@dataclass(frozen=True)
class DreamCandidate:
    plot_ids: List[str]
    score: float
    resonance: float
    tags: List[str]


@dataclass(frozen=True)
class RepairTarget:
    anchor_id: str
    plot_ids: List[str]
    score: float


class GraphDreamOperator:
    def __init__(self, *, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def propose(
        self,
        *,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        samples: int,
        steps: int,
    ) -> List[DreamCandidate]:
        adjacency: Dict[str, Dict[str, float]] = {plot_id: {} for plot_id in plots.keys()}
        for src, dst, belief in graph.iter_edge_items(sign=1):
            if src not in plots or dst not in plots:
                continue
            weight = max(1e-6, belief.pagerank_weight())
            adjacency.setdefault(src, {})[dst] = adjacency.get(src, {}).get(dst, 0.0) + weight
            adjacency.setdefault(dst, {})[src] = adjacency.get(dst, {}).get(src, 0.0) + weight
        if not adjacency:
            return []

        ordered = sorted(
            plots.values(),
            key=lambda plot: (plot.tension + plot.contradiction, plot.ts),
            reverse=True,
        )
        if not ordered:
            return []
        seeds = ordered[: max(1, min(8, len(ordered)))]
        candidates: List[DreamCandidate] = []
        for _ in range(max(1, samples)):
            seed = seeds[int(self.rng.integers(0, len(seeds)))]
            current = seed.id
            walked = [current]
            for _step in range(max(1, steps)):
                neighbors = list(adjacency.get(current, {}).keys())
                if not neighbors:
                    break
                weights = []
                for neighbor in neighbors:
                    weight = float(adjacency[current].get(neighbor, 1.0))
                    age = max(1.0, now_ts() - plots[neighbor].ts)
                    decay = 1.0 / math.log1p(age)
                    novelty = 1.0 - max(0.0, cosine_sim(plots[current].embedding, plots[neighbor].embedding))
                    weights.append(max(1e-6, weight * (0.55 + 0.30 * novelty + 0.15 * decay)))
                current = neighbors[int(self.rng.choice(np.arange(len(neighbors)), p=np.asarray(weights) / np.sum(weights)))]
                if current not in walked:
                    walked.append(current)
            if len(walked) < 2:
                continue
            member_plots = [plots[plot_id] for plot_id in walked if plot_id in plots]
            tags: List[str] = []
            for plot in member_plots:
                for tag in plot.frame.tags[:6]:
                    if tag not in tags:
                        tags.append(tag)
            resonance = float(
                np.mean([plot.tension + 0.5 * plot.contradiction for plot in member_plots])
            )
            long_jump = 0.0
            if len(member_plots) >= 2:
                pair_scores = []
                for left, right in zip(member_plots, member_plots[1:]):
                    pair_scores.append(1.0 - max(0.0, cosine_sim(left.embedding, right.embedding)))
                long_jump = float(np.mean(pair_scores)) if pair_scores else 0.0
            score = resonance * (0.7 + 0.3 * long_jump)
            candidates.append(
                DreamCandidate(
                    plot_ids=[plot.id for plot in member_plots],
                    score=score,
                    resonance=resonance,
                    tags=tags[:10],
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        deduped: List[DreamCandidate] = []
        seen = set()
        for candidate in candidates:
            key = tuple(candidate.plot_ids)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped


class GraphRepairOperator:
    def find_targets(
        self,
        *,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        anchor_ids: Sequence[str],
        limit: int = 1,
    ) -> List[RepairTarget]:
        neg: Dict[str, set[str]] = {}
        for src, dst, belief in graph.iter_edge_items(sign=-1):
            if belief.sign >= 0:
                continue
            neg.setdefault(src, set()).add(dst)
            neg.setdefault(dst, set()).add(src)

        targets: List[RepairTarget] = []
        for anchor_id in anchor_ids:
            if anchor_id not in neg:
                continue
            component = set()
            stack = [anchor_id]
            while stack:
                node_id = stack.pop()
                if node_id in component:
                    continue
                component.add(node_id)
                stack.extend(neighbor for neighbor in neg.get(node_id, ()) if neighbor not in component)
            component_plot_ids = [node_id for node_id in component if node_id in plots]
            if not component_plot_ids:
                continue
            score = float(
                np.mean(
                    [
                        plots[plot_id].contradiction + 0.5 * plots[plot_id].tension
                        for plot_id in component_plot_ids
                    ]
                )
            )
            targets.append(
                RepairTarget(anchor_id=anchor_id, plot_ids=component_plot_ids, score=score)
            )
        targets.sort(key=lambda item: item.score, reverse=True)
        return targets[:limit]
