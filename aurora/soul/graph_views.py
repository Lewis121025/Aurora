from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import igraph as ig
import numpy as np

from aurora.soul.models import Plot, StoryArc, Theme, l2_normalize
from aurora.soul.retrieval import MemoryGraph, VectorIndex
from aurora.utils.math_utils import cosine_sim
from aurora.utils.time_utils import now_ts


@dataclass
class GraphViewStats:
    graph_edge_version: int = -1
    refreshed_step: int = 0
    plot_count: int = 0
    story_count: int = 0
    theme_count: int = 0

    def to_state_dict(self) -> Dict[str, int]:
        return {
            "graph_edge_version": int(self.graph_edge_version),
            "refreshed_step": int(self.refreshed_step),
            "plot_count": int(self.plot_count),
            "story_count": int(self.story_count),
            "theme_count": int(self.theme_count),
        }

    @classmethod
    def from_state_dict(cls, data: Dict[str, int]) -> "GraphViewStats":
        return cls(
            graph_edge_version=int(data.get("graph_edge_version", -1)),
            refreshed_step=int(data.get("refreshed_step", 0)),
            plot_count=int(data.get("plot_count", 0)),
            story_count=int(data.get("story_count", 0)),
            theme_count=int(data.get("theme_count", 0)),
        )


def _stable_id(prefix: str, members: Iterable[str]) -> str:
    ordered = sorted(str(member) for member in members)
    digest = hashlib.md5("|".join(ordered).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _safe_centroid(vecs: Sequence[np.ndarray], dim: int) -> np.ndarray:
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    centroid = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
    return l2_normalize(centroid.astype(np.float32))


class GraphViewBuilder:
    def __init__(self, *, seed: int = 0):
        self.seed = seed

    def build(
        self,
        *,
        graph: MemoryGraph,
        vindex: VectorIndex,
        plots: Dict[str, Plot],
        previous_stories: Dict[str, StoryArc],
        previous_themes: Dict[str, Theme],
        step: int,
    ) -> tuple[Dict[str, StoryArc], Dict[str, Theme], GraphViewStats]:
        stories = self._build_stories(graph=graph, plots=plots, previous=previous_stories, dim=vindex.dim)
        themes = self._build_themes(stories=stories, previous=previous_themes, dim=vindex.dim)
        self._sync_view_nodes(graph=graph, vindex=vindex, stories=stories, themes=themes, plots=plots)
        stats = GraphViewStats(
            graph_edge_version=graph.edge_version,
            refreshed_step=int(step),
            plot_count=len(plots),
            story_count=len(stories),
            theme_count=len(themes),
        )
        return stories, themes, stats

    def _louvain_communities(
        self,
        *,
        node_ids: Sequence[str],
        weighted_edges: Sequence[tuple[str, str, float]],
    ) -> List[set[str]]:
        if not node_ids:
            return []
        if not weighted_edges:
            return [{str(node_id)} for node_id in node_ids]
        id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        graph = ig.Graph(
            n=len(node_ids),
            edges=[(id_to_index[src], id_to_index[dst]) for src, dst, _ in weighted_edges],
            directed=False,
        )
        graph.es["weight"] = [float(weight) for _, _, weight in weighted_edges]
        communities = graph.community_multilevel(weights=graph.es["weight"])
        return [{str(node_ids[index]) for index in community} for community in communities]

    def _build_stories(
        self,
        *,
        graph: MemoryGraph,
        plots: Dict[str, Plot],
        previous: Dict[str, StoryArc],
        dim: int,
    ) -> Dict[str, StoryArc]:
        edge_weights: Dict[tuple[str, str], float] = {}
        for src, dst, belief in graph.iter_edge_items(sign=1):
            if src not in plots or dst not in plots:
                continue
            key = tuple(sorted((src, dst)))
            edge_weights[key] = edge_weights.get(key, 0.0) + max(
                1e-6, float(belief.weight) * float(belief.mean())
            )
        communities = self._louvain_communities(
            node_ids=list(plots.keys()),
            weighted_edges=[(src, dst, weight) for (src, dst), weight in edge_weights.items()],
        )
        story_map: Dict[str, StoryArc] = {}
        for plot in plots.values():
            plot.story_id = None
            plot.theme_id = None
        for members in sorted(
            communities,
            key=lambda community: min(plots[plot_id].ts for plot_id in community),
        ):
            member_plots = sorted((plots[plot_id] for plot_id in members), key=lambda plot: plot.ts)
            story_id = _stable_id("story", members)
            base = previous.get(story_id)
            centroid = _safe_centroid([plot.embedding for plot in member_plots], dim)
            story = StoryArc(
                id=story_id,
                created_ts=member_plots[0].ts,
                updated_ts=member_plots[-1].ts,
                plot_ids=[plot.id for plot in member_plots],
                centroid=centroid,
                reference_count=0 if base is None else base.reference_count,
                status="active",
            )
            for plot in member_plots:
                story.actor_counts.update(
                    {actor: story.actor_counts.get(actor, 0) + 1 for actor in plot.actors}
                )
                story.source_counts[plot.source] = story.source_counts.get(plot.source, 0) + 1
                for tag in plot.frame.tags[:8]:
                    story.tag_counts[tag] = story.tag_counts.get(tag, 0) + 1
                story.tension_curve.append(float(plot.tension))
                story.unresolved_energy += float(plot.contradiction)
                plot.story_id = story.id
            dists = [float(np.linalg.norm(plot.embedding - centroid)) for plot in member_plots]
            if dists:
                story.dist_n = len(dists)
                story.dist_mean = float(np.mean(dists))
                story.dist_m2 = float(np.var(dists, ddof=1) * (len(dists) - 1)) if len(dists) > 1 else 0.0
            gaps = [
                max(0.0, member_plots[idx].ts - member_plots[idx - 1].ts)
                for idx in range(1, len(member_plots))
            ]
            if gaps:
                story.gap_n = len(gaps)
                story.gap_mean = float(np.mean(gaps))
                story.gap_m2 = float(np.var(gaps, ddof=1) * (len(gaps) - 1)) if len(gaps) > 1 else 0.0
            story_map[story.id] = story
        return story_map

    def _build_themes(
        self,
        *,
        stories: Dict[str, StoryArc],
        previous: Dict[str, Theme],
        dim: int,
    ) -> Dict[str, Theme]:
        if not stories:
            return {}
        edge_weights: Dict[tuple[str, str], float] = {}
        story_items = list(stories.values())
        for idx, left in enumerate(story_items):
            if left.centroid is None:
                continue
            left_tags = set(left.tag_counts.keys())
            for right in story_items[idx + 1 :]:
                if right.centroid is None:
                    continue
                sim = cosine_sim(left.centroid, right.centroid)
                shared_tags = len(left_tags & set(right.tag_counts.keys()))
                weight = max(0.0, sim) + min(shared_tags, 3) * 0.08
                if weight >= 0.45:
                    edge_weights[tuple(sorted((left.id, right.id)))] = weight
        communities = self._louvain_communities(
            node_ids=list(stories.keys()),
            weighted_edges=[(src, dst, weight) for (src, dst), weight in edge_weights.items()],
        )
        theme_map: Dict[str, Theme] = {}
        for members in communities:
            member_stories = [stories[story_id] for story_id in sorted(members)]
            theme_id = _stable_id("theme", members)
            base = previous.get(theme_id)
            member_centroids = [story.centroid for story in member_stories if story.centroid is not None]
            prototype = _safe_centroid(member_centroids, dim)
            tag_counts: Dict[str, int] = {}
            for story in member_stories:
                for tag, count in story.tag_counts.items():
                    tag_counts[tag] = tag_counts.get(tag, 0) + count
            label = " / ".join(
                tag for tag, _count in sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)[:3]
            ) or "emergent_theme"
            theme = Theme(
                id=theme_id,
                created_ts=now_ts() if base is None else base.created_ts,
                updated_ts=now_ts(),
                story_ids=[story.id for story in member_stories],
                prototype=prototype,
                a=1.0 if base is None else base.a,
                b=1.0 if base is None else base.b,
                label=label,
                name=label,
                description=f"Emergent theme around: {label}",
            )
            theme_map[theme.id] = theme
        return theme_map

    def _sync_view_nodes(
        self,
        *,
        graph: MemoryGraph,
        vindex: VectorIndex,
        stories: Dict[str, StoryArc],
        themes: Dict[str, Theme],
        plots: Dict[str, Plot],
    ) -> None:
        for node_id in list(graph.nodes_of_kind("story")):
            graph.remove_node(node_id)
            vindex.remove(node_id)
        for node_id in list(graph.nodes_of_kind("theme")):
            graph.remove_node(node_id)
            vindex.remove(node_id)

        for story in stories.values():
            graph.add_node(story.id, "story", story)
            centroid = story.centroid if story.centroid is not None else np.zeros(vindex.dim, dtype=np.float32)
            vindex.add(story.id, centroid, kind="story")
            for plot_id in story.plot_ids:
                if plot_id in plots:
                    graph.ensure_edge(plot_id, story.id, "belongs_to")
                    graph.ensure_edge(story.id, plot_id, "contains")

        theme_story_ids: Dict[str, str] = {}
        for theme in themes.values():
            graph.add_node(theme.id, "theme", theme)
            prototype = theme.prototype if theme.prototype is not None else np.zeros(vindex.dim, dtype=np.float32)
            vindex.add(theme.id, prototype, kind="theme")
            for story_id in theme.story_ids:
                theme_story_ids[story_id] = theme.id
                graph.ensure_edge(story_id, theme.id, "instantiates")
                graph.ensure_edge(theme.id, story_id, "grounds")

        for story in stories.values():
            theme_id = theme_story_ids.get(story.id)
            if theme_id is None:
                continue
            for plot_id in story.plot_ids:
                plot = plots.get(plot_id)
                if plot is None:
                    continue
                plot.theme_id = theme_id
                graph.ensure_edge(plot.id, theme_id, "suggests_theme")
                graph.ensure_edge(theme_id, plot.id, "evidenced_by")
