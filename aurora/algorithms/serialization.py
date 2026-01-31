"""
AURORA Serialization Module
===========================

State serialization and deserialization functionality.

Key responsibilities:
- Serialize AuroraMemory state to JSON-compatible dict
- Deserialize state dict back to AuroraMemory instance
- Handle version migration for forward compatibility

Design principles:
- Human-readable state inspection
- Cross-version compatibility
- Partial state recovery
- State diffing and debugging
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict

from aurora.algorithms.components.assignment import CRPAssigner, StoryModel, ThemeModel
from aurora.algorithms.components.bandit import ThompsonBernoulliGate
from aurora.algorithms.components.density import OnlineKDE
from aurora.algorithms.components.metric import LowRankMetric
from aurora.algorithms.constants import RECENT_ENCODED_PLOTS_WINDOW
from aurora.algorithms.graph.memory_graph import MemoryGraph
from aurora.algorithms.graph.vector_index import VectorIndex
from aurora.algorithms.models.config import MemoryConfig
from aurora.algorithms.models.plot import Plot
from aurora.algorithms.models.story import StoryArc
from aurora.algorithms.models.theme import Theme
from aurora.algorithms.retrieval.field_retriever import FieldRetriever


class SerializationMixin:
    """Mixin providing state serialization and deserialization functionality."""

    # -------------------------------------------------------------------------
    # State Serialization
    # -------------------------------------------------------------------------

    def to_state_dict(self) -> Dict[str, Any]:
        """
        Serialize entire AuroraMemory state to JSON-compatible dict.
        
        This replaces pickle-based serialization with structured JSON,
        enabling:
        - Human-readable state inspection
        - Cross-version compatibility
        - Partial state recovery
        - State diffing and debugging
        
        Returns:
            JSON-compatible dictionary representing the full state
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
            
            # Bookkeeping (convert deque to list for JSON)
            "recent_encoded_plot_ids": list(self._recent_encoded_plot_ids),
            
            # Relationship-centric additions
            "relationship_story_index": self._relationship_story_index,
            "identity_dimensions": self._identity_dimensions,
        }

    # -------------------------------------------------------------------------
    # State Deserialization
    # -------------------------------------------------------------------------

    @classmethod
    def from_state_dict(cls, d: Dict[str, Any]) -> "SerializationMixin":
        """Reconstruct AuroraMemory from state dict."""
        cfg = MemoryConfig.from_state_dict(d["cfg"])
        obj = cls(cfg=cfg, seed=d.get("seed", 0))
        
        obj._restore_learnable_components(d)
        obj._restore_memory_data(d)
        obj._rebuild_indices_and_models(d)
        obj._restore_bookkeeping(d)
        
        return obj

    def _restore_learnable_components(self, d: Dict[str, Any]) -> None:
        """Restore learnable components from state dict."""
        self.kde = OnlineKDE.from_state_dict(d["kde"])
        self.metric = LowRankMetric.from_state_dict(d["metric"])
        self.gate = ThompsonBernoulliGate.from_state_dict(d["gate"])
        self.crp_story = CRPAssigner.from_state_dict(d["crp_story"])
        self.crp_theme = CRPAssigner.from_state_dict(d["crp_theme"])

    def _restore_memory_data(self, d: Dict[str, Any]) -> None:
        """Restore plots, stories, and themes from state dict."""
        self.plots = {pid: Plot.from_state_dict(pd) for pid, pd in d.get("plots", {}).items()}
        self.stories = {sid: StoryArc.from_state_dict(sd) for sid, sd in d.get("stories", {}).items()}
        self.themes = {tid: Theme.from_state_dict(td) for tid, td in d.get("themes", {}).items()}

    def _rebuild_indices_and_models(self, d: Dict[str, Any]) -> None:
        """Rebuild graph, vector index, and models."""
        payloads: Dict[str, Any] = {}
        payloads.update(self.plots)
        payloads.update(self.stories)
        payloads.update(self.themes)
        
        self.graph = MemoryGraph.from_state_dict(d["graph"], payloads=payloads)
        self.vindex = VectorIndex.from_state_dict(d["vindex"])
        
        self.story_model = StoryModel(metric=self.metric)
        self.theme_model = ThemeModel(metric=self.metric)
        self.retriever = FieldRetriever(metric=self.metric, vindex=self.vindex, graph=self.graph)

    def _restore_bookkeeping(self, d: Dict[str, Any]) -> None:
        """Restore bookkeeping data structures."""
        self._recent_encoded_plot_ids = deque(
            d.get("recent_encoded_plot_ids", []),
            maxlen=RECENT_ENCODED_PLOTS_WINDOW
        )
        
        self._relationship_story_index = d.get("relationship_story_index", {})
        self._identity_dimensions = d.get("identity_dimensions", {})
        
        # Rebuild relationship index from stories if not present
        if not self._relationship_story_index:
            for sid, story in self.stories.items():
                if story.relationship_with:
                    self._relationship_story_index[story.relationship_with] = sid
