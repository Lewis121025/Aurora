"""Budget utilities for the Aurora v2 kernel."""

from __future__ import annotations

from aurora.core.types import BudgetConfig, PosteriorGroup, TraceEdge, TraceRecord
from aurora.replay.reconsolidate import trace_utility


class BudgetController:
    def __init__(self, config: BudgetConfig):
        self.config = config

    def pressure(self, *, trace_mass: float, edge_mass: float, group_mass: float) -> float:
        return max(
            trace_mass / max(self.config.max_traces, 1),
            edge_mass / max(self.config.max_edges, 1),
            group_mass / max(self.config.max_groups, 1),
        )

    def trace_score(self, trace: TraceRecord, *, now_ts: float) -> float:
        return trace_utility(trace, now_ts)

    def edge_score(self, edge: TraceEdge) -> float:
        return 0.60 * edge.support_ema + 0.30 * edge.weight + 0.10 * edge.bf_sep_ema

    def group_score(self, group: PosteriorGroup) -> float:
        return 0.55 * group.unresolved_mass + 0.25 * len(group.member_ids) + 0.20 * max(group.temperature, 0.0)
