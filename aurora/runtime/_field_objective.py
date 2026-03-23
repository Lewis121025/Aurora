"""Objective and replay-evaluation helpers for AuroraField."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from aurora.core.math import EPS
from aurora.core.types import ExperienceFrame, TraceRecord, Workspace
from aurora.readout.workspace import settle_workspace
from aurora.runtime.objective import ObjectiveObservation, ObjectiveTerms, empirical_patch_terms


class FieldObjectiveMixin:
    def _degree(self: Any, trace_id: str) -> int:
        return sum(1 for edge in self.edges.values() if edge.src == trace_id or edge.dst == trace_id)

    def _collect_local_frames(
        self: Any,
        trace_ids: Sequence[str],
        *,
        session_id: str = "",
        limit: int | None = None,
    ) -> list[ExperienceFrame]:
        wanted = {trace_id for trace_id in trace_ids if trace_id in self.traces}
        if not wanted:
            return []
        budget = int(limit or self.config.objective_local_window)
        frames: list[ExperienceFrame] = []
        for frame in reversed(self.frames):
            if session_id and frame.session_id and frame.session_id != session_id:
                continue
            if frame.trace_id in wanted or (frame.next_trace_id is not None and frame.next_trace_id in wanted):
                frames.append(frame)
            elif any(trace_id in wanted for trace_id in frame.active_ids):
                frames.append(frame)
            if len(frames) >= budget:
                break
        frames.reverse()
        return frames

    def _objective_observations(
        self: Any,
        x: np.ndarray,
        c: np.ndarray,
        pred: Any,
        *,
        trace_ids: Sequence[str],
        session_id: str = "",
        extra_frames: Sequence[ExperienceFrame] | None = None,
    ) -> list[ObjectiveObservation]:
        observations: list[ObjectiveObservation] = [
            ObjectiveObservation(
                x=np.asarray(x, dtype=np.float64),
                c=np.asarray(c, dtype=np.float64),
                target=np.asarray(pred.mu, dtype=np.float64),
                target_sigma_diag=np.asarray(pred.sigma_diag, dtype=np.float64),
            )
        ]
        seen_frame_ids: set[str] = set()
        frames = (
            list(extra_frames)
            if extra_frames is not None
            else self._collect_local_frames(trace_ids, session_id=session_id)
        )
        for frame in frames:
            if frame.frame_id in seen_frame_ids:
                continue
            seen_frame_ids.add(frame.frame_id)
            target = None if frame.next_x is None else np.asarray(frame.next_x, dtype=np.float64)
            observations.append(
                ObjectiveObservation(
                    x=np.asarray(frame.x, dtype=np.float64),
                    c=np.asarray(frame.context, dtype=np.float64),
                    target=target,
                )
            )
        return observations

    def _replay_workspace_for_frame(self: Any, frame: ExperienceFrame) -> Workspace:
        candidates = self._make_candidates(frame.x, frame.context, frame.session_id)
        size = min(max(len(frame.active_ids), 1), self.config.workspace_size)
        return settle_workspace(
            candidate_ids=candidates,
            traces=self.traces,
            edges=self.edges,
            groups=self.groups,
            cue=np.asarray(frame.x, dtype=np.float64),
            context=np.asarray(frame.context, dtype=np.float64),
            frontier=np.asarray(frame.frontier_vec, dtype=np.float64),
            pred_mu=np.asarray(frame.workspace_vec, dtype=np.float64),
            config=self.config,
            posterior_slice_fn=self._posterior_slice,
            session_id=frame.session_id,
            workspace_size=size,
        )

    def _transition_support(self: Any, src: str, dst: str) -> float:
        support = 0.0
        temporal = self.edges.get((src, dst, "temporal"))
        if temporal is not None:
            support = max(support, float(np.clip(temporal.weight, 0.0, 1.0)))
        option = self.edges.get((src, dst, "option"))
        if option is not None:
            total = max(option.success_alpha + option.success_beta, EPS)
            support = max(support, float(option.success_alpha / total))
        return float(np.clip(support, 0.0, 1.0))

    def _objective_terms(
        self: Any,
        candidate_ids: Sequence[str],
        x: np.ndarray,
        c: np.ndarray,
        pred: Any,
        *,
        shadow: Mapping[str, TraceRecord] | None = None,
        extra_trace_cost: float = 0.0,
        extra_edge_cost: float = 0.0,
        plasticity: float = 0.0,
        inhibit_bonus: float = 0.0,
        observations: Sequence[ObjectiveObservation] | None = None,
    ) -> tuple[ObjectiveTerms, dict[str, float]]:
        shadow = shadow or {}
        observations = list(
            observations
            or [
                ObjectiveObservation(
                    x=np.asarray(x, dtype=np.float64),
                    c=np.asarray(c, dtype=np.float64),
                    target=np.asarray(pred.mu, dtype=np.float64),
                    target_sigma_diag=np.asarray(pred.sigma_diag, dtype=np.float64),
                )
            ]
        )
        drift = float(plasticity)
        workspace_drift = 0.0
        if self.workspace.summary_vector.size > 0:
            workspace_drift = float(
                np.mean(np.square(self.workspace.summary_vector - np.asarray(pred.mu, dtype=np.float64)))
            )
            drift += workspace_drift
        terms, responsibilities = empirical_patch_terms(
            config=self.config,
            candidate_ids=candidate_ids,
            traces=self.traces,
            edges=self.edges,
            groups=self.groups,
            observations=observations,
            shadow=shadow,
            extra_trace_cost=extra_trace_cost,
            extra_edge_cost=extra_edge_cost,
            plasticity=plasticity,
            inhibit_bonus=inhibit_bonus,
            extra_drift=drift,
        )
        if workspace_drift > 0.0:
            terms.extras["workspace_drift"] = float(workspace_drift)
        return terms, responsibilities
