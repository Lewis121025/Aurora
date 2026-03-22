from __future__ import annotations

import numpy as np
from pathlib import Path

from aurora.core.types import FieldConfig, TraceEdge, TraceRecord
from aurora.readout import WorkspaceSerializer
from aurora.runtime import AuroraField

from tests.conftest import FieldFactory


def _inject_sample_field(field: AuroraField) -> None:
    field.inject(
        {
            "session_id": "session-a",
            "turn_id": "turn-1",
            "source": "user",
            "payload": "I live in Hangzhou.",
            "meta": {"role": "user"},
        }
    )
    field.inject(
        {
            "session_id": "session-a",
            "turn_id": "turn-2",
            "source": "assistant",
            "payload": "I will remind you tomorrow.",
            "meta": {"role": "assistant"},
        }
    )
    field.maintenance_cycle(ms_budget=12)


def test_workspace_serializer_renders_structured_sections(field_factory: FieldFactory) -> None:
    field = field_factory()
    _inject_sample_field(field)
    workspace = field.read_workspace("What do I remember?", k=6)

    serializer = WorkspaceSerializer()
    rendered = serializer.render_text(workspace, cue="What do I remember?", prompt="Preamble")

    assert isinstance(rendered, str)
    assert rendered.startswith("Preamble")
    assert "ongoing_context:" in rendered
    assert "active_hypotheses:" in rendered
    assert "relevant_procedures:" in rendered
    assert "provenance_anchors:" in rendered


def test_workspace_projection_stays_expression_side_only(field_factory: FieldFactory) -> None:
    field = field_factory()
    _inject_sample_field(field)
    workspace = field.read_workspace("Where do I live?", k=6)
    prompt = WorkspaceSerializer().render_text(workspace, cue="Where do I live?")

    assert prompt.startswith("cue: Where do I live?")
    assert not hasattr(field, "respond")
    assert not hasattr(field, "local_decoder")


def test_workspace_settle_prefers_session_frontier_over_cross_session_hot_trace(tmp_path: Path) -> None:
    traces = {
        "trace-a-frontier": TraceRecord(
            trace_id="trace-a-frontier",
            z_mu=np.asarray((0.8, 0.2), dtype=np.float64),
            z_sigma_diag=np.asarray((0.2, 0.2), dtype=np.float64),
            ctx_mu=np.asarray((0.8, 0.2), dtype=np.float64),
            ctx_sigma_diag=np.asarray((0.2, 0.2), dtype=np.float64),
            evidence=0.35,
            stability=0.4,
            access_ema=0.25,
            last_access_ts=2.0,
            t_start=2.0,
            t_end=2.0,
            metadata={"session_id": "session-a"},
        ),
        "trace-b-hot": TraceRecord(
            trace_id="trace-b-hot",
            z_mu=np.asarray((0.8, 0.2), dtype=np.float64),
            z_sigma_diag=np.asarray((0.2, 0.2), dtype=np.float64),
            ctx_mu=np.asarray((0.8, 0.2), dtype=np.float64),
            ctx_sigma_diag=np.asarray((0.2, 0.2), dtype=np.float64),
            evidence=0.8,
            stability=0.6,
            access_ema=0.95,
            last_access_ts=3.0,
            t_start=3.0,
            t_end=3.0,
            metadata={"session_id": "session-b"},
        ),
    }
    edges = {
        ("trace-b-hot", "trace-a-frontier", "inhib"): TraceEdge(
            src="trace-b-hot",
            dst="trace-a-frontier",
            kind="inhib",
            weight=0.7,
            support_ema=1.0,
            last_update_ts=4.0,
        )
    }
    config = FieldConfig(
        data_dir=str(tmp_path / "aurora-field"),
        db_path=str(tmp_path / "aurora-field" / "aurora.sqlite"),
        blob_dir=str(tmp_path / "aurora-field" / "blobs"),
        latent_dim=2,
        context_dim=2,
        workspace_size=1,
        settle_steps=3,
    )
    field = AuroraField(config)
    field.current_ts = 4.0
    field.session_frontiers["session-a"] = {"trace-a-frontier": 1.0}
    for trace in traces.values():
        field.trace_store.add(trace)
        field.ann_index.add_or_update(trace)
    for edge in edges.values():
        field.edge_store.upsert(edge)
    cue = np.asarray((0.8, 0.2), dtype=np.float64)
    workspace = field._settle_workspace(cue, cue, np.zeros(2, dtype=np.float64), "session-a", workspace_size=1)

    assert workspace.active_trace_ids == ("trace-a-frontier",)
    assert workspace.metadata["session_id"] == "session-a"
