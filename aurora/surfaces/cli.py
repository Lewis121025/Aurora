"""CLI for Aurora's canonical runtime surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from aurora.llm.config import load_llm_settings
from aurora.runtime.system import AuroraSystem, AuroraSystemConfig, build_llm_provider, to_dict
from aurora.surfaces.http import build_app


def _print(payload: Any) -> None:
    print(json.dumps(to_dict(payload), ensure_ascii=False, indent=2))


def _load_provider() -> Any:
    settings = load_llm_settings()
    if settings is None:
        return None
    return build_llm_provider(settings)


def _make_system(args: argparse.Namespace) -> AuroraSystem:
    root = Path(args.data_dir or ".aurora")
    config = AuroraSystemConfig(
        data_dir=str(root),
        db_path=args.db,
        blob_dir=str(root / "blobs"),
        autosave=not getattr(args, "no_autosave", False),
        max_snapshots=getattr(args, "max_snapshots", 256),
        encoder_dim=getattr(args, "encoder_dim", 128),
        packet_chars=getattr(args, "packet_chars", 512),
        ann_top_k=getattr(args, "ann_top_k", 64),
        hot_trace_limit=getattr(args, "hot_trace_limit", 32),
        settle_steps=getattr(args, "settle_steps", 4),
        workspace_k=getattr(args, "workspace_k", 12),
        maintenance_ms_budget=getattr(args, "maintenance_ms_budget", 20),
        trace_budget=getattr(args, "trace_budget", 256),
        edge_budget=getattr(args, "edge_budget", 2048),
        anchor_budget=getattr(args, "anchor_budget", 1024),
        local_decoder_backend=getattr(args, "local_decoder_backend", "transformers"),
        background_maintenance_interval=getattr(args, "background_maintenance_interval", 5.0),
        background_maintenance_budget=getattr(args, "background_maintenance_budget", 6),
    )
    return AuroraSystem(config, llm=_load_provider())


def _load_metadata(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise SystemExit("metadata must be a JSON object")
    return payload


def cmd_inject(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(
            system.inject(
                {
                    "payload": args.payload,
                    "session_id": args.session_id,
                    "turn_id": args.turn_id,
                    "source": args.source,
                    "payload_type": args.payload_type,
                    "ts": args.ts,
                    "metadata": _load_metadata(args.metadata),
                }
            )
        )
    finally:
        system.close()


def cmd_read_workspace(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(system.read_workspace({"payload": args.cue, "session_id": args.session_id}, k=args.k))
    finally:
        system.close()


def cmd_maintenance_cycle(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(system.maintenance_cycle(ms_budget=args.ms_budget))
    finally:
        system.close()


def cmd_respond(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(
            system.respond(
                {
                    "payload": args.cue,
                    "session_id": args.session_id,
                    "turn_id": args.turn_id,
                    "source": args.source,
                    "ts": args.ts,
                    "metadata": _load_metadata(args.metadata),
                }
            )
        )
    finally:
        system.close()


def cmd_snapshot(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(system.snapshot())
    finally:
        system.close()


def cmd_field_stats(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(system.field_stats())
    finally:
        system.close()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    system = _make_system(args)
    if args.background_maintenance:
        system.start_background_maintenance(
            interval=args.background_maintenance_interval,
            ms_budget=args.background_maintenance_budget,
        )
    app = build_app(system)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        system.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aurora", description="Aurora unified trace-field runtime")
    parser.add_argument("--data-dir", default=".aurora")
    parser.add_argument("--db", default=".aurora/aurora.sqlite", help="SQLite snapshot store path")
    parser.add_argument("--max-snapshots", type=int, default=256)
    parser.add_argument("--no-autosave", action="store_true")
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--packet-chars", type=int, default=512)
    parser.add_argument("--ann-top-k", type=int, default=64)
    parser.add_argument("--hot-trace-limit", type=int, default=32)
    parser.add_argument("--settle-steps", type=int, default=4)
    parser.add_argument("--anchor-budget", type=int, default=1024)
    parser.add_argument("--trace-budget", type=int, default=256)
    parser.add_argument("--edge-budget", type=int, default=2048)
    parser.add_argument("--workspace-k", type=int, default=12)
    parser.add_argument("--maintenance-ms-budget", type=int, default=20)
    parser.add_argument("--background-maintenance-interval", type=float, default=5.0)
    parser.add_argument("--background-maintenance-budget", type=int, default=6)
    parser.add_argument("--local-decoder-backend", default="transformers")
    sub = parser.add_subparsers(dest="command", required=True)

    inject = sub.add_parser("inject")
    inject.add_argument("--payload", required=True)
    inject.add_argument("--session-id", default="")
    inject.add_argument("--turn-id", default=None)
    inject.add_argument("--source", default="user")
    inject.add_argument("--payload-type", default="text")
    inject.add_argument("--ts", type=int, default=None)
    inject.add_argument("--metadata", default="{}", help="JSON object")
    inject.set_defaults(func=cmd_inject)

    workspace = sub.add_parser("read-workspace")
    workspace.add_argument("--cue", required=True)
    workspace.add_argument("--session-id", default="")
    workspace.add_argument("--k", type=int, default=None)
    workspace.set_defaults(func=cmd_read_workspace)

    maintenance = sub.add_parser("maintenance-cycle")
    maintenance.add_argument("--ms-budget", type=int, default=None)
    maintenance.set_defaults(func=cmd_maintenance_cycle)

    respond = sub.add_parser("respond")
    respond.add_argument("--cue", required=True)
    respond.add_argument("--session-id", default="")
    respond.add_argument("--turn-id", default=None)
    respond.add_argument("--source", default="user")
    respond.add_argument("--ts", type=int, default=None)
    respond.add_argument("--metadata", default="{}", help="JSON object")
    respond.set_defaults(func=cmd_respond)

    snapshot = sub.add_parser("snapshot")
    snapshot.set_defaults(func=cmd_snapshot)

    stats = sub.add_parser("field-stats")
    stats.set_defaults(func=cmd_field_stats)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--background-maintenance", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
