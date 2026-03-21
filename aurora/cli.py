"""CLI for AuroraSystem."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from aurora.api import build_app
from aurora.llm.config import load_llm_settings
from aurora.system import (
    AuroraSystem,
    AuroraSystemConfig,
    build_llm_provider,
    event_ingest_to_dict,
    recall_result_to_dict,
    response_output_to_dict,
)


def _print(payload: Any) -> None:
    if not isinstance(payload, dict):
        payload = cast(Any, asdict(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _load_provider() -> Any:
    settings = load_llm_settings()
    if settings is None:
        return None
    return build_llm_provider(settings)


def _make_system(args: argparse.Namespace) -> AuroraSystem:
    config = AuroraSystemConfig(
        db_path=args.db,
        seed=getattr(args, "seed", 7),
        autosave=not getattr(args, "no_autosave", False),
        save_on_retrieve=not getattr(args, "no_save_on_retrieve", False),
        max_snapshots=getattr(args, "max_snapshots", 256),
        background_replay_interval=getattr(args, "background_replay_interval", 5.0),
        background_replay_budget=getattr(args, "background_replay_budget", 6),
        session_context_messages=getattr(args, "session_context_messages", 12),
    )
    return AuroraSystem(config, llm=_load_provider())


def cmd_ingest(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        metadata = json.loads(args.metadata) if args.metadata else {}
        _print(event_ingest_to_dict(system.ingest(args.text, metadata=metadata, source=args.source, now_ts=args.now_ts)))
    finally:
        system.close()


def cmd_ingest_batch(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        events = json.loads(Path(args.file).read_text(encoding="utf-8"))
        if not isinstance(events, list):
            raise SystemExit("batch file must be a JSON list")
        _print(system.ingest_batch(events, source=args.source))
    finally:
        system.close()


def cmd_retrieve(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(recall_result_to_dict(system.retrieve(args.cue, top_k=args.top_k, propagation_steps=args.propagation_steps)))
    finally:
        system.close()


def cmd_current_state(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(recall_result_to_dict(system.current_state(top_k=args.top_k)))
    finally:
        system.close()


def cmd_replay(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(system.replay(budget=args.budget))
    finally:
        system.close()


def cmd_respond(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        metadata = json.loads(args.metadata) if args.metadata else {}
        _print(
            response_output_to_dict(
                system.respond(
                    args.session_id,
                    args.text,
                    metadata=metadata,
                    source=args.source,
                    top_k=args.top_k,
                    propagation_steps=args.propagation_steps,
                    now_ts=args.now_ts,
                )
            )
        )
    finally:
        system.close()


def cmd_stats(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(system.stats())
    finally:
        system.close()


def cmd_operations(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print({"operations": system.operation_history(limit=args.limit)})
    finally:
        system.close()


def cmd_atom(args: argparse.Namespace) -> None:
    system = _make_system(args)
    try:
        _print(system.get_atom(args.atom_id))
    finally:
        system.close()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    system = _make_system(args)
    if args.background_replay:
        system.start_background_replay(
            interval=args.background_replay_interval,
            budget=args.background_replay_budget,
        )
    app = build_app(system)
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        system.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aurora", description="Aurora unified memory system")
    parser.add_argument("--db", default=".aurora/aurora.sqlite", help="SQLite snapshot store path")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-snapshots", type=int, default=256)
    parser.add_argument("--no-autosave", action="store_true")
    parser.add_argument("--no-save-on-retrieve", action="store_true")
    parser.add_argument("--background-replay-interval", type=float, default=5.0)
    parser.add_argument("--background-replay-budget", type=int, default=6)
    parser.add_argument("--session-context-messages", type=int, default=12)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest")
    ingest.add_argument("--text", required=True)
    ingest.add_argument("--metadata", default="{}", help="JSON object")
    ingest.add_argument("--source", default="dialogue")
    ingest.add_argument("--now-ts", type=float, default=None)
    ingest.set_defaults(func=cmd_ingest)

    ingest_batch = sub.add_parser("ingest-batch")
    ingest_batch.add_argument("--file", required=True, help="JSON file containing event list")
    ingest_batch.add_argument("--source", default="dialogue")
    ingest_batch.set_defaults(func=cmd_ingest_batch)

    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--cue", required=True)
    retrieve.add_argument("--top-k", type=int, default=8)
    retrieve.add_argument("--propagation-steps", type=int, default=3)
    retrieve.set_defaults(func=cmd_retrieve)

    current_state = sub.add_parser("current-state")
    current_state.add_argument("--top-k", type=int, default=10)
    current_state.set_defaults(func=cmd_current_state)

    replay = sub.add_parser("replay")
    replay.add_argument("--budget", type=int, default=8)
    replay.set_defaults(func=cmd_replay)

    respond = sub.add_parser("respond")
    respond.add_argument("--session-id", required=True)
    respond.add_argument("--text", required=True)
    respond.add_argument("--metadata", default="{}", help="JSON object")
    respond.add_argument("--source", default="dialogue")
    respond.add_argument("--top-k", type=int, default=8)
    respond.add_argument("--propagation-steps", type=int, default=3)
    respond.add_argument("--now-ts", type=float, default=None)
    respond.set_defaults(func=cmd_respond)

    stats = sub.add_parser("stats")
    stats.set_defaults(func=cmd_stats)

    operations = sub.add_parser("operations")
    operations.add_argument("--limit", type=int, default=50)
    operations.set_defaults(func=cmd_operations)

    atom = sub.add_parser("atom")
    atom.add_argument("atom_id")
    atom.set_defaults(func=cmd_atom)

    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--background-replay", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
