"""Aurora memory-field CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from aurora.runtime.engine import AuroraKernel


def _require_non_empty(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise argparse.ArgumentTypeError("must not be empty")
    return normalized


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(prog="aurora")
    subparsers = parser.add_subparsers(dest="command", required=True)

    turn_parser = subparsers.add_parser("turn", help="Run one subject-scoped turn")
    turn_parser.add_argument("text", type=_require_non_empty, help="Input text for Aurora")
    turn_parser.add_argument("--subject-id", required=True, type=_require_non_empty)
    turn_parser.add_argument("--session-id", required=True, type=_require_non_empty)

    finalize_parser = subparsers.add_parser("finalize-session", help="Finalize one session and distill memory")
    finalize_parser.add_argument("--subject-id", required=True, type=_require_non_empty)
    finalize_parser.add_argument("--session-id", required=True, type=_require_non_empty)
    finalize_parser.add_argument("--ended-at", type=float, default=None)

    state_parser = subparsers.add_parser("state", help="Show the current memory field")
    state_parser.add_argument("--subject-id", required=True, type=_require_non_empty)

    recall_parser = subparsers.add_parser("recall", help="Recall a local memory-field slice")
    recall_parser.add_argument("query", type=_require_non_empty, help="Recall query")
    recall_parser.add_argument("--subject-id", required=True, type=_require_non_empty)
    recall_parser.add_argument("--limit", type=int, default=8)

    subparsers.add_parser("status", help="Show kernel status")

    args = parser.parse_args()
    kernel = AuroraKernel.create()

    try:
        if args.command == "turn":
            turn_output = kernel.turn(subject_id=args.subject_id, session_id=args.session_id, text=args.text)
            print(turn_output.response_text)
            return

        if args.command == "finalize-session":
            ingest_output = kernel.finalize_session(args.subject_id, args.session_id, ended_at=args.ended_at)
            print(json.dumps(asdict(ingest_output), ensure_ascii=False, indent=2))
            return

        if args.command == "state":
            print(json.dumps(asdict(kernel.state(args.subject_id)), ensure_ascii=False, indent=2))
            return

        if args.command == "recall":
            print(
                json.dumps(
                    asdict(
                        kernel.recall(
                            args.subject_id,
                            args.query,
                            limit=args.limit,
                        )
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        if args.command == "status":
            print(json.dumps({"status": "ok", "subjects": kernel.store.subject_count()}, ensure_ascii=False, indent=2))
            return
    finally:
        kernel.close()
