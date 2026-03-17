"""Aurora v2 CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from aurora.runtime.engine import AuroraKernel


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(prog="aurora")
    subparsers = parser.add_subparsers(dest="command", required=True)

    turn_parser = subparsers.add_parser("turn", help="Run one interaction turn")
    turn_parser.add_argument("text", help="Input text for Aurora")
    turn_parser.add_argument("--session-id", default="default")

    compile_parser = subparsers.add_parser("compile", help="Compile pending turns")
    compile_parser.add_argument("--session-id")

    snapshot_parser = subparsers.add_parser("snapshot", help="Show relation snapshot")
    snapshot_parser.add_argument("--session-id", default="default")

    recall_parser = subparsers.add_parser("recall", help="Recall archived facts and events")
    recall_parser.add_argument("query", help="Recall query")
    recall_parser.add_argument("--session-id", default="default")
    recall_parser.add_argument("--limit", type=int, default=5)

    subparsers.add_parser("status", help="Show kernel status")

    args = parser.parse_args()
    kernel = AuroraKernel.create()

    try:
        if args.command == "turn":
            output = kernel.turn(session_id=args.session_id, text=args.text)
            print(output.response_text)
            return

        if args.command == "compile":
            print(json.dumps(asdict(kernel.compile_pending(session_id=args.session_id)), ensure_ascii=False, indent=2))
            return

        if args.command == "snapshot":
            print(json.dumps(asdict(kernel.snapshot(args.session_id)), ensure_ascii=False, indent=2))
            return

        if args.command == "recall":
            print(
                json.dumps(
                    asdict(kernel.recall(args.session_id, args.query, limit=args.limit)),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        if args.command == "status":
            print(json.dumps({"status": "ok", "relations": kernel.store.relation_count()}, ensure_ascii=False, indent=2))
            return
    finally:
        kernel.close()
