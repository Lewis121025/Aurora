"""Aurora v3 CLI."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from aurora.runtime.engine import AuroraKernel


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(prog="aurora")
    subparsers = parser.add_subparsers(dest="command", required=True)

    turn_parser = subparsers.add_parser("turn", help="Run one relation-scoped turn")
    turn_parser.add_argument("text", help="Input text for Aurora")
    turn_parser.add_argument("--relation-id", default="default")

    snapshot_parser = subparsers.add_parser("snapshot", help="Show relation snapshot")
    snapshot_parser.add_argument("--relation-id", default="default")

    recall_parser = subparsers.add_parser("recall", help="Recall relation-scoped atoms and evidence")
    recall_parser.add_argument("query", help="Recall query")
    recall_parser.add_argument("--relation-id", default="default")
    recall_parser.add_argument("--limit", type=int, default=5)

    subparsers.add_parser("status", help="Show kernel status")

    args = parser.parse_args()
    kernel = AuroraKernel.create()

    try:
        if args.command == "turn":
            output = kernel.turn(relation_id=args.relation_id, text=args.text)
            print(output.response_text)
            return

        if args.command == "snapshot":
            print(json.dumps(asdict(kernel.snapshot(args.relation_id)), ensure_ascii=False, indent=2))
            return

        if args.command == "recall":
            print(
                json.dumps(
                    asdict(kernel.recall(args.relation_id, args.query, limit=args.limit)),
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
