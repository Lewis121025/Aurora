from __future__ import annotations

import argparse

from aurora.host_runtime.runtime import AuroraRuntime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aurora")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat = subparsers.add_parser("chat")
    chat.add_argument("user_text", nargs="?")
    chat.add_argument("--language", default="auto")

    subparsers.add_parser("health")
    subparsers.add_parser("integrity")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    runtime = AuroraRuntime()

    if args.command == "chat":
        text = args.user_text or input("user> ").strip()
        outcome = runtime.handle_input(text, language=args.language)
        print(outcome.output_text or "<silence>")
        print(f"event_id={outcome.event_id}")
        print(f"next_wake_at={outcome.next_wake_at}")
        return

    if args.command == "health":
        health = runtime.health()
        print(f"version={health.version}")
        print(f"substrate_alive={health.substrate_alive}")
        print(f"sealed_state_version={health.sealed_state_version}")
        print(f"anchor_count={health.anchor_count}")
        print(f"next_wake_at={health.next_wake_at}")
        print(f"provider_healthy={health.provider_healthy}")
        return

    if args.command == "integrity":
        report = runtime.integrity()
        print(f"version={report.version}")
        print(f"runtime_boundary={report.runtime_boundary}")
        print(f"substrate_transport={report.substrate_transport}")
        print(f"sealed_state_version={report.sealed_state_version}")
        print(f"config_fingerprint={report.config_fingerprint}")
        print(f"generated_at={report.generated_at}")
        return

    raise SystemExit(2)
