from __future__ import annotations

import argparse

from aurora.runtime.engine import AuroraEngine


def main() -> None:
    parser = argparse.ArgumentParser(prog="aurora-next")
    subparsers = parser.add_subparsers(dest="command", required=True)

    turn_parser = subparsers.add_parser("turn", help="Run one interaction turn")
    turn_parser.add_argument("text", help="Input text for Aurora")
    turn_parser.add_argument("--session-id", default="default")

    subparsers.add_parser("doze", help="Enter doze phase")
    subparsers.add_parser("sleep", help="Enter sleep phase")

    args = parser.parse_args()

    engine = AuroraEngine.create()
    if args.command == "turn":
        turn_output = engine.handle_turn(session_id=args.session_id, text=args.text)
        print(turn_output.response_text)
        return
    if args.command == "doze":
        phase_output = engine.doze()
        print(f"phase={phase_output.phase} transition={phase_output.transition_id}")
        return

    phase_output = engine.sleep()
    print(f"phase={phase_output.phase} transition={phase_output.transition_id}")
