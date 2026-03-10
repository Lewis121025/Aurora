#!/usr/bin/env python3

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from aurora.interfaces.terminal.observer import build_parser, run_observer

    args = build_parser().parse_args()
    run_observer(
        data_dir=args.data_dir,
        session_id=args.session_id,
        max_hits=args.max_hits,
    )


if __name__ == "__main__":
    main()
