"""命令行接口模块。

提供 Aurora 的 CLI 入口：
- aurora turn "text": 执行一次认知循环
- aurora status: 查看当前状态
"""

from __future__ import annotations

import argparse

from aurora.runtime.engine import AuroraEngine


def main() -> None:
    """CLI 入口函数。"""
    parser = argparse.ArgumentParser(prog="aurora")
    subparsers = parser.add_subparsers(dest="command", required=True)

    turn_parser = subparsers.add_parser("turn", help="Run one interaction turn")
    turn_parser.add_argument("text", help="Input text for Aurora")
    turn_parser.add_argument("--session-id", default="default")

    subparsers.add_parser("status", help="Show engine status")

    args = parser.parse_args()

    engine = AuroraEngine.create()

    if args.command == "turn":
        output = engine.handle_turn(session_id=args.session_id, text=args.text)
        print(output.response_text)
        return

    if args.command == "status":
        print("Aurora is running")
        print(f"Active relations: {len(engine.relational_states)}")
        return
