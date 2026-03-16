"""命令行接口模块。

提供 Aurora 的 CLI 入口：
- aurora turn "text": 执行一次认知循环
- aurora doze: 进入 doze 状态
- aurora sleep: 进入 sleep 状态
"""
from __future__ import annotations

import argparse

from aurora.runtime.engine import AuroraEngine


def main() -> None:
    """CLI 入口函数。

    解析命令行参数，调用相应的引擎方法。
    """
    parser = argparse.ArgumentParser(prog="aurora-next")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # turn 子命令
    turn_parser = subparsers.add_parser("turn", help="Run one interaction turn")
    turn_parser.add_argument("text", help="Input text for Aurora")
    turn_parser.add_argument("--session-id", default="default")

    # doze 子命令
    subparsers.add_parser("doze", help="Enter doze phase")

    # sleep 子命令
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
