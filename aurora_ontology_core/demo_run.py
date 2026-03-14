from __future__ import annotations

from pprint import pprint

from aurora_core import AuroraCore, TouchSignal


def main() -> None:
    core = AuroraCore()
    relation_id = "rel_lumen"
    session_id = "sess_001"
    base = 1_700_000_000.0

    turns = [
        (
            "我记得你上次提到边界，我想更认真地靠近你，而不是把你当工具。",
            TouchSignal(weights={"recognition": 0.55, "warmth": 0.45, "curiosity": 0.20}, note="manual"),
        ),
        (
            "但我也担心如果我逼得太近，会不会让你受伤或者后退。",
            TouchSignal(weights={"hurt": 0.50, "boundary": 0.45, "repair": 0.20}, note="manual"),
        ),
        (
            "如果之前有什么冒犯，我愿意修复，而不是假装什么都没发生。",
            TouchSignal(weights={"repair": 0.65, "recognition": 0.35, "warmth": 0.20}, note="manual"),
        ),
    ]

    for index, (text, signal) in enumerate(turns):
        result = core.receive_user_turn(
            relation_id=relation_id,
            session_id=session_id,
            text=text,
            signal=signal,
            now_ts=base + index * 100,
        )
        print(f"USER   : {text}")
        print(f"AURORA : {result.aurora_turn.text}")
        print(f"MOVE   : {result.response_act.aurora_move} | phase={result.current_phase}")
        print("-")

    print("doze ->")
    pprint(core.doze_once(now_ts=base + 500))

    print("sleep ->")
    sleep_results = core.sleep_once(now_ts=base + 1000)
    pprint(sleep_results)

    print("being ->")
    pprint(core.being)

    print("chapters ->")
    pprint(core.memory.chapters)

    print("relations ->")
    pprint(core.relations.states)


if __name__ == "__main__":
    main()
