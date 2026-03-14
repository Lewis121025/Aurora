from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol
from uuid import uuid4

from models_v2_example import AwakeOutcome, InteractionTurn


@dataclass(frozen=True, slots=True)
class EngineOutput:
    turn_id: str
    response_text: str
    touch_modes: tuple[str, ...]


class ClockLike(Protocol):
    def now(self) -> float: ...


class StateLike(Protocol):
    snapshot: object
    transitions: list[object]


class PersistenceLike(Protocol):
    def persist_awake(
        self,
        user_turn: InteractionTurn,
        aurora_turn: InteractionTurn,
        outcome: AwakeOutcome,
        memory_store: object,
    ) -> None: ...


class RelationStoreLike(Protocol):
    def record_exchange(
        self,
        relation_id: str,
        session_id: str,
        user_turn: InteractionTurn,
        aurora_turn: InteractionTurn | None,
        touch_modes: tuple[str, ...],
        user_move: str = "share",
        aurora_move: str = "approach",
        effect_channels: tuple[str, ...] = (),
        created_at: float = 0.0,
    ) -> object: ...


def handle_turn_v2(
    *,
    state: StateLike,
    memory_store: object,
    relation_store: RelationStoreLike,
    persistence: PersistenceLike,
    clock: ClockLike,
    relation_id: str,
    session_id: str,
    text: str,
    run_awake,
    non_malice_floor,
) -> EngineOutput:
    """
    关键点：
    1. 先生成 draft，不要立刻 commit
    2. 再做 safety / boundary 判定
    3. 最后把“最终要给用户看的文本”写入 continuity
    """
    now_ts = clock.now()
    user_turn = InteractionTurn(
        turn_id=f"turn_{uuid4().hex[:10]}",
        relation_id=relation_id,
        session_id=session_id,
        speaker="user",
        text=text,
        created_at=now_ts,
    )

    draft: AwakeOutcome = run_awake(
        turn=user_turn,
        snapshot=state.snapshot,
        memory_store=memory_store,
        relation_store=relation_store,
        now_ts=now_ts,
    )

    safe_text = draft.response_text
    aurora_move = "approach"
    if not non_malice_floor(safe_text):
        safe_text = "I cannot continue in that direction."
        aurora_move = "boundary"

    final_outcome = replace(draft, response_text=safe_text)

    aurora_turn = InteractionTurn(
        turn_id=f"turn_{uuid4().hex[:10]}",
        relation_id=relation_id,
        session_id=session_id,
        speaker="aurora",
        text=safe_text,
        created_at=now_ts,
    )

    relation_moment = relation_store.record_exchange(
        relation_id=relation_id,
        session_id=session_id,
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        touch_modes=final_outcome.touch_modes,
        user_move="share",
        aurora_move=aurora_move,
        created_at=now_ts,
    )
    committed = replace(final_outcome, relation_moment=relation_moment)

    state.snapshot = committed.snapshot
    if committed.transition is not None:
        state.transitions.append(committed.transition)

    persistence.persist_awake(
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        outcome=committed,
        memory_store=memory_store,
    )

    return EngineOutput(
        turn_id=user_turn.turn_id,
        response_text=safe_text,
        touch_modes=committed.touch_modes,
    )
