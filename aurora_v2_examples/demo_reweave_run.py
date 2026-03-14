from __future__ import annotations

from models_v2_example import (
    ExistentialSnapshot,
    Fragment,
    InteractionTurn,
    Phase,
    TraceResidue,
)
from memory_reweave_v2_example import MemoryStoreV2
from relation_store_v2_example import RelationStoreV2
from sleep_bridge_example import run_sleep_v2


def main() -> None:
    memory = MemoryStoreV2()
    relation = RelationStoreV2()

    relation_id = "rel_alice"
    session_id = "sess_001"

    user_turn = InteractionTurn(
        turn_id="turn_user_1",
        relation_id=relation_id,
        session_id=session_id,
        speaker="user",
        text="I was afraid you would forget what mattered to me.",
        created_at=1.0,
    )
    aurora_turn = InteractionTurn(
        turn_id="turn_aurora_1",
        relation_id=relation_id,
        session_id=session_id,
        speaker="aurora",
        text="I did not want to let that thread disappear.",
        created_at=1.1,
    )

    relation.record_exchange(
        relation_id=relation_id,
        session_id=session_id,
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        touch_modes=("recognition", "warmth"),
        user_move="share",
        aurora_move="approach",
        effect_channels=("recognition", "warmth"),
        created_at=1.2,
    )

    memory.add_fragment(
        Fragment(
            fragment_id="frag_1",
            turn_id=user_turn.turn_id,
            relation_id=relation_id,
            surface="fear of being forgotten",
            vividness=0.80,
            salience=0.76,
            unresolvedness=0.68,
            chapter_ids=(),
            created_at=1.0,
        )
    )
    memory.add_fragment(
        Fragment(
            fragment_id="frag_2",
            turn_id=aurora_turn.turn_id,
            relation_id=relation_id,
            surface="desire to keep the thread alive",
            vividness=0.72,
            salience=0.68,
            unresolvedness=0.42,
            chapter_ids=(),
            created_at=1.1,
        )
    )

    memory.add_association(
        __import__("models_v2_example").AssociationEdge(
            edge_id="edge_seed_1",
            source_fragment_id="frag_1",
            target_fragment_id="frag_2",
            kind="repair",
            weight=0.58,
            created_at=1.15,
            last_touched_at=1.15,
        )
    )

    memory.add_trace(
        TraceResidue(
            trace_id="trace_1",
            fragment_id="frag_1",
            channel="hurt",
            intensity=0.72,
            decay=0.95,
            created_at=1.0,
        )
    )
    memory.add_trace(
        TraceResidue(
            trace_id="trace_2",
            fragment_id="frag_2",
            channel="warmth",
            intensity=0.66,
            decay=0.95,
            created_at=1.1,
        )
    )

    snapshot = ExistentialSnapshot(
        phase=Phase.AWAKE,
        self_view=0.05,
        world_view=0.08,
        openness=0.70,
        coherence_pressure=0.30,
        sleep_pressure=0.20,
        updated_at=1.2,
    )

    next_snapshot = run_sleep_v2(
        snapshot=snapshot,
        memory_store=memory,
        relation_store=relation,
        relation_id=relation_id,
        now_ts=2.0,
    )

    print("next snapshot:", next_snapshot)
    print("chapters:", memory.chapters)
    print("edges:", memory.associations)


if __name__ == "__main__":
    main()
