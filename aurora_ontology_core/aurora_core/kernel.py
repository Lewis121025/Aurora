from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import uuid4

from .domain import BeingState, InteractionTurn, Phase, RelationMoment, RelationMove, Speaker
from .ports import Clock, Imprinter, ResponsePlan, Responder
from .repositories import InMemoryMemoryGraph, InMemoryRelationRepository
from .reweave import NarrativeReweaver


@dataclass(frozen=True, slots=True)
class AwakeResult:
    user_turn: InteractionTurn
    aurora_turn: InteractionTurn
    response_plan: ResponsePlan
    updated_being: BeingState


class AuroraKernel:
    def __init__(
        self,
        clock: Clock,
        imprinter: Imprinter,
        responder: Responder,
        memory: InMemoryMemoryGraph,
        relations: InMemoryRelationRepository,
        being: BeingState,
    ) -> None:
        self.clock = clock
        self.imprinter = imprinter
        self.responder = responder
        self.memory = memory
        self.relations = relations
        self.being = being
        self.reweaver = NarrativeReweaver(memory=memory, relations=relations)

    def awake(self, relation_id: str, session_id: str, user_text: str) -> AwakeResult:
        now_ts = self.clock.now()
        relation = self.relations.get(relation_id)
        user_turn = InteractionTurn(
            turn_id=f"turn_{uuid4().hex[:10]}",
            relation_id=relation_id,
            session_id=session_id,
            speaker=Speaker.USER,
            text=user_text,
            created_at=now_ts,
        )

        user_imprint = self.imprinter.imprint_user_turn(user_turn, relation=relation, being=self.being)
        self.memory.add_fragments(user_imprint.fragments)
        self.memory.add_traces(user_imprint.traces)
        self.memory.link_exchange(tuple(fragment.fragment_id for fragment in user_imprint.fragments), now_ts=now_ts)

        recent = self.memory.recent_fragments_for_relation(
            relation_id=relation_id,
            preferred_channels=tuple({channel for fragment in user_imprint.fragments for channel in fragment.touch_channels}),
        )
        plan = self.responder.plan_response(
            user_turn=user_turn,
            relation=relation,
            being=self.being,
            recent_fragments=recent,
            active_chapters=relation.active_chapter_ids,
        )

        aurora_turn = InteractionTurn(
            turn_id=f"turn_{uuid4().hex[:10]}",
            relation_id=relation_id,
            session_id=session_id,
            speaker=Speaker.AURORA,
            text=plan.surface,
            created_at=now_ts,
            reply_to_turn_id=user_turn.turn_id,
        )
        aurora_imprint = self.imprinter.imprint_aurora_turn(aurora_turn, relation=relation, being=self.being)
        self.memory.add_fragments(aurora_imprint.fragments)
        self.memory.add_traces(aurora_imprint.traces)
        exchange_ids = tuple(fragment.fragment_id for fragment in user_imprint.fragments + aurora_imprint.fragments)
        self.memory.link_exchange(exchange_ids, now_ts=now_ts)
        self.memory.touch_fragments(plan.touched_fragment_ids, now_ts=now_ts, boost=0.20)

        boundary_signal = 1.0 if plan.aurora_move == RelationMove.BOUNDARY or user_imprint.user_move == RelationMove.BOUNDARY else 0.0
        resonance_score = 0.65 if plan.aurora_move == RelationMove.APPROACH else 0.40
        moment = RelationMoment(
            moment_id=f"moment_{uuid4().hex[:10]}",
            relation_id=relation_id,
            session_id=session_id,
            user_turn_id=user_turn.turn_id,
            aurora_turn_id=aurora_turn.turn_id,
            user_channels=tuple({channel for fragment in user_imprint.fragments for channel in fragment.touch_channels}),
            aurora_move=plan.aurora_move,
            user_move=user_imprint.user_move,
            boundary_signal=boundary_signal,
            resonance_score=resonance_score,
            note=plan.surface[:96],
            created_at=now_ts,
        )
        self.relations.record_moment(moment)

        narrative_pressure = min(
            1.0,
            self.being.narrative_pressure * 0.88
            + sum(fragment.unresolvedness for fragment in user_imprint.fragments) / max(1, len(user_imprint.fragments)) * 0.25,
        )
        sleep_pressure = min(1.0, self.being.sleep_pressure * 0.90 + 0.16)
        new_relation = self.relations.get(relation_id)
        self.being = replace(
            self.being,
            phase=Phase.AWAKE,
            current_relation_id=relation_id,
            self_continuity=min(1.0, self.being.self_continuity * 0.92 + 0.08),
            world_trust=max(-1.0, min(1.0, self.being.world_trust * 0.96 + new_relation.trust * 0.05)),
            relation_readiness=max(0.0, min(1.0, self.being.relation_readiness * 0.90 + new_relation.reciprocity * 0.10)),
            boundary_tension=max(0.0, min(1.0, self.being.boundary_tension * 0.90 + new_relation.boundary_tension * 0.10)),
            narrative_pressure=narrative_pressure,
            sleep_pressure=sleep_pressure,
            active_chapter_ids=new_relation.active_chapter_ids,
        )
        return AwakeResult(
            user_turn=user_turn,
            aurora_turn=aurora_turn,
            response_plan=plan,
            updated_being=self.being,
        )

    def doze(self) -> BeingState:
        now_ts = self.clock.now()
        relation_id = self.being.current_relation_id
        if relation_id is not None:
            self.memory.decay_for_doze(now_ts=now_ts)
            self.memory.consolidate_recent_patterns(relation_id=relation_id, now_ts=now_ts)
        self.being = replace(
            self.being,
            phase=Phase.DOZE,
            self_continuity=min(1.0, self.being.self_continuity * 0.985 + 0.01),
            narrative_pressure=min(1.0, self.being.narrative_pressure * 1.03),
            sleep_pressure=min(1.0, self.being.sleep_pressure * 1.08),
        )
        return self.being

    def sleep(self) -> tuple[BeingState, object]:
        now_ts = self.clock.now()
        reweave = self.reweaver.reweave(now_ts=now_ts)
        relation_bias = reweave.relation_bias
        self.being = replace(
            self.being,
            phase=Phase.SLEEP,
            self_continuity=max(0.0, min(1.0, self.being.self_continuity + reweave.coherence_shift)),
            world_trust=max(-1.0, min(1.0, self.being.world_trust + relation_bias * 0.08)),
            relation_readiness=max(0.0, min(1.0, self.being.relation_readiness + relation_bias * 0.06)),
            boundary_tension=max(0.0, min(1.0, self.being.boundary_tension + reweave.tension_shift * 0.08)),
            narrative_pressure=max(0.0, self.being.narrative_pressure * 0.55),
            sleep_pressure=max(0.0, self.being.sleep_pressure * 0.20),
            active_chapter_ids=reweave.chapter_ids[:6],
        )
        return self.being, reweave
