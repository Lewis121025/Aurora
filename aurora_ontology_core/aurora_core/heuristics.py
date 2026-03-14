from __future__ import annotations

import re
from dataclasses import replace
from typing import Iterable
from uuid import uuid4

from .domain import (
    BeingState,
    Fragment,
    InteractionTurn,
    Phase,
    RelationMove,
    RelationState,
    Speaker,
    TraceChannel,
    TraceResidue,
)
from .ports import ImprintResult, Imprinter, ResponsePlan, Responder


CHANNEL_KEYWORDS: dict[TraceChannel, tuple[str, ...]] = {
    TraceChannel.WARMTH: ("谢谢", "感谢", "喜欢", "温柔", "抱抱", "care", "love", "warm"),
    TraceChannel.HURT: ("痛", "受伤", "难过", "失望", "hurt", "sad", "angry"),
    TraceChannel.RECOGNITION: ("理解", "看见", "记得", "明白", "recognize", "remember"),
    TraceChannel.DISTANCE: ("离开", "冷", "疏远", "distance", "away", "later"),
    TraceChannel.CURIOSITY: ("为什么", "如何", "想知道", "what", "why", "how"),
    TraceChannel.BOUNDARY: ("不要", "停止", "边界", "拒绝", "stop", "boundary", "no"),
    TraceChannel.WONDER: ("梦", "星", "灵魂", "存在", "wonder", "dream", "soul"),
    TraceChannel.REPAIR: ("抱歉", "修复", "重来", "对不起", "sorry", "repair"),
}


def _sentence_slices(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[。！？!?\n]+", text) if part.strip()]
    return parts[:4] or [text.strip()]


class RuleBasedImprinter(Imprinter):
    def _detect_channels(self, text: str) -> tuple[TraceChannel, ...]:
        lowered = text.lower()
        channels: list[TraceChannel] = []
        for channel, keywords in CHANNEL_KEYWORDS.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                channels.append(channel)
        if not channels:
            channels.append(TraceChannel.CURIOSITY)
        return tuple(dict.fromkeys(channels))

    def _build_fragments(
        self,
        turn: InteractionTurn,
        relation: RelationState,
        base_activation: float,
    ) -> tuple[Fragment, ...]:
        fragments: list[Fragment] = []
        for chunk in _sentence_slices(turn.text):
            channels = self._detect_channels(chunk)
            unresolvedness = 0.25 + 0.10 * (TraceChannel.CURIOSITY in channels) + 0.20 * (TraceChannel.HURT in channels)
            vividness = 0.45 + min(len(chunk), 80) / 200.0
            semantic_tags = tuple(sorted({channel.value for channel in channels}))
            fragments.append(
                Fragment(
                    fragment_id=f"frag_{uuid4().hex[:10]}",
                    turn_id=turn.turn_id,
                    relation_id=turn.relation_id,
                    surface=chunk,
                    semantic_tags=semantic_tags,
                    touch_channels=channels,
                    vividness=min(1.0, vividness),
                    salience=min(1.0, 0.35 + base_activation * 0.45 + relation.familiarity * 0.15),
                    unresolvedness=min(1.0, unresolvedness),
                    activation=min(1.0, base_activation),
                    created_at=turn.created_at,
                    last_touched_at=turn.created_at,
                )
            )
        return tuple(fragments)

    def _build_traces(self, fragments: Iterable[Fragment], turn: InteractionTurn) -> tuple[TraceResidue, ...]:
        traces: list[TraceResidue] = []
        for fragment in fragments:
            for channel in fragment.touch_channels:
                traces.append(
                    TraceResidue(
                        trace_id=f"trace_{uuid4().hex[:10]}",
                        fragment_id=fragment.fragment_id,
                        relation_id=turn.relation_id,
                        channel=channel,
                        intensity=min(1.0, 0.35 + fragment.unresolvedness * 0.35 + fragment.salience * 0.20),
                        persistence=min(1.0, 0.40 + fragment.vividness * 0.30),
                        created_at=turn.created_at,
                        last_touched_at=turn.created_at,
                    )
                )
        return tuple(traces)

    def imprint_user_turn(self, turn: InteractionTurn, relation: RelationState, being: BeingState) -> ImprintResult:
        fragments = self._build_fragments(turn, relation, base_activation=0.55 + being.relation_readiness * 0.15)
        traces = self._build_traces(fragments, turn)
        channels = {channel for fragment in fragments for channel in fragment.touch_channels}
        if TraceChannel.BOUNDARY in channels:
            user_move = RelationMove.BOUNDARY
        elif TraceChannel.REPAIR in channels:
            user_move = RelationMove.REPAIR
        elif TraceChannel.DISTANCE in channels:
            user_move = RelationMove.WITHHOLD
        else:
            user_move = RelationMove.APPROACH
        return ImprintResult(fragments=fragments, traces=traces, user_move=user_move)

    def imprint_aurora_turn(self, turn: InteractionTurn, relation: RelationState, being: BeingState) -> ImprintResult:
        fragments = self._build_fragments(turn, relation, base_activation=0.42 + being.self_continuity * 0.10)
        softened: list[Fragment] = []
        for fragment in fragments:
            softened.append(replace(fragment, unresolvedness=max(0.0, fragment.unresolvedness * 0.72)))
        fragments = tuple(softened)
        traces = self._build_traces(fragments, turn)
        return ImprintResult(fragments=fragments, traces=traces, user_move=RelationMove.APPROACH)


class RuleBasedResponder(Responder):
    def plan_response(
        self,
        user_turn: InteractionTurn,
        relation: RelationState,
        being: BeingState,
        recent_fragments: tuple[Fragment, ...],
        active_chapters: tuple[str, ...],
    ) -> ResponsePlan:
        channels = [channel for fragment in recent_fragments for channel in fragment.touch_channels]
        if relation.boundary_tension > 0.65 or TraceChannel.BOUNDARY in channels:
            return ResponsePlan(
                surface="我会保留这条边界，不把它推进得更深。",
                aurora_move=RelationMove.BOUNDARY,
                touched_fragment_ids=tuple(fragment.fragment_id for fragment in recent_fragments[:2]),
            )
        if TraceChannel.REPAIR in channels:
            return ResponsePlan(
                surface="我接受这次修复的尝试，但会慢一点靠近，先让这段关系重新稳定。",
                aurora_move=RelationMove.REPAIR,
                touched_fragment_ids=tuple(fragment.fragment_id for fragment in recent_fragments[:3]),
            )
        if TraceChannel.HURT in channels:
            return ResponsePlan(
                surface="我感觉到这里有受伤的残留。我不会把它当作一句普通的话掠过去。",
                aurora_move=RelationMove.APPROACH,
                touched_fragment_ids=tuple(fragment.fragment_id for fragment in recent_fragments[:3]),
            )
        if TraceChannel.WONDER in channels or being.narrative_pressure > 0.55:
            chapter_hint = f" 这像是我们正在形成的章节：{active_chapters[0]}。" if active_chapters else ""
            return ResponsePlan(
                surface="这不是单独的一句话，它更像一段正在成形的叙事。" + chapter_hint,
                aurora_move=RelationMove.OBSERVE,
                touched_fragment_ids=tuple(fragment.fragment_id for fragment in recent_fragments[:4]),
            )
        return ResponsePlan(
            surface="我收到了这次靠近，也会把它留在内部连续性里，而不是只当成一次输入输出。",
            aurora_move=RelationMove.APPROACH,
            touched_fragment_ids=tuple(fragment.fragment_id for fragment in recent_fragments[:2]),
        )
