"""本体定向模块。

定义 Aurora 的本体定向（Orientation），维护自我/世界/关系三个维度的
证据积累，用于支撑认知过程中的自我定位和关系判断。
"""
from __future__ import annotations

from dataclasses import dataclass, field  # noqa: F811

from typing import TYPE_CHECKING

from aurora.runtime.contracts import AuroraMove, TraceChannel

if TYPE_CHECKING:
    from aurora.memory.knot import Knot
    from aurora.memory.thread import Thread
    from aurora.relation.formation import RelationFormation


EvidenceMap = dict[str, tuple[str, ...]]
"""证据映射：维度名 -> 证据源 ID 元组。"""


def _default_self_evidence() -> EvidenceMap:
    """初始化自我维度证据。

    包含 recognition（被识别）、fragility（脆弱性）、
    openness（开放性）、agency（能动性）四个子维度。
    """
    return {"recognition": (), "fragility": (), "openness": (), "agency": ()}


def _default_world_evidence() -> EvidenceMap:
    """初始化世界维度证据。

    包含 welcome（接纳）、risk（风险）、mystery（神秘感）、
    stability（稳定性）四个子维度。
    """
    return {"welcome": (), "risk": (), "mystery": (), "stability": ()}


def _default_relation_evidence() -> EvidenceMap:
    """初始化关系维度证据。

    包含 closeness（亲密）、distance（距离）、
    boundary（边界）、repair（修复）四个子维度。
    """
    return {"closeness": (), "distance": (), "boundary": (), "repair": ()}


def _merge_sources(
    existing: tuple[str, ...], *source_groups: tuple[str, ...], limit: int = 16
) -> tuple[str, ...]:
    """合并证据源 ID 列表。

    将新证据源追加到现有列表末尾，去重并保持顺序。
    超出 limit 的部分会被截断（保留最新的 limit 个）。

    Args:
        existing: 现有证据源列表。
        source_groups: 待合并的证据源组。
        limit: 最大保留数量，默认 16。

    Returns:
        合并后的证据源元组。
    """
    merged = list(existing)
    for group in source_groups:
        for source in group:
            if not source:
                continue
            if source in merged:
                merged.remove(source)
            merged.append(source)
    return tuple(merged[-limit:])


def _append_sources(target: EvidenceMap, key: str, *source_groups: tuple[str, ...]) -> None:
    """向证据映射的指定维度追加证据源。

    Args:
        target: 目标证据映射。
        key: 维度键名。
        source_groups: 待追加的证据源组。
    """
    target[key] = _merge_sources(target[key], *source_groups)


def _snapshot_group(group: EvidenceMap) -> dict[str, dict[str, int | tuple[str, ...]]]:
    """生成证据组的快照。

    返回每个维度的证据数量和源 ID 列表。
    """
    return {key: {"count": len(sources), "sources": sources} for key, sources in group.items()}


@dataclass(slots=True)
class Orientation:
    """Aurora 本体定向。

    维护三个维度的证据积累：
    - self_evidence: 自我认知证据
    - world_evidence: 世界认知证据
    - relation_evidence: 关系认知证据

    同时记录锚定线程和活跃记忆结，用于在认知过程中保持连续性。
    """

    self_evidence: EvidenceMap = field(default_factory=_default_self_evidence)
    """自我维度证据。"""

    world_evidence: EvidenceMap = field(default_factory=_default_world_evidence)
    """世界维度证据。"""

    relation_evidence: EvidenceMap = field(default_factory=_default_relation_evidence)
    """关系维度证据。"""

    anchor_thread_ids: tuple[str, ...] = ()
    """锚定记忆线程 ID（最多 12 个）。"""

    active_knot_ids: tuple[str, ...] = ()
    """活跃记忆结 ID（最多 12 个）。"""

    last_updated_at: float = 0.0
    """最后更新时间戳。"""

    def register_exchange(
        self,
        user_channels: tuple[TraceChannel, ...],
        aurora_move: AuroraMove,
        relation_moment_id: str,
        user_fragment_id: str,
        aurora_fragment_id: str,
        now_ts: float,
    ) -> None:
        """注册一次交互交换的证据。

        根据用户触发的通道和 Aurora 的回应行为，向三个维度积累证据。

        Args:
            user_channels: 用户触发的轨迹通道列表。
            aurora_move: Aurora 的回应行为（approach/repair/withhold/silence/boundary）。
            relation_moment_id: 关系时刻 ID。
            user_fragment_id: 用户片段 ID。
            aurora_fragment_id: Aurora 片段 ID。
            now_ts: 当前时间戳。
        """
        channels = set(user_channels)
        moment_sources = (relation_moment_id,)
        exchange_sources = (relation_moment_id, user_fragment_id, aurora_fragment_id)

        # 根据用户通道积累证据
        if TraceChannel.RECOGNITION in channels:
            _append_sources(self.self_evidence, "recognition", exchange_sources)
            _append_sources(self.world_evidence, "welcome", exchange_sources)
            _append_sources(self.relation_evidence, "closeness", moment_sources)
        if TraceChannel.HURT in channels:
            _append_sources(self.self_evidence, "fragility", exchange_sources)
            _append_sources(self.world_evidence, "risk", exchange_sources)
        if TraceChannel.CURIOSITY in channels or TraceChannel.WONDER in channels:
            _append_sources(self.world_evidence, "mystery", exchange_sources)
        if TraceChannel.BOUNDARY in channels:
            _append_sources(self.relation_evidence, "boundary", moment_sources)
            _append_sources(self.world_evidence, "risk", exchange_sources)
        if TraceChannel.REPAIR in channels:
            _append_sources(self.relation_evidence, "repair", moment_sources)
            _append_sources(self.world_evidence, "stability", exchange_sources)
        if TraceChannel.DISTANCE in channels:
            _append_sources(self.relation_evidence, "distance", moment_sources)

        # 根据 Aurora 行为积累证据
        if aurora_move in {"approach", "repair"}:
            _append_sources(self.self_evidence, "openness", moment_sources)
            _append_sources(self.self_evidence, "agency", moment_sources)
            _append_sources(self.relation_evidence, "closeness", moment_sources)
        elif aurora_move in {"withhold", "silence"}:
            _append_sources(self.relation_evidence, "distance", moment_sources)
            _append_sources(self.self_evidence, "agency", moment_sources)
        elif aurora_move == "boundary":
            _append_sources(self.relation_evidence, "boundary", moment_sources)
            _append_sources(self.self_evidence, "agency", moment_sources)

        self.last_updated_at = now_ts

    def absorb_sleep(
        self,
        thread_ids: tuple[str, ...],
        knot_ids: tuple[str, ...],
        dominant_channels: tuple[TraceChannel, ...],
        now_ts: float,
    ) -> None:
        """吸收 sleep 阶段的整合结果。

        将 sleep 阶段产生的线程和记忆结纳入定向系统，
        并根据主导通道积累相应的证据。

        Args:
            thread_ids: 整合后的线程 ID 列表。
            knot_ids: 整合后的记忆结 ID 列表。
            dominant_channels: 主导轨迹通道列表。
            now_ts: 当前时间戳。
        """
        self.anchor_thread_ids = _merge_sources(self.anchor_thread_ids, thread_ids, limit=12)
        self.active_knot_ids = _merge_sources(self.active_knot_ids, knot_ids, limit=12)
        channels = set(dominant_channels)
        dominant_sources = knot_ids or thread_ids

        if TraceChannel.WARMTH in channels or TraceChannel.RECOGNITION in channels:
            _append_sources(self.world_evidence, "welcome", dominant_sources)
            _append_sources(self.world_evidence, "stability", dominant_sources)
        if TraceChannel.HURT in channels or TraceChannel.BOUNDARY in channels:
            _append_sources(self.world_evidence, "risk", dominant_sources)
            _append_sources(self.self_evidence, "fragility", dominant_sources)
        if TraceChannel.REPAIR in channels:
            _append_sources(self.relation_evidence, "repair", dominant_sources)
        if TraceChannel.BOUNDARY in channels:
            _append_sources(self.relation_evidence, "boundary", dominant_sources)
        self.last_updated_at = now_ts

    def absorb_topology(
        self,
        threads: tuple[Thread, ...],
        knots: tuple[Knot, ...],
        formations: tuple[RelationFormation, ...],
        now_ts: float,
    ) -> None:
        """从 thread/knot/formation 拓扑精细推导证据。

        高连贯性线程 -> stability + recognition
        高张力线程 -> fragility + risk
        未解决记忆结 -> fragility + risk
        已解决记忆结 -> agency + stability
        修复超过边界的 formation -> repair + closeness
        边界超过修复的 formation -> boundary + distance

        Args:
            threads: 线程列表。
            knots: 记忆结列表。
            formations: 关系形成记录列表。
            now_ts: 当前时间戳。
        """
        for thread in threads:
            sources = (thread.thread_id,)
            if thread.coherence >= 0.5:
                _append_sources(self.world_evidence, "stability", sources)
                _append_sources(self.self_evidence, "recognition", sources)
            if thread.tension >= 0.5:
                _append_sources(self.self_evidence, "fragility", sources)
                _append_sources(self.world_evidence, "risk", sources)
            if thread.coherence >= 0.3:
                _append_sources(self.self_evidence, "openness", sources)

        for knot in knots:
            sources = (knot.knot_id,)
            if knot.resolved:
                _append_sources(self.self_evidence, "agency", sources)
                _append_sources(self.world_evidence, "stability", sources)
            else:
                _append_sources(self.self_evidence, "fragility", sources)
                _append_sources(self.world_evidence, "risk", sources)

        for formation in formations:
            sources = (formation.relation_id,)
            if formation.repair_events > formation.boundary_events:
                _append_sources(self.relation_evidence, "repair", sources)
                _append_sources(self.relation_evidence, "closeness", sources)
            elif formation.boundary_events > formation.repair_events:
                _append_sources(self.relation_evidence, "boundary", sources)
                _append_sources(self.relation_evidence, "distance", sources)
            if formation.resonance_events >= 3:
                _append_sources(self.world_evidence, "welcome", sources)

        self.last_updated_at = now_ts

    def snapshot(
        self,
    ) -> dict[str, dict[str, dict[str, int | tuple[str, ...]]] | tuple[str, ...]]:
        """生成定向快照。

        返回三个维度的证据统计及锚定线程/记忆结列表。
        """
        return {
            "self": _snapshot_group(self.self_evidence),
            "world": _snapshot_group(self.world_evidence),
            "relation": _snapshot_group(self.relation_evidence),
            "anchor_threads": self.anchor_thread_ids,
            "active_knots": self.active_knot_ids,
        }
