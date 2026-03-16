"""结构驱动认知上下文构建模块。

从记忆图拓扑构建结构化认知输入，使单次认知能感知：
- 当前关系所处的长期阶段
- 当前输入是否触碰旧 thread 或 knot
- 建议的战略姿态（延续/修复/退后/重新理解）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from aurora.memory.tags import extract_tags
from aurora.relation.stage import RelationStage, infer_stage
from aurora.runtime.contracts import TraceChannel

if TYPE_CHECKING:
    from aurora.memory.store import MemoryStore
    from aurora.relation.formation import RelationFormation

Posture = Literal["continue", "repair", "retreat", "re_understand"]


@dataclass(frozen=True, slots=True)
class StructuralContext:
    """结构化认知上下文。

    Attributes:
        relation_stage: 关系阶段。
        touched_thread_ids: 当前输入触碰的线程 ID。
        touched_knot_ids: 当前输入触碰的 knot ID。
        posture: 建议战略姿态。
        hint: 供 LLM 消费的结构提示文本。
    """

    relation_stage: RelationStage
    touched_thread_ids: tuple[str, ...]
    touched_knot_ids: tuple[str, ...]
    posture: Posture
    hint: str


def build_structural_context(
    user_text: str,
    relation_id: str,
    formation: RelationFormation | None,
    memory_store: MemoryStore,
) -> StructuralContext:
    """从用户输入和记忆图构建结构化认知上下文。

    Args:
        user_text: 用户输入文本。
        relation_id: 关系 ID。
        formation: 关系形成记录（可能不存在）。
        memory_store: 记忆存储。

    Returns:
        StructuralContext: 结构化上下文。
    """
    from aurora.relation.formation import RelationFormation

    if formation is None:
        formation = RelationFormation(relation_id=relation_id)

    stage = infer_stage(formation)
    touched_threads, touched_knots = _detect_touches(user_text, relation_id, memory_store)
    posture = _infer_posture(stage, touched_knots, formation)
    hint = _format_hint(stage, touched_threads, touched_knots, posture)

    return StructuralContext(
        relation_stage=stage,
        touched_thread_ids=touched_threads,
        touched_knot_ids=touched_knots,
        posture=posture,
        hint=hint,
    )


def _detect_touches(
    user_text: str,
    relation_id: str,
    memory_store: MemoryStore,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """检测用户输入触碰的线程和 knot。

    通过标签和关键词重叠判定。
    """
    input_tags = set(extract_tags(user_text))
    if not input_tags:
        return (), ()

    touched_threads: list[str] = []
    for tid in memory_store.relation_threads.get(relation_id, []):
        thread = memory_store.threads.get(tid)
        if thread is None:
            continue
        thread_tags: set[str] = set()
        for fid in thread.fragment_ids:
            frag = memory_store.fragments.get(fid)
            if frag is not None:
                thread_tags.update(frag.tags)
        if input_tags & thread_tags:
            touched_threads.append(tid)

    touched_knots: list[str] = []
    for kid in memory_store.relation_knots.get(relation_id, []):
        knot = memory_store.knots.get(kid)
        if knot is None:
            continue
        knot_tags: set[str] = set()
        for fid in knot.fragment_ids:
            frag = memory_store.fragments.get(fid)
            if frag is not None:
                knot_tags.update(frag.tags)
        if input_tags & knot_tags:
            touched_knots.append(kid)

    return tuple(touched_threads[:4]), tuple(touched_knots[:4])


def _infer_posture(
    stage: RelationStage,
    touched_knots: tuple[str, ...],
    formation: RelationFormation,
) -> Posture:
    """从关系阶段和触碰状态推断战略姿态。"""
    if stage == "strained":
        return "retreat"
    if stage == "repairing" or touched_knots:
        return "repair"
    if stage == "initial":
        return "re_understand"
    return "continue"


_STAGE_LABELS: dict[RelationStage, str] = {
    "initial": "初识阶段",
    "developing": "关系发展中",
    "established": "关系已建立",
    "strained": "关系紧张",
    "repairing": "关系修复中",
}

_POSTURE_LABELS: dict[Posture, str] = {
    "continue": "延续当前关系节奏",
    "repair": "优先修复与对方的裂痕",
    "retreat": "保持距离，不要主动推进",
    "re_understand": "重新理解对方，不要假设已有的了解",
}


def _format_hint(
    stage: RelationStage,
    touched_threads: tuple[str, ...],
    touched_knots: tuple[str, ...],
    posture: Posture,
) -> str:
    """格式化供 LLM 的结构提示。"""
    parts: list[str] = [
        f"[关系阶段] {_STAGE_LABELS[stage]}",
        f"[建议姿态] {_POSTURE_LABELS[posture]}",
    ]
    if touched_threads:
        parts.append(f"[触碰旧线程] {len(touched_threads)}条")
    if touched_knots:
        parts.append(f"[触碰旧记忆结] {len(touched_knots)}个（注意敏感区域）")
    return " | ".join(parts)
