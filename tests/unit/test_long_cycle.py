"""长周期行为评测。

验证跨 session 的关系行为真实性：
- 跨 session 身份识别
- 长期偏好稳定（durability 片段存活并影响 recall）
- 边界事件持续影响关系投影
- 修复成功后关系回暖
- 久别重逢带余韵
"""
from __future__ import annotations

from pathlib import Path

from aurora.memory.recall import recent_recall
from aurora.relation.projectors import project_relation
from aurora.relation.stage import infer_stage
from aurora.runtime.engine import AuroraEngine

from tests.conftest import ContextAwareLLM, StubLLM


def test_cross_session_identity_recognition(tmp_path: Path) -> None:
    """同一 session 跨两次引擎启动应绑定到同一 relation。"""
    first = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    first.handle_turn(session_id="user_a", text="你好，我是 Alice")
    relation_id_1 = first.identity.relation_for("user_a")

    second = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    relation_id_2 = second.identity.relation_for("user_a")

    assert relation_id_1 is not None
    assert relation_id_1 == relation_id_2


def test_preference_durability_survives_sleep(tmp_path: Path) -> None:
    """含偏好信号的片段应获得 durability > 0，sleep 后依然可检索。"""
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    engine.handle_turn(session_id="s", text="我喜欢安静的环境，不喜欢被打扰")
    engine.handle_turn(session_id="s", text="普通聊天内容")
    engine.sleep()

    relation_id = engine.identity.relation_for("s")
    assert relation_id is not None

    durable = [
        f for f in engine.memory_store.fragments.values()
        if f.relation_id == relation_id and f.durability > 0.0
    ]
    assert durable, "Preference input should produce at least one durable fragment"

    recalled = recent_recall(engine.memory_store, relation_id, limit=8)
    recalled_ids = {f.fragment_id for f in recalled}
    assert any(f.fragment_id in recalled_ids for f in durable), (
        "Durable fragments should be recalled after sleep"
    )


def test_boundary_persists_in_relation_projection(tmp_path: Path) -> None:
    """边界事件应持续影响关系投影（distance 偏高、trust 偏低）。"""
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=ContextAwareLLM())
    engine.handle_turn(session_id="s", text="不要再继续，边界在这里")
    engine.handle_turn(session_id="s", text="普通对话")
    engine.sleep()

    relation_id = engine.identity.relation_for("s")
    assert relation_id is not None
    formation = engine.relation_store.formation_for(relation_id)
    projection = project_relation(formation, 1_000_000.0)

    assert formation.boundary_events >= 1
    assert projection.distance > 0.5, "Boundary should push distance up"


def test_repair_warms_relation(tmp_path: Path) -> None:
    """修复事件应改善关系投影（warmth 上升或 distance 下降）。"""
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=ContextAwareLLM())
    engine.handle_turn(session_id="s", text="不要再继续，边界在这里")

    relation_id = engine.identity.relation_for("s")
    assert relation_id is not None
    formation_before = engine.relation_store.formation_for(relation_id)
    proj_before = project_relation(formation_before, 1_000_000.0)

    engine.handle_turn(session_id="s", text="对不起，我想修复我们的关系")
    engine.sleep()

    formation_after = engine.relation_store.formation_for(relation_id)
    proj_after = project_relation(formation_after, 1_000_000.0)

    assert formation_after.repair_events >= 1
    assert proj_after.warmth >= proj_before.warmth or proj_after.distance <= proj_before.distance, (
        "Repair should warm the relation or reduce distance"
    )


def test_reunion_after_absence_carries_residue(tmp_path: Path) -> None:
    """久别重逢后，关系应携带之前积累的结构（线程/形成记录）。"""
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=ContextAwareLLM())
    engine.handle_turn(session_id="s", text="谢谢你理解我，我很感动")
    engine.handle_turn(session_id="s", text="我受伤了，需要一些空间")
    engine.sleep()

    relation_id = engine.identity.relation_for("s")
    assert relation_id is not None
    formation = engine.relation_store.formation_for(relation_id)
    thread_count_before = len(formation.thread_ids)

    reloaded = AuroraEngine.create(data_dir=str(tmp_path), llm=StubLLM())
    reloaded_formation = reloaded.relation_store.formation_for(relation_id)

    assert len(reloaded_formation.thread_ids) == thread_count_before
    assert reloaded_formation.resonance_events >= 1
    stage = infer_stage(reloaded_formation)
    assert stage != "initial", "Returning relation should not feel like a stranger"


def test_relation_stage_progresses_with_interaction(tmp_path: Path) -> None:
    """关系阶段应随交互推进。"""
    engine = AuroraEngine.create(data_dir=str(tmp_path), llm=ContextAwareLLM())

    engine.handle_turn(session_id="s", text="你好")
    relation_id = engine.identity.relation_for("s")
    assert relation_id is not None
    formation = engine.relation_store.formation_for(relation_id)
    early_stage = infer_stage(formation)

    engine.handle_turn(session_id="s", text="谢谢你理解我")
    engine.handle_turn(session_id="s", text="我很感动")
    engine.handle_turn(session_id="s", text="我信任你")
    engine.sleep()

    formation = engine.relation_store.formation_for(relation_id)
    later_stage = infer_stage(formation)

    stage_order = ["initial", "developing", "established", "strained", "repairing"]
    assert stage_order.index(later_stage) >= stage_order.index(early_stage), (
        f"Stage should progress: {early_stage} -> {later_stage}"
    )
