# Aurora 代码示例（面向你当前结构）

下面这份示例不是“推倒重写”，而是按你当前 `engine.py / models.py / store.py / sleep.py` 的形状给的兼容补丁。

---

## 1. 先修 `handle_turn()`：不要在 safety 之前 commit

你现在的 `handle_turn()` 是先 `_run_awake()`，后 `non_malice_floor()`。  
这会让内部已提交的状态和用户真正看到的文本发生分叉。

可以改成下面这种两阶段提交：

```python
from dataclasses import replace
from uuid import uuid4

def handle_turn(self, relation_id: str, session_id: str, text: str) -> EngineOutput:
    now_ts = self.clock.now()
    user_turn = InteractionTurn(
        turn_id=f"turn_{uuid4().hex[:10]}",
        relation_id=relation_id,
        session_id=session_id,
        speaker="user",
        text=text,
        created_at=now_ts,
    )

    draft = run_awake(
        turn=user_turn,
        snapshot=self.state.snapshot,
        memory_store=self.memory_store,
        relation_store=self.relation_store,
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

    self._commit_awake(
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        outcome=final_outcome,
        aurora_move=aurora_move,
        now_ts=now_ts,
    )

    return EngineOutput(
        turn_id=user_turn.turn_id,
        response_text=safe_text,
        touch_modes=final_outcome.touch_modes,
    )
```

对应的 commit 方法：

```python
def _commit_awake(
    self,
    user_turn: InteractionTurn,
    aurora_turn: InteractionTurn,
    outcome: AwakeOutcome,
    aurora_move: str,
    now_ts: float,
) -> None:
    self.state.snapshot = outcome.snapshot
    if outcome.transition is not None:
        self.state.transitions.append(outcome.transition)

    relation_moment = self.relation_store.record_exchange(
        relation_id=user_turn.relation_id,
        session_id=user_turn.session_id,
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        touch_modes=outcome.touch_modes,
        user_move="share",
        aurora_move=aurora_move,
        created_at=now_ts,
    )

    # 如果你的 AwakeOutcome 还是旧结构，可以在这里 replace 一下
    committed = replace(outcome, relation_moment=relation_moment)

    # 这里需要你把 persistence 接口扩成能同时存 user_turn / aurora_turn
    self.persistence.persist_awake(
        user_turn=user_turn,
        aurora_turn=aurora_turn,
        outcome=committed,
        memory_store=self.memory_store,
    )
```

---

## 2. `RelationStore` 不要再只按全局 moments 聚合

你当前是：

- 一个全局 `moments: list[RelationMoment]`
- `current_tone()` 不带 `relation_id`
- `tone_strength()` 不带 `relation_id`

更合理的是：

```python
@dataclass(slots=True)
class RelationAggregate:
    state: RelationState
    moments: list[RelationMoment] = field(default_factory=list)


@dataclass(slots=True)
class RelationStore:
    relations: dict[str, RelationAggregate] = field(default_factory=dict)

    def current_tone(self, relation_id: str) -> Tone:
        aggregate = self.relations[relation_id]
        scores = self._tone_scores(aggregate.moments)
        return max(scores.items(), key=lambda item: item[1])[0]
```

这样多用户、多关系才不会污染。

完整可运行示例见 `relation_store_v2_example.py`。

---

## 3. sleep 不要再只让 `run_sleep_reweave()` 返回一个 delta

你现在是：

```python
delta = run_sleep_reweave(...)
world_view = ...
openness = ...
self_view = ...
```

更好的桥接版是让 reweave 返回 richer result：

```python
@dataclass(frozen=True, slots=True)
class ReweaveResult:
    delta: float
    chapter_ids: tuple[str, ...]
    coherence_shift: float
    tension_shift: float
    relation_bias: float
```

然后 `sleep.py` 用这些量来更新 snapshot：

```python
reweave = run_sleep_reweave_v2(...)

world_view = _clamp(
    snapshot.world_view
    + 0.18 * (world_target - snapshot.world_view)
    + 0.16 * reweave.coherence_shift
    - 0.12 * reweave.tension_shift
    + 0.10 * reweave.relation_bias,
    -1.0,
    1.0,
)
```

完整可运行示例见 `memory_reweave_v2_example.py` 和 `sleep_bridge_example.py`。

---

## 4. `models.py` 先做增量重构，不要一步改成完全抽象 latent

最小增量建议：

- `InteractionTurn` 增加 `relation_id`
- `Fragment` 增加 `salience / unresolvedness / relation_id / chapter_ids`
- `Trace` 升级成 `TraceResidue(channel=...)`
- `AssociationDelta` 升级成持久 `AssociationEdge(kind=...)`
- 新增 `Chapter`
- `RelationMoment` 增加 `user_move / aurora_move / aurora_turn_id`

完整示例见 `models_v2_example.py`。

---

## 5. 推荐你的实际迁移顺序

第一步，只改 `handle_turn()` 提交顺序。  
第二步，把 `relation_id` 打通到 `InteractionTurn / RelationStore / sleep()`。  
第三步，引入 `Chapter` 和 `AssociationEdge(kind=...)`。  
第四步，再把 recall / expression 改成读取 `Fragment + Chapter + TraceResidue`。
