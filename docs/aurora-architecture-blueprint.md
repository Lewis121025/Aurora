# Aurora vNext Architecture Blueprint

## Position

Aurora vNext 是一个单一主体的人类记忆内核。主路径只围绕单一 `subject_id` 组织连续性，并把所有持久化内容统一成 `MemoryAtom` ledger。`memory_atoms` 是唯一内部持久化真相；公开 surface 只暴露从当前可见记忆投影出的 state views，不暴露 raw atoms 或 provenance。

其中 `atom_kind` 固定为：

- `evidence`
- `episode`
- `semantic`
- `procedural`
- `cognitive`
- `affective`
- `narrative`
- `inhibition`

## Runtime

每轮 `turn` 固定执行：

1. 写入 user evidence atom
2. 先把 user message 结构化编译进当前记忆
3. 读取更新后的 subject state
4. 执行 `temporal_scope="current"` 的 blended recall
5. 生成回复
6. 写入 assistant evidence atom
7. 构建一个新的 episode atom
8. 编译 assistant commitments / narrative updates
9. 运行 reconsolidation / inhibition pass

如果 distillation 失败，不回滚 evidence atom；只追加 `compile_failure` evidence atom。

## State Projection

`state(subject_id)` 返回 `SubjectMemoryState` 的公开投影，只暴露主体当前可见记忆：

- `semantic_self_model`
- `semantic_world_model`
- `procedural_memory`
- `active_cognition`
- `affective_state`
- `narrative_state`
- `recent_episodes`

其中 assistant 明确承诺的未来动作会被蒸馏成 `procedural` atom，`trigger="assistant_commitment"`，并保留在当前状态里，直到被新的语义覆盖或显式抑制。

其中 `cognitive`、`affective`，以及当前 `trigger="plan"` 的 `procedural` atoms 都代表当前状态快照，不做无限累积；新的同类 atom 会 supersede 旧快照。

## Recall

Aurora vNext 的 `recall(subject_id, query, limit=5, mode="blended", temporal_scope=...)` 要求显式指定 `temporal_scope`：

- `current` 只召回当前有效记忆
- `historical` 只召回历史记忆
- `both` 同时召回当前与历史记忆

`blended recall` 只是 atom selection 的模式，不是独立记忆层：

- `episode` atoms 提供情景
- `semantic` / `procedural` / `cognitive` / `affective` / `narrative` atoms 提供长期语义

被 `inhibited` 的 atoms 不会进入 recall。`superseded` atoms 只在 `temporal_scope="historical"` 或 `temporal_scope="both"` 时重新可见。

## Lifecycle

### Reconsolidation

新的 semantic atom 如果与旧 atom 在同一语义槽位冲突：

- 旧 atom 变为 `superseded`
- 新 atom 通过 `supersedes_atom_id` 指向旧 atom

Aurora 不会静默覆盖旧事实。

### Inhibition

显式 forget 不会删除 evidence。

Aurora 会：

- 创建一个 `inhibition` atom
- 通过 `inhibits_atom_ids` 指向目标 atoms
- 把目标 atoms 变为 `inhibited`

所以 evidence 仍在，但后续 state / recall 不再把它们当作当前有效记忆。

## Storage

默认后端是本地 SQLite。

核心表只有：

- `metadata`
- `memory_atoms`

Aurora vNext 使用新的 `aurora_vnext.db` 和 `memory_atoms` schema，不保留兼容分支。

## Surface

核心库提供 Python SDK。

外层适配只有：

- CLI
- FastAPI
- MCP stdio server

它们都只是 `AuroraKernel` 的薄封装，不持有额外业务状态。

## Acceptance Criteria

Aurora vNext 的验收标准是：

1. 一轮交互会沉淀成 `evidence -> episode -> semantic/procedural/cognitive/affective/narrative/inhibition atoms`
2. semantic reconsolidation 不会静默覆盖旧记忆
3. inhibition 会隐藏记忆，但不会删除 evidence
4. cognitive 只保留结构化摘要，不存 raw thought
5. affective state 与 episode emotion markers 分层存在
6. narrative arcs 能跨 episode 累积
7. relation-first views 不再出现在主路径

## Rejected Paths

以下路径不再属于 Aurora：

- relation-first continuity kernel
- 三层持久化真相作为对外主模型
- `snapshot()` / `field / facts / open_loops`
- 暴露任意 trace mutation 接口给外部 surface
