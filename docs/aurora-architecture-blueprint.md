# Aurora vNext Architecture Blueprint

## Position

Aurora vNext 是一个单一主体的人类记忆内核。主路径围绕单一 `subject_id` 组织连续性，但它不再把“当前状态”落成固定字段，也不再通过外部规则去改写旧记忆。

Aurora vNext 的内部真相是：

- 不可变 `memory_atoms`
- 不可变 `memory_edges`
- 派生 `activation_cache`

其中 `atom_kind` 固定为：

- `evidence`
- `memory`
- `episode`
- `inhibition`

## Memory Field Axioms

Aurora 的 memory field kernel 只承诺以下公理：

1. 真相对象只有不可变 `memory_atoms`、不可变 `memory_edges` 和派生 `activation_cache`。
2. 节点的内禀激活只由本地 retention 和 query seed 决定。
3. 正边只传播超过内禀基线的剩余激活，因此静止场不会自我放大。
4. 负边按源节点当前可达性传播抑制压力，不承担逻辑否定或真值裁决语义。
5. `state()` 和 `recall()` 只暴露当前 memory field 的投影，不制造唯一真值层。
6. `recall()` 以缓存场为初值但保持只读；激活演化始终有界并收敛到稳定固定点。

## Compiler Boundary

`distillation` 只负责提出 memory-field proposal，不负责真值裁决。固定边界如下：

1. user compiler 只允许 `memory | inhibition`；completed-turn compiler 只允许 `memory | episode | inhibition`。
2. compiler 若返回非 JSON 或非 object，kernel 只追加 `compile_failure` evidence，不修补输出。
3. 非法 `kind`、空文本、非法数值范围、未知引用、自环边都会被丢弃。
4. 任何触及 `evidence` 的边都会被过滤；`evidence` 不参与场耦合。

## Storage Invariants

SQLite store 是真相守门层，而不只是序列化层。固定约束如下：

1. `atom_kind` 只能落在最小本体集合内；`confidence/salience/activation/influence` 都有数据库级范围约束。
2. `memory_edges` 与 `activation_cache` 只能引用同一 `subject_id` 下真实存在的 atom。
3. edge 不允许自环，也不允许 source/target 指向 `evidence` atom。
4. `evidence` 只记录观察痕迹，不允许带非空 `source_atom_ids`。
5. `replace_activation_cache()` 必须是原子替换；任何失败都回滚整批更新。
6. 持久化 payload 的恢复必须严格校验；非法 kind、坏 JSON、错误形状和缺失必填字段都直接失败，不允许静默降级成合法 memory。

## Runtime

每轮 `turn` 固定执行：

1. 写入 user evidence atom
2. 把 user message 编译成新的 memory field nodes 和 signed weighted edges
3. 对当前 subject 的 memory field 执行写时演化，刷新 activation cache
4. 从当前 memory field 和 query 切片构造 `MemoryBrief`，按 `current_mainline / query_relevant / recent_changes / active_tensions / ongoing_commitments` 组织回复上下文
5. 生成回复
6. 写入 assistant evidence atom
7. 把完整 user/assistant turn 编译成新的 memory field nodes 和 edges
8. 再执行一次写时演化

如果 distillation 失败，不回滚 evidence atom；只追加 `compile_failure` evidence atom。

## State

`state(subject_id)` 返回的不是旧 projection schema，而是当前 memory field 视图：

- `summary`
- `atoms`
- `edges`

当前态不是“谁被标成 active”，也不是“系统承诺为真的命题集合”，而是**当前 memory field 中高激活节点和局部关系的投影**。

## Recall

Aurora vNext 的 `recall(subject_id, query, limit=...)` 返回的是查询驱动的局部 memory field 切片：

- query 不再走 `current/historical/both` 这种离散状态过滤
- query 会在当前 memory field 上形成局部扰动，但不会回写真相对象
- recall 结果包含 `summary`、`atoms`、`edges`

## Lifecycle

Aurora 不再做以下旧式生命周期操作：

- 不再直接把旧 atom 改成 `superseded`
- 不再直接把旧 atom 改成 `inhibited`
- 不再按 kind 硬编码“单活快照”

新的冲突、淡化、修正、忘记和延续，都通过新节点进入场后，借由带符号、带强度的边关系自然改变当前激活分布。

## Storage

默认后端是本地 SQLite。

核心表固定为：

- `metadata`
- `memory_atoms`
- `memory_edges`
- `activation_cache`

Aurora vNext 使用新的 field schema，不保留兼容分支。

## Surface

核心库提供 Python SDK。

外层适配只有：

- CLI
- FastAPI
- MCP stdio server

它们都只是 `AuroraKernel` 的薄封装，不持有额外业务状态。

## Acceptance Criteria

Aurora vNext 的验收标准是：

1. 一轮交互会沉淀成新的 atom 和 edge，而不是改写旧 atom
2. 冲突写入会自然改变当前 memory field，但旧节点仍保留
3. forget 不删除 evidence，只改变当前激活分布
4. `state()` 和 `recall()` 都只读 field，不反向制造新状态
5. 回复上下文来自 memory field 摘要，而不是固定 projection fields
6. 连续多轮后不会出现无界扩散或随机记忆漂移

## Rejected Paths

以下路径不再属于 Aurora：

- status-driven lifecycle kernel
- slot / trigger / scope 驱动的硬编码语义系统
- 固定 `SubjectMemoryState` projection world
- 对外暴露任意 trace mutation 接口
