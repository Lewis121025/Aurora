# Aurora v3 Architecture Blueprint

## Position

Aurora v3 是 relation-only 的陪伴内核。主路径只围绕单一 `relation_id` 组织连续性，不保留并行的 session 语义。

系统只有两类持久化真相：

1. `Evidence Log`
   append-only 的用户、助手与编译失败事件流。
2. `Memory Atoms`
   唯一长期语义原子，承载事实、规则、词汇、未完成事项、修订与遗忘。

`RelationField / OpenLoops / Facts` 都是运行时派生视图，不是并列真相层。

## Runtime

每轮 `turn` 固定执行：

1. 写入 user event
2. 运行 pre-response 规则，吸收显式高价值信号
3. 读取当前 atoms，派生 `RelationField + OpenLoops + RecallHits`
4. 生成回复
5. 写入 assistant event
6. 运行 post-response compiler，追加新的 atoms

编译失败不会回滚当轮 evidence，只会追加 `compile_failure` 事件。

## Compiler

Aurora v3 有两条收敛后的编译路径：

1. pre-response deterministic rules
   处理显式 forget、interaction rule 与高置信度 loop 信号。
2. post-response LLM compiler
   输出类型化 `MemoryOp`，再由 reducer 落成 atoms。

允许的 atom op 只有：

- `fact`
- `rule`
- `lexicon`
- `loop`
- `revision`
- `forget`

Reducer 是唯一允许修改长期语义的入口。

## Derived Views

### Relation Field

当前关系姿态只由 lifecycle-effective atoms 派生，不直接持久化为独立真相层。

### Open Loops

只保留四种 loop：

- `commitment`
- `contradiction`
- `unfinished_thread`
- `unresolved_question`

loop 的当前状态来自 loop atoms 的归并结果。

### Facts

事实视图来自 fact atoms；修订不会静默覆盖，而是通过新的 fact atom 与 revision atom 表达。

## Storage

默认后端是本地 SQLite。

核心表只有：

- `events`
- `memory_atoms`

不引入额外状态表，不保留旧关系场/事实表双轨模型。

## Surface

核心库提供 Python SDK。

外层适配只有：

- CLI
- FastAPI

它们都只是 `AuroraKernel` 的薄封装，不持有额外业务状态。

## Acceptance Criteria

Aurora v3 的验收标准是语义收敛是否稳定成立：

1. 同一 `relation_id` 下连续性由 atoms 稳定延续
2. 用户更正旧信息时不会静默覆盖
3. forget 影响可见性，而不是硬删 evidence
4. 编译失败不会污染已有 atoms

## Rejected Paths

以下路径不再属于 Aurora：

- session-scoped 双轨记忆
- 把 relation field、facts、loops 落成独立持久化真相层，而不是保持为 derived views
- 为旧架构保留兼容垫片
- 默认先做全局检索再决定如何回应
