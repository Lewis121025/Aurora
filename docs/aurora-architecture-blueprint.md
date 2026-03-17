# Aurora v2 Architecture Blueprint

## Position

Aurora v2 不是 RAG 平台，也不是生命周期模拟器。它是一个把交互历史编译成当前关系状态的嵌入式 SDK。

系统本体只有三层：

1. `Evidence Log`
   记录 `user_turn`、`assistant_turn`、`compile_failure`。它是唯一真相源，append-only。
2. `Relation Field`
   默认挂载到每轮回复的长期主观状态。它表达过去如何改变了当前姿态。
3. `Archive`
   只在需要精确回忆时激活，负责事实与 transcript 的混合检索，不负责人格塑形。

## Runtime

热路径严格限制为：

1. 写入 user turn
2. 读取 `RelationField + Top OpenLoops`
3. 判断是否需要 archive recall
4. 生成回复
5. 写入 assistant turn

默认不做全局检索，不做状态编译，不做阶段流转。

## Compiler

后台 compiler 只输出类型化 `MemoryOp`：

- `assert_fact`
- `revise_fact`
- `patch_relation`
- `open_loop`
- `resolve_loop`
- `add_rule`
- `update_lexicon`

Reducer 是唯一允许修改长期状态的入口。失败时回滚事务，状态保持不变。

## State Model

### Relation Field

显式关系场包含：

- `trust`
- `distance`
- `warmth`
- `tension`
- `repair_debt`
- `shared_lexicon`
- `interaction_rules`
- `last_compiled_at`

### Open Loop

只保留四种未完成事项：

- `commitment`
- `contradiction`
- `unfinished_thread`
- `unresolved_question`

Loop 允许衰减紧迫度，但不允许无痕消失。

### Facts

事实以版本链存储：

- 新事实用 `assert_fact`
- 更正事实用 `revise_fact`
- 冲突不会静默覆盖，而是生成新的 active fact，并打开 `contradiction` loop

## Storage

默认后端是本地 SQLite。

核心表只有：

- `events`
- `relation_fields`
- `open_loops`
- `facts`
- `fact_embeddings`

不引入外部向量数据库，不保留旧状态机兼容层。

## Surface

核心库只提供 Python SDK。

外层适配只有两个：

- CLI
- FastAPI

它们都只是 `AuroraKernel` 的薄封装，不持有额外业务状态。

## Acceptance Criteria

Aurora v2 的验收标准不是召回率，而是以下行为是否稳定出现：

1. 多轮互动后关系姿态持续一致
2. 用户更正旧事实时不会静默覆盖
3. 承诺与未完事项会持续存在直到被解决
4. 只有在明确需要时才触发 archive recall
5. 编译失败不会损坏长期状态

## Rejected Paths

以下路径不再属于 Aurora：

- `awake / doze / sleep`
- `metabolic` 生命周期模拟
- 从图结构反推关系状态
- “默认先检索，再决定怎么说”的主路径
- 为旧架构保留兼容垫片
