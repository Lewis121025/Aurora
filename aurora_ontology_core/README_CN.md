# Aurora 对象本体论核心骨架

这不是对现有 `Store/Profile/Engine/Reweave/SleepCycle` 的修补版，而是一套新的最小 core scaffold，设计目标是：

1. 抛弃“用户画像式 profile”作为系统中心。
2. 抛弃“上下文字符串拼接式 reweave”。
3. 以对象本体论为中心：`Fragment / TraceResidue / AssociationEdge / Chapter / RelationMoment / RelationState / BeingState / ReweaveResult`。
4. 把 `awake / doze / sleep` 当成真实生命周期，而不是几个工具函数。

## 文件结构

- `core_models.py`
  - 全部一等对象定义。
- `memory_core.py`
  - 内存图：fragment、trace、association、chapter。
  - 提供 recall、doze 衰减、sleep reweave。
- `relation_core.py`
  - 关系账本：按 `relation_id` 存储 `RelationState` 和 `RelationMoment`。
- `lifecycle_core.py`
  - `AuroraCore`：真正的 awake / doze / sleep 入口。
- `demo_run.py`
  - 一个最小可运行演示。

## 设计上的硬转向

### 1. 不再有 profile 作为人格来源

人格、边界、开放性不再来自外部 profile 配置，而来自：

- 关系历史
- 记忆图形态
- chapter 重织结果
- being latent drift

### 2. 不再做字符串拼接上下文

表达层的输入对象是 `ActivationBundle`：

- 当前关系状态
- 被激活的 fragments
- 相关 traces
- 相关 chapters
- 主导 channels
- 是否需要边界

至于你以后接 LLM、symbolic generator 或其他表达系统，都应该读取这个 bundle，而不是读取一串拼接文本。

### 3. sleep 的产物不是 delta，而是重织结果

`sleep()` 返回 `ReweaveResult`，包含：

- 新 chapter ids
- 被更新的 fragment ids
- 被加强的 edge ids
- coherence shift
- tension shift
- self/world drift
- narrative notes

真正改变系统的是 memory graph 被改写，而不是某个数值被线性推进。

## 迁移建议

### 先替换的不是 LLM，而是对象层

优先级建议：

1. 把旧 `models.py` 改造成 `core_models.py` 这种结构。
2. 把旧 `RelationStore` 改为按 `relation_id` 的 `RelationLedger`。
3. 把旧 `sleep.py` 改为 `memory.reweave(...) -> ReweaveResult`。
4. 把旧 `engine.py` 改为生命周期调度器，而不是核心语义发生器。

### 对旧系统保留什么

只保留两类东西：

- 工程壳：持久化、时钟、日志、调度。
- 表达接线：把 `ActivationBundle` 交给你真正的表达器。

其余 `profile` 中心语义应该全部移除。

## 这份骨架当前没有做的事

- 没接真实嵌入模型
- 没接图数据库或 Postgres
- 没接 LLM response planner
- 没接 event sourcing
- chapter 聚类还是轻量启发式

但它已经完成了最关键的转向：

**Aurora 的中心从“配置 + 字符串”切换到“对象 + 痕迹 + 关系 + 重织”。**
