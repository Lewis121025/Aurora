# Aurora Ontology Core

这不是对旧 `Store/Profile/Engine/Reweave/SleepCycle` 的兼容补丁，而是一套新的本体级骨架。

## 设计原则

1. **不要以 profile 作为人格来源**
   - 没有 `user_profile`、`style_profile`、`persona_profile`
   - 所谓“人格”只来自对象历史、关系历史、章节化叙事和 being latent 的长期漂移

2. **不要把 reweave 做成 context 拼接**
   - awake 的响应计划读取的是 `Fragment / RelationState / Chapter`
   - sleep 的主要输出是 `Chapter + association 重挂接 + being drift`
   - 不是把若干 memory snippets 拼成 prompt

3. **不要把关系做成 tone 汇总**
   - 关系由 `RelationMoment` 形成
   - 关系持续态由 `RelationState` 保存
   - 边界、修复、靠近、保留都是第一类关系动作

4. **生命周期是真实过程，不是 API 名字**
   - awake: 形成经历、激活记忆、产出回应并提交交换
   - doze: 衰减激活、巩固共激活、生成暂态结构
   - sleep: 选区、聚团、成章、回写、漂移

## 模块说明

- `domain.py`
  - 核心对象：Turn / Fragment / Trace / Association / Chapter / RelationMoment / RelationState / BeingState / ReweaveResult
- `repositories.py`
  - 内存图与关系仓库
- `heuristics.py`
  - 规则版 imprint / response，只用于最小可运行示例
- `reweave.py`
  - 叙事重编织器
- `kernel.py`
  - 生命周期协调器

## 核心流转

### awake

1. 创建用户 `InteractionTurn`
2. 用 `Imprinter` 生成用户 `Fragment + Trace`
3. 写入 memory graph
4. 按 relation + channel 召回相关 fragment
5. 用 `Responder` 生成响应计划
6. 生成 Aurora `InteractionTurn`
7. 把这次交换固化为 `RelationMoment`
8. 更新 `BeingState`

### doze

1. fragment activation 衰减
2. trace intensity 缓慢衰减
3. 强化 recent co-activation resonance
4. 提升 narrative pressure / sleep pressure

### sleep

1. 选择 narrative potential 高的候选 fragment
2. 按 relation / channel / tag / existing association 聚团
3. 生成 `Chapter`
4. 对 cluster 写入 chapter association
5. 让 chapter 影响 `RelationState`
6. 让 reweave 结果漂移 `BeingState`

## 和旧实现的根本区别

旧实现：
- `profile` 给参数
- `engine` 调用若干函数
- `store` 保存几段文本
- `sleep` 返回一个 delta

新骨架：
- 对象本体先于参数
- 生命周期先于字符串拼接
- 章节化叙事先于摘要
- 关系动作先于 tone 标签

## 建议的下一步

1. 把当前项目里的 `response.py` 改造成 `Responder` port
2. 把持久化从 flat store 升级为 event log + object tables
3. 用真实 embedding / semantic parser 替换 `RuleBasedImprinter`
4. 把 `InMemoryMemoryGraph` 替换为 Postgres + pgvector 或 graph-backed store
