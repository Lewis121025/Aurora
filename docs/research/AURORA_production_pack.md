# AURORA 叙事记忆：生产级拼图补全包（可插拔 / 可观测 / 可回放）

> 你之前要的“从第一性原理出发、不要被原文档束缚”的算法核心，我已经用 `AURORA_memory_algorithm.md` 给出了可运行实现（Thompson gate + CRP + attractor retrieval + Beta edge + learnable metric）。  
> 这份文档与代码仓库则专注于你说的“**补齐生产级拼图**”：把算法从“能跑”推进到“能上线、能回放、能演进、能排障”。  
>
> 与你上传的两份文档的关系：  
> - 继承“Plot→Story→Theme→Self”的叙事层级与引擎分工（Weaver/Narrator/Evolver/Guardian）的目标，但把**阈值/固定权重/硬编码规则**替换为可学习或可配置的机制；fileciteturn0file1  
> - 继承“自由能/场论/信息几何”的动机，但在生产层面补上**幂等、持久化、回放、观测、隐私、安全、测试**这类不可回避的工程现实；fileciteturn0file0

---

## 0. 你到底缺的“生产拼图”是什么？

算法实现再漂亮，上线时通常会在下面这些地方崩掉：

1. **幂等性**：同一个交互被重复上报（重试、网络抖动、异步回调）会不会写两次？
2. **可回放**：重启/迁移/回归测试时，能不能从日志重建同一份记忆状态？
3. **可观测**：encode rate、检索成功率、延迟、存储膨胀、LLM token 成本，能不能量化？
4. **多租户隔离**：不同 user 的记忆互不污染；热用户不会拖垮冷用户。
5. **冷热分层**：Plot 明细不能无限增长；压缩/吸收要可控、可审计。
6. **版本化与迁移**：schema/prompt/embedding 模型升级时，旧数据怎么兼容？
7. **隐私与合规**：PII 如何最小化写入、如何脱敏、如何删（Right to be forgotten）？
8. **测试与回归**：指标退化你如何发现？怎么做 A/B 与消融？
9. **生产运行时**：API、调度、后台演化任务、容灾、备份。

下面给你的生产包就是把这些拼图一块补齐。

---

## 1. 交付物一：可下载的生产骨架代码（可直接改造成你的服务）

我给你生成了一个**可运行的参考工程**（默认：in-memory + sqlite event log + MockLLM，可在你机器上跑通；生产时替换 provider 即可）：

- **Zip 工程**：`aurora_prod/`（Python 包 + CLI + FastAPI API 可选 + sqlite event sourcing + snapshot + tests）  
- 关键入口：
  - CLI demo：`python -m aurora.interfaces.cli demo`
  - API（可选）：`uvicorn aurora.interfaces.api.app:app --port 8000`

> 你可以把这个工程当成“生产外壳”，把核心算法换成你自己的实现或我给你的 AURORA core。

---

## 2. 生产系统的第一性原理：必须把“不确定性”当成一等公民

你两份文档都强调“不要硬编码阈值/权重”，但生产系统里还有一类“硬编码”更致命：**把不确定性当成确定值**。

生产版 AURORA 的原则是：

- **数据层不确定**：Plot/Claim/Theme/Self 都带 `confidence` / `evidence_strength` / `Beta(a,b)` 后验；
- **决策层不确定**：encode、merge、theme-emerge、contradiction 都尽可能用概率决策；
- **工程层确定**：幂等、回放、版本、观测必须是确定的。

> 算法层追求“涌现”，工程层追求“可控”。两者不冲突。

---

## 3. 生产架构（与 Weaver/Narrator/Evolver/Guardian 的对齐）

你在架构文档里把叙事系统分成四个引擎（Weaver/Narrator/Evolver/Guardian）。fileciteturn0file1  
生产化的关键是：**把每个引擎做成“有边界、可替换、可观测”的模块**。

### 3.1 生产版组件边界

- **Ingress / Weaver（热路径）**
  - 幂等写 event log
  - PII redaction（可插拔）
  - PlotExtraction（LLM 可选，失败可降级）
  - 调用核心算法 `AuroraMemory.ingest(..., event_id=...)`
  - 只写“轻量 derived doc”：plot_extraction / ingest_result

- **Narrator（查询路径）**
  - 调用核心算法 `query` 得到 ranked ids + attractor_path
  - 从 doc store 拉摘要/证据链
  - 可选：LLM 做“重述”（reconstruction）而非“读取”

- **Evolver（后台）**
  - 周期性 `mem.evolve()`
  - 主题命名/自我叙事更新（LLM 可选）
  - 压缩/吸收/归档策略（在预算压力下）

- **Guardian（后台或查询时）**
  - Claim contradiction 发现与调和
  - Theme 冲突的条件化（“X成立，除非Y”）
  - Self consistency（能力叙事 vs 实际成功率）

### 3.2 我在代码里怎么落地？

- `aurora/runtime/runtime.py`：**AuroraRuntime**
  - `ingest_interaction()`：幂等 + WAL + 应用核心算法 + 写 doc
  - `query()`：检索并组装返回
  - `feedback()`：把任务成败反馈给核心算法（用于边权/门控/度量学习）
  - `evolve()`：触发后台演化
  - 自动快照：按事件间隔把状态写入本地 snapshot

- `aurora/runtime/bootstrap.py`：provider 装配与内存创建
  - 创建 LLM provider、embedding provider 与内存实例

- `aurora/integrations/storage/*`：
  - `SQLiteEventLog`（append-only）
  - `SnapshotStore`（pickle）
  - `SQLiteDocStore`（derived artifacts）

---

## 4. “生产级”最关键的一点：可回放 + 确定性 ID

如果你不做可回放，后面任何“模型升级、数据回归、线上追责”都会变成噩梦。

### 4.1 为什么确定性 ID 必须做？

- event log 里记录的是 `event_id`
- 内存里生成的 plot/story/theme id 必须在 replay 后一致  
否则：  
- 用户选择了 `chosen_id` 做反馈，你重启服务后 id 变了，反馈就断了；
- 你想做回归评测，也无法对齐同一条记忆。

### 4.2 我在核心算法里做了什么改动？

在 `aurora/core/memory/engine.py` 我做了**最小侵入式补丁**：

- `AuroraMemory.ingest(..., event_id=...)` 支持传入事件 ID
- plot/story/theme id 使用 `uuid5(namespace, seed)` 生成：
  - plot_id = det_id("plot", event_id)
  - story_id = det_id("story", plot_id)
  - theme_id = det_id("theme", sid)

这样 replay 会稳定复现同一张记忆图。

---

## 5. LLM 的生产拼图：结构化输出 + Schema 版本化 + 失败降级

你架构文档中大量环节用 LLM 辅助（Plot 提取、故事重构、主题命名、自我更新、矛盾检测）。fileciteturn0file1  
生产里最常见的坑是：LLM 输出不稳定导致“脏数据”进入长期记忆。

### 5.1 必须做到的三件事

1. **强制 JSON schema 输出**（不要让 LLM 自由写文本）
2. **schema_version** 固化（升级时可迁移）
3. **失败降级**（LLM 失败时写最小字段，不阻塞热路径）

### 5.2 你可以直接用的 schema & prompts

- `aurora/integrations/llm/schemas.py`：`PlotExtraction / StoryUpdate / ThemeCandidate / SelfNarrativeUpdate / ContradictionJudgement`
- `aurora/integrations/llm/prompts.py`：全部 prompt 模板（带 “ONLY JSON” 指令）

---

## 6. 观测与指标（建议你上线就打）

建议的最小指标集（先打日志和本地状态摘要就够用）：

- **写入指标**
  - ingest_qps
  - encode_rate（plot.id in mem.plots 的比例）
  - snapshot_size_bytes / snapshot_interval
- **检索指标**
  - query_latency_ms
  - hit_kind_distribution（plot/story/theme）
  - feedback_success_rate（有监督时）
- **演化指标**
  - story_count / theme_count / plot_count
  - compression_rate（absorbed/archived）
  - contradiction_edges_count（Guardian）

---

## 7. 隐私与合规拼图（最小可用版）

生产里最容易忽视的是：你并不是“能记就记”，而是“**允许记的才记**”。

我在工程里给了一个最小的 PII redaction hook：

- `aurora/privacy/pii.py`：email/phone/信用卡（极简 regex）
- `AuroraRuntime.ingest_interaction()` 写入 event log 前可脱敏

生产建议：
- PII detector 用更可靠的库/模型
- event log 支持“按 user_id 删除”（Right to be forgotten）
- doc store 与 snapshot 都要支持 purge

---

## 8. 如何把这套生产包接到你的 Agent 上（无侵入接入）

### 8.1 热路径（每次对话一条 event）

- 生成 `event_id`（全局唯一；建议包含 user_id + session_id + turn_index）
- 调用 `/v1/memory/ingest`

### 8.2 查询路径（需要记忆时才 query）

- 将用户问题 + 当前任务目标拼成 query text
- 调用 `/v1/memory/query`
- 把 hits 注入到 Agent 的 prompt（或作为 tool result）

### 8.3 反馈（可选，但强烈推荐）

- 当你能判断某次回答是否成功（explicit rating / implicit success），调用 `/v1/memory/feedback`
- 这会塑形：
  - encode gate（学会“什么值得记”）
  - edge posterior（学会“什么关系有用”）
  - metric（学会“什么相似度对任务有用”）

---

## 9. 你接下来最值得做的 4 个“上线级增强”（按 ROI 排序）

1. **把 embedding provider 替换成更强的生产 embedding**（OpenAI / bge / e5）  
2. **Narrator 加上 story/theme 的 LLM 生成摘要**（写进 doc store）  
3. **Guardian 做 claim contradiction**（生成 contradiction edge + 条件化 theme）  
4. **离线评测 harness**：固定一组 query + label，持续回归（不然你永远不知道升级有没有退化）

---

## 10. 文件清单（你下载后能看到）

- `aurora/runtime/runtime.py`：单用户运行时封装（幂等/回放/快照）
- `aurora/runtime/bootstrap.py`：provider 与内存装配
- `aurora/integrations/storage/event_log.py`：sqlite append-only event log
- `aurora/integrations/storage/snapshot.py`：pickle snapshot
- `aurora/integrations/storage/doc_store.py`：sqlite derived doc store
- `aurora/integrations/llm/*`：schema + prompts + mock provider
- `aurora/core/memory/engine.py`：核心算法（已补确定性 id）
- `aurora/interfaces/api/app.py`：FastAPI endpoints（可选）
- `tests/*`：确定性回放/幂等测试

---

## 11. 下一步我建议你怎么继续（如果你要“完美上线版”）

如果你要继续把它打磨到“完美”，下一步可以继续补齐：

- **FAISS** 的本地向量索引（更大规模记忆时的可选加速）
- **Neo4j** 的 GraphStore（或其他专用图存储）
- **分级存储**：hot plots / warm stories / cold archive（对象存储）
- **审核链路**：每个 Theme/ Self 更新必须能追溯证据 story ids
- **Prompt/Model 版本管理**：每次 LLM 输出记录 model/prompt hash
- **A/B**：对比不同 gate/metric/guardian 策略的真实任务成功率

你如果愿意，我也可以在这份工程基础上继续把 **更强 embedding provider、本地 FAISS 加速、异步演化调度、回归评测 harness** 一次性补齐到更完整的部署形态（但我会默认按你现在的接口风格做，不再反复问你确认）。
