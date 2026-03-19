# Aurora

Aurora vNext 是一个单一主体的人类记忆内核。它不再把“当前状态”建模成一组固定字段，也不再通过外部规则去改写旧记忆；Aurora 的真相改为**不可变 memory atoms + 不可变 memory-field edges + 派生 activation cache**。当前态、查询结果和回复上下文都来自当前 memory field 的只读投影。

`MemoryAtom` 的 `atom_kind` 目前覆盖四类节点：

- `evidence`
- `memory`
- `episode`
- `inhibition`

## 安装

```bash
pip install -e '.[dev]'
```

## 配置

在 `.env` 中配置：

```env
AURORA_LLM_PROVIDER=openai
AURORA_LLM_CONFIG_BASE_URL=https://api.openai.com/v1
AURORA_LLM_CONFIG_MODEL=gpt-4o-mini
AURORA_LLM_CONFIG_API_KEY=your-api-key
AURORA_API_KEY=your-http-api-key
```

`AURORA_API_KEY` 可选，用于 HTTP API 鉴权。设置后，除 `/health`、`/docs` 和 `/openapi.json` 外，其余接口都需要 `Authorization: Bearer ...`。

Aurora 的公开 LLM 配置固定为 `llm_settings = {"provider": "...", "config": {...}}`。环境变量只是这套结构的平铺版本。Aurora 的内部 LLM 协议仍然只要求一个 `complete(messages)` 接口。默认适配器当前实现的是 OpenAI-compatible `/chat/completions` 端点。

## Python SDK

```python
from aurora.runtime.engine import AuroraKernel

kernel = AuroraKernel.create(
    llm_settings={
        "provider": "openai",
        "config": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini",
            "api_key": "your-api-key",
        },
    }
)

turn = kernel.turn("subject-alice", "我在杭州工作，也喜欢爵士乐。")
state = kernel.state("subject-alice")
recall = kernel.recall("subject-alice", "杭州 生活", limit=8)

kernel.close()
```

省略 `llm_settings` 时，`AuroraKernel.create()` 会从 `AURORA_LLM_PROVIDER` 和 `AURORA_LLM_CONFIG_*` 读取同一套配置。

公开接口固定为：

- `turn(subject_id, text, now_ts=None) -> TurnOutput`
- `state(subject_id) -> SubjectMemoryState`
- `recall(subject_id, query, limit=8) -> RecallResult`
- `close() -> None`

`state(subject_id)` 返回当前 memory field 视图：

- `summary`: 当前 memory field 的文本摘要
- `atoms`: 当前高激活节点
- `edges`: 当前局部高影响边

`recall(subject_id, query, limit=8)` 返回查询驱动的局部 memory field 切片，同样只包含 `summary`、`atoms`、`edges`。

## CLI

```bash
aurora turn "Hello Aurora" --subject-id subject-alice
aurora state --subject-id subject-alice
aurora recall "我现在住在哪里？" --subject-id subject-alice --limit 8
aurora status
```

## MCP

Aurora 也可以作为 `stdio` MCP server 提供给其他 agent：

```bash
aurora-mcp
```

MCP vNext 暴露：

- tools: `aurora_turn`, `aurora_recall`
- resources: `aurora://subject/{subject_id}/memory-field`

## HTTP API

启动服务：

```bash
uv run uvicorn aurora.surface.api:create_app --factory --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Body | 说明 |
| --- | --- | --- | --- |
| `/health` | `GET` | - | 健康检查 |
| `/turn` | `POST` | `{"subject_id": "...", "text": "...", "now_ts": 0}` | 执行一轮 subject-scoped turn |
| `/state/{subject_id}` | `GET` | - | 查看主体当前 memory field |
| `/recall` | `POST` | `{"subject_id": "...", "query": "...", "limit": 8}` | 返回查询驱动的局部 memory field 切片 |

## 运行模型

每轮 `turn` 固定执行：

1. 写入 user evidence atom
2. 将用户输入编译成新的 memory / episode / inhibition nodes 和 signed weighted edges
3. 对当前 subject 的 memory field 执行写时演化，刷新 activation cache
4. 从当前 memory field 和 query 切片生成 `MemoryBrief`，按 `current_mainline / query_relevant / recent_changes / active_tensions / ongoing_commitments` 组织回复上下文
5. 生成回复
6. 写入 assistant evidence atom
7. 将完整回合编译成 episode / memory / inhibition nodes 和 edges
8. 再执行一次写时演化

Aurora 不再直接改写旧 atom。冲突、淡化、遗忘和延续都通过不可变节点进入图后，借由边关系自然改变当前激活分布。

## LLM Providers

当前内置支持以下 `AURORA_LLM_PROVIDER`：

- `openai`
- `openai_compatible`
- `bailian`

## 系统公理

Aurora 的 memory field kernel 只承诺以下公理：

1. 真相只由不可变 `memory_atoms`、不可变 `memory_edges` 和派生 `activation_cache` 组成。
2. 每个节点都有一个只由本地 retention 和 query seed 决定的内禀激活。
3. 正边只传播高于内禀基线的剩余激活，因此静止场不会自我放大。
4. 负边按源节点当前可达性传播抑制压力，不表达逻辑否定或真值裁决。
5. `state()` 和 `recall()` 返回的是当前 memory field 的只读投影，不是唯一真值声明。
6. `recall()` 以缓存场为初值，但不会回写或污染缓存；激活演化始终有界并收敛到稳定固定点。

## Compiler Boundary

`distillation` 只是一层 proposal compiler，不是真值裁决器。它遵守以下边界：

1. user compiler 只能提议 `memory | inhibition`，completed-turn compiler 只能提议 `memory | episode | inhibition`。
2. 非 JSON、非 object 的 compiler 输出不会被修补，而是沉淀为 `compile_failure` evidence。
3. 非法 `kind`、空文本、非法数值范围、未知引用、自环边都会被直接丢弃。
4. 任何触及 `evidence` 的边都会被过滤；`evidence` 只记录观察痕迹，不进入场耦合。

## Storage Invariants

SQLite store 现在直接守住以下真相边界：

1. `atom_kind` 只能是 `evidence | memory | episode | inhibition`，核心数值范围必须落在合法区间。
2. edge 和 activation cache 都只能引用同一 `subject_id` 下真实存在的 atom。
3. edge 不允许自环，也不允许触及 `evidence` atom。
4. `evidence` 只能作为观察痕迹存在，不能带来源链。
5. `activation_cache` 的替换是原子事务，失败时不会留下半更新状态。
6. 持久化 payload 的恢复是严格的：非法 kind、坏 JSON、错误形状和缺失必填字段都会显式失败，不做静默兜底。

## 质量保障

```bash
uv run pytest -q
uv run mypy aurora tests --show-error-codes --pretty
uv run ruff check aurora tests
python -m compileall -q aurora tests
```

真实百炼链路 smoke 默认不进常规测试，显式开启。该测试使用仓库根目录的 `.env`，并要求 `AURORA_LLM_PROVIDER=bailian` 与对应的 `AURORA_LLM_CONFIG_*`：

```bash
AURORA_LIVE_TESTS=1 uv run pytest -q tests/test_live_bailian_vnext.py
```

## 项目结构

```text
aurora/
├── __main__.py
├── expression/
│   ├── cognition.py
│   └── context.py
├── llm/
│   ├── config.py
│   ├── openai_compat.py
│   └── provider.py
├── memory/
│   ├── ledger.py
│   ├── state.py
│   └── store.py
├── pipelines/
│   └── distillation.py
├── runtime/
│   ├── contracts.py
│   ├── engine.py
│   └── projections.py
└── surface/
    ├── api.py
    ├── cli.py
    └── mcp.py
```

## 架构说明

[`docs/aurora-architecture-blueprint.md`](docs/aurora-architecture-blueprint.md)

## MCP 说明

[`docs/aurora-mcp-v1.md`](docs/aurora-mcp-v1.md)

## 许可证

Proprietary
