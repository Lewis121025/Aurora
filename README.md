# Aurora

Aurora vNext 是一个单一主体的人类记忆内核。它把长期记忆统一收敛成一个 `MemoryAtom` ledger。`memory_atoms` 是唯一内部持久化真相；公开 `state` 只暴露主体当前可见的 pure projection views，不回传 raw atoms 或 provenance。evidence、episode、semantic、procedural、cognitive、affective、narrative、inhibition 都只是 `atom_kind`，不再是并列系统。

`MemoryAtom` 的 `atom_kind` 目前覆盖八类长期记忆：

- `evidence`
- `episode`
- `semantic`
- `procedural`
- `cognitive`
- `affective`
- `narrative`
- `inhibition`

## 安装

```bash
pip install -e '.[dev]'
```

## 配置

复制 `.env.example` 至 `.env` 并配置：

```env
AURORA_LLM_BASE_URL=https://api.openai.com/v1
AURORA_LLM_MODEL=gpt-4o-mini
AURORA_LLM_API_KEY=your-api-key
AURORA_API_KEY=your-http-api-key
```

`AURORA_API_KEY` 可选，用于 HTTP API 鉴权。设置后，除 `/health`、`/docs` 和 `/openapi.json` 外，其余接口都需要 `Authorization: Bearer ...`。

Aurora 的 LLM 只要求一个 `complete(messages)` 接口。默认适配器当前实现的是 OpenAI-compatible `/chat/completions` 端点。

## Python SDK

```python
from aurora.runtime.engine import AuroraKernel

kernel = AuroraKernel.create()

turn = kernel.turn("subject-alice", "我在杭州工作，也喜欢爵士乐。")
state = kernel.state("subject-alice")
recall = kernel.recall("subject-alice", "杭州 生活", mode="blended", temporal_scope="current")

kernel.close()
```

公开接口固定为：

- `turn(subject_id, text, now_ts=None) -> TurnOutput`
- `state(subject_id) -> SubjectMemoryState`
- `recall(subject_id, query, limit=5, mode="blended", temporal_scope="current|historical|both") -> RecallResult`
- `close() -> None`

`state(subject_id)` 只返回投影视图字段：

- `semantic_self_model`
- `semantic_world_model`
- `procedural_memory`
- `active_cognition`
- `affective_state`
- `narrative_state`
- `recent_episodes`

## CLI

```bash
aurora turn "Hello Aurora" --subject-id subject-alice
aurora state --subject-id subject-alice
aurora recall "我现在住在哪里？" --subject-id subject-alice --mode blended --temporal-scope current
aurora recall "我以前住在哪里？" --subject-id subject-alice --mode blended --temporal-scope historical
aurora status
```

## MCP

Aurora 也可以作为 `stdio` MCP server 提供给其他 agent：

```bash
aurora-mcp
```

MCP vNext 暴露：

- tools: `aurora_turn`, `aurora_recall`
- resources: `aurora://subject/{subject_id}/state`

`aurora_recall` 需要显式传入 `temporal_scope`，例如 `current`、`historical` 或 `both`。

## HTTP API

启动服务：

```bash
uv run uvicorn aurora.surface.api:create_app --factory --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Body | 说明 |
| --- | --- | --- | --- |
| `/health` | `GET` | - | 健康检查 |
| `/turn` | `POST` | `{"subject_id": "...", "text": "...", "now_ts": 0}` | 执行一轮 subject-scoped turn |
| `/state/{subject_id}` | `GET` | - | 查看主体当前可见记忆状态 |
| `/recall` | `POST` | `{"subject_id": "...", "query": "...", "limit": 5, "mode": "blended", "temporal_scope": "current"}` | scope-aware recall，返回 `temporal_scope` 和 hits |

`/state/{subject_id}` 只返回投影视图字段，不包含 raw atoms 或 provenance。`/recall` 的返回结果包含 `subject_id`、`query`、`mode`、`temporal_scope` 和 `hits`。

## 运行模型

每轮 `turn` 固定执行：

1. 写入 user evidence atom
2. 先把 user message 结构化编译进当前记忆
3. 用显式 `temporal_scope` recall 读取当前、历史或双态记忆
4. 基于更新后的 `state + recall` 生成回复
5. 写入 assistant evidence atom
6. 生成 episode atom，并编译 assistant commitments / narrative updates
7. 运行 reconsolidation / inhibition lifecycle 更新

`inhibition` 不会删除 evidence atom。它只会抑制相关 atoms 的可访问性和当前连续性。公开 `state` 只返回 projection views，不回传 raw atoms 或 provenance。

assistant 侧明确承诺的未来动作，例如“我会提醒你……”，会进入 `procedural_memory`，并带上 `owner="aurora"` 与 `trigger="assistant_commitment"`，这样它不只停留在 episode 场景里。

`cognitive`、`affective`，以及当前 `trigger="plan"` 的 `procedural` atoms 在公开状态里都被当作当前快照，而不是无限累积的历史列表。新的同类状态会 supersede 旧快照。

## 质量保障

```bash
uv run pytest -q
uv run mypy aurora tests --show-error-codes --pretty
uv run ruff check aurora tests
python -m compileall -q aurora tests
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

[`docs/aurora-architecture-blueprint.md`](/Users/lewis/Aurora/docs/aurora-architecture-blueprint.md)

## MCP 说明

[`docs/aurora-mcp-v1.md`](/Users/lewis/Aurora/docs/aurora-mcp-v1.md)

## 许可证

Proprietary
