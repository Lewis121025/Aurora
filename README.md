# Aurora

Aurora v3 是一个 relation-only 的关系型陪伴内核。它不再围绕 `session` 组织记忆，而是把每段长期连续性收口到单一 `relation_id` 下。

内核只有两类持久化真相：

1. `Evidence Log`：append-only 的用户/助手事件流
2. `Memory Atoms`：唯一长期语义原子

`RelationField / OpenLoops / Facts` 都是由当前 atoms 派生出来的运行时视图，不再是并列真相层。

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
```

Aurora 只依赖 OpenAI 兼容的 `complete(messages)` 接口。

## Python SDK

```python
from aurora.runtime.engine import AuroraKernel

kernel = AuroraKernel.create()

turn = kernel.turn("default", "以后别用安抚式表达，直接一点。")
snapshot = kernel.snapshot("default")
recall = kernel.recall("default", "我现在住在哪里？")

kernel.close()
```

公开接口固定为：

- `turn(relation_id, text, now_ts=None) -> TurnOutput`
- `snapshot(relation_id) -> RelationSnapshot`
- `recall(relation_id, query, limit=5) -> RecallResult`
- `close() -> None`

## CLI

```bash
aurora turn "Hello Aurora"
aurora snapshot --relation-id default
aurora recall "我现在住在哪里？" --relation-id default
aurora status
```

## HTTP API

| Endpoint | Method | Body | 说明 |
| --- | --- | --- | --- |
| `/health` | `GET` | - | 健康检查 |
| `/turn` | `POST` | `{"relation_id": "...", "text": "...", "now_ts": 0}` | 执行一轮 relation-scoped turn |
| `/snapshot/{relation_id}` | `GET` | - | 查看关系快照 |
| `/recall` | `POST` | `{"relation_id": "...", "query": "...", "limit": 5}` | 调试/运维向的 recall |

## 运行模型

每轮 `turn` 固定执行：

1. 记录 user event
2. 吸收显式高价值信号到 pre-response atoms
3. 基于 atoms 派生 `RelationField + OpenLoops + RecallHits`
4. 生成回复并记录 assistant event
5. 运行 post-response compiler，补充事实、修订、遗忘和 loop 原子

`forget` 不是删除命令，而是新的 memory atom。它会局部影响相关 atoms 的可见性和后续投影，但不会硬删 evidence。

## 质量保障

```bash
uv run pytest -q
uv run mypy aurora tests --show-error-codes --pretty
uv run ruff check aurora tests
```

真实 LLM smoke test 不进入默认测试套件，需要显式触发：

```bash
AURORA_RUN_LIVE_TESTS=1 uv run pytest -q tests/live_llm_smoke.py
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
│   ├── atoms.py
│   ├── ledger.py
│   └── store.py
├── pipelines/
│   └── distillation.py
├── relation/
│   ├── state.py
│   └── tension.py
├── runtime/
│   ├── contracts.py
│   ├── engine.py
│   └── projections.py
└── surface/
    ├── api.py
    └── cli.py
```

## 架构说明

[`docs/aurora-architecture-blueprint.md`](docs/aurora-architecture-blueprint.md)

## 许可证

Proprietary
