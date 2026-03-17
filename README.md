# Aurora

Aurora v2 是一个嵌入式记忆 SDK。它不把记忆当作默认检索层，而是把交互历史编译成一个可持久化、可调试的当前关系状态。

系统只有三层：

1. `Evidence Log`：原始对话与编译失败事件的 append-only 证据流
2. `Relation Field`：默认挂载的长期主观状态，表达过去如何改变了当前姿态
3. `Archive`：按需激活的完整检索层，只负责精确回忆事实与证据

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
report = kernel.compile_pending("default")
snapshot = kernel.snapshot("default")
recall = kernel.recall("default", "我现在住在哪里？")

kernel.close()
```

公开接口固定为：

- `turn(session_id, text, now_ts=None) -> TurnOutput`
- `compile_pending(session_id=None, now_ts=None) -> CompileReport`
- `snapshot(session_id) -> RelationSnapshot`
- `recall(session_id, query, limit=5) -> RecallResult`
- `close() -> None`

## CLI

```bash
aurora turn "Hello Aurora"
aurora compile --session-id default
aurora snapshot --session-id default
aurora recall "我现在住在哪里？" --session-id default
aurora status
```

## HTTP API

| Endpoint | Method | Body | 说明 |
| --- | --- | --- | --- |
| `/health` | `GET` | - | 健康检查 |
| `/turn` | `POST` | `{"session_id": "...", "text": "...", "now_ts": 0}` | 执行一轮热路径 |
| `/compile` | `POST` | `{"session_id": "...", "now_ts": 0}` | 编译 pending turns |
| `/snapshot/{session_id}` | `GET` | - | 查看关系快照 |
| `/recall` | `POST` | `{"session_id": "...", "query": "...", "limit": 5}` | 精确召回 archive |

## 运行模型

热路径只做四件事：

1. 记录 user turn 到 evidence log
2. 读取 `RelationField + Top OpenLoops`
3. 必要时激活 archive recall
4. 生成回复并记录 assistant turn

后台 compiler 单独负责把 pending turns 编译为 `MemoryOp`，再由 reducer 应用到：

- `RelationField`
- `OpenLoop`
- `FactRecord`

冲突事实不会被静默覆盖，而是生成版本链并打开 `contradiction` loop。

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

`AuroraKernel.create()` 现在会优先读取进程环境变量；如果当前 shell 没有导出，也会从仓库根目录的 `.env` 读取通用键 `AURORA_LLM_BASE_URL`、`AURORA_LLM_MODEL`、`AURORA_LLM_API_KEY`，或者按 `AURORA_LLM_PROVIDER` 映射到 provider scoped 键（例如 `AURORA_BAILIAN_LLM_*`）。

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

## 文档

[`docs/aurora-architecture-blueprint.md`](docs/aurora-architecture-blueprint.md)

## 许可证

Proprietary
