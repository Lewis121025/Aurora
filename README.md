# AURORA

面向 Agent 的叙事记忆系统。

仓库已经按维护者视角重组，默认阅读主线是：

1. `aurora/__init__.py`
2. `aurora/runtime/bootstrap.py`
3. `aurora/runtime/tenant.py`
4. `aurora/core/memory/engine.py`

## 快速开始

```bash
pip install -e .
```

```python
from aurora import AuroraMemory, MemoryConfig

memory = AuroraMemory(cfg=MemoryConfig(dim=64, max_plots=100), seed=42)
memory.ingest("用户：你好。助理：你好，有什么可以帮你？", actors=("user", "assistant"))
trace = memory.query("刚才聊了什么？", k=5)
```

默认嵌入器现在是 `LocalSemanticEmbedding`，不再把新用户带到随机检索路径。

## 目录分层

```text
aurora/
  core/           纯领域和算法
  runtime/        settings、tenant、hub、CQRS 编排
  integrations/   embeddings、llm、storage
  interfaces/     CLI、API、MCP
  benchmarks/     评测与适配器
tests/            与源码分离的测试目录
docs/
  research/
```

## 3 条阅读路径

- 想理解运行时主线：继续看下面的“维护者导读”
- 想改核心记忆逻辑：从 [engine.py](aurora/core/memory/engine.py) 开始
- 想看评测和研究材料：从 `aurora/benchmarks/` 和 `docs/research/` 开始

## 维护者导读

### 阅读主线

先看这 4 个入口，足够建立维护主线：

1. `aurora/__init__.py`
2. `aurora/runtime/bootstrap.py`
3. `aurora/runtime/tenant.py`
4. `aurora/core/memory/engine.py`

### 分层结构

```text
aurora/
  core/           纯领域和算法
  runtime/        运行时装配、tenant、hub、CQRS
  integrations/   embeddings、llm、storage 等外部接入
  interfaces/     CLI、API、MCP
  benchmarks/     评测与适配器
```

### 目录职责

- `core`: 不放 API、CLI、benchmark 编排代码
- `runtime`: 只负责把 settings、provider、tenant、hub 串起来
- `integrations`: 所有第三方或外部系统接入点
- `interfaces`: 用户可直接调用的入口
- `benchmarks`: 研究和评测，不参与默认运行时主线

### 常见改动入口

- 改记忆编码/检索逻辑：`aurora/core/memory/engine.py`
- 改关系/演化/压力：`aurora/core/memory/`
- 改 provider 选择：`aurora/runtime/bootstrap.py`
- 改租户持久化与重放：`aurora/runtime/tenant.py`
- 改 CLI / API / MCP：`aurora/interfaces/`

### 可以先忽略的部分

- `aurora/benchmarks/`
- `docs/research/`
- `aurora/scripts/`

这些内容对理解默认运行时不是必需的。

## 接口入口

- CLI: `aurora`
- API: `uvicorn aurora.interfaces.api.app:app --host 0.0.0.0 --port 8000`
- MCP: `aurora.interfaces.mcp`

## 许可

Proprietary. All rights reserved.
