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
  maintainer_guide.md
  research/
```

## 3 条阅读路径

- 想理解运行时主线：看 [maintainer_guide.md](docs/maintainer_guide.md)
- 想改核心记忆逻辑：从 [engine.py](aurora/core/memory/engine.py) 开始
- 想看评测和研究材料：从 `aurora/benchmarks/` 和 `docs/research/` 开始

## 接口入口

- CLI: `aurora`
- API: `uvicorn aurora.interfaces.api.app:app --host 0.0.0.0 --port 8000`
- MCP: `aurora.interfaces.mcp`

## 许可

Proprietary. All rights reserved.
