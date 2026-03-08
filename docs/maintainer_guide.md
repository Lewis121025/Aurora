# AURORA 维护者导读

## 阅读主线

先看这 4 个入口，足够建立维护主线：

1. `aurora/__init__.py`
2. `aurora/runtime/bootstrap.py`
3. `aurora/runtime/tenant.py`
4. `aurora/core/memory/engine.py`

## 分层结构

```text
aurora/
  core/           纯领域和算法
  runtime/        运行时装配、tenant、hub、CQRS
  integrations/   embeddings、llm、storage 等外部接入
  interfaces/     CLI、API、MCP
  benchmarks/     评测与适配器
```

## 目录职责

- `core`: 不放 API、CLI、benchmark 编排代码
- `runtime`: 只负责把 settings、provider、tenant、hub 串起来
- `integrations`: 所有第三方或外部系统接入点
- `interfaces`: 用户可直接调用的入口
- `benchmarks`: 研究和评测，不参与默认运行时主线

## 常见改动入口

- 改记忆编码/检索逻辑：`aurora/core/memory/engine.py`
- 改关系/演化/压力：`aurora/core/memory/`
- 改 provider 选择：`aurora/runtime/bootstrap.py`
- 改租户持久化与重放：`aurora/runtime/tenant.py`
- 改 CLI / API / MCP：`aurora/interfaces/`

## 可以先忽略的部分

- `aurora/benchmarks/`
- `docs/research/`
- `aurora/scripts/`

这些内容对理解默认运行时不是必需的。
