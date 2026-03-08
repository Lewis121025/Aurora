# Aurora

**面向单用户 AI Agent 的本地叙事记忆系统**

Aurora 是一个为 AI Agent 设计的本地长期记忆运行时。它围绕单个用户组织对话记忆，通过语义嵌入、关系图谱和记忆演化机制，让 Agent 在本机上持续存储、检索和理解历史交互、用户偏好与长期线索。

## 核心特性

- **语义记忆检索**：基于语义相似度的高效记忆查询
- **关系图谱**：自动构建和维护记忆间的关系网络
- **记忆演化**：支持记忆的动态更新和压力衰减
- **单用户运行时**：围绕单个聊天主体组织记忆与持久化
- **本地优先部署**：事件日志、文档存储和快照都默认落在本地目录
- **灵活集成**：支持本地嵌入、阿里云百炼和火山方舟等 provider

## 快速开始

### 安装

```bash
pip install -e .
```

可选依赖：

```bash
pip install -e '.[api]'
pip install -e '.[bailian]'
pip install -e '.[ark]'
pip install -e '.[faiss]'
```

### 基础示例

```python
from aurora import AuroraRuntime, AuroraSettings

runtime = AuroraRuntime(
    settings=AuroraSettings(
        data_dir="./data",
        llm_provider="bailian",
        embedding_provider="bailian",
    )
)

runtime.ingest_interaction(
    event_id="evt_001",
    session_id="chat_main",
    user_message="我想做一个能持续学习的聊天记忆系统。",
    agent_message="可以先把对话编码成 plot，再逐步聚合成 story 和 theme。",
)

results = runtime.query(text="持续学习的聊天记忆系统", k=5)
```

**推荐本地配置**：使用百炼 `qwen3.5-plus` 作为文本模型、`text-embedding-v4` 作为嵌入模型，事件日志和派生文档保存在本地 SQLite 中。

先在项目根目录放一个 `.env`：

```dotenv
AURORA_LLM_PROVIDER=bailian
AURORA_EMBEDDING_PROVIDER=bailian
AURORA_BAILIAN_LLM_API_KEY=你的百炼LLM Key
AURORA_BAILIAN_EMBEDDING_API_KEY=你的百炼Embedding Key
AURORA_BAILIAN_LLM_BASE_URL=https://coding.dashscope.aliyuncs.com/v1
AURORA_BAILIAN_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AURORA_BAILIAN_LLM_MODEL=qwen3.5-plus
AURORA_BAILIAN_EMBEDDING_MODEL=text-embedding-v4
```

## 架构设计

### 分层结构

```
aurora/
├── core/              核心算法与领域逻辑
│   ├── memory/        记忆引擎、关系管理、演化机制
│   ├── graph/         图结构与向量索引
│   └── retrieval/     查询分类与检索策略
├── runtime/           运行时编排与生命周期管理
│   ├── bootstrap.py   系统初始化与 provider 配置
│   ├── runtime.py     单用户聊天运行时
│   └── settings.py    配置管理
├── integrations/      第三方系统集成
│   ├── embeddings/    本地与云端嵌入 provider
│   ├── llm/           LLM provider 封装
│   └── storage/       本地 SQLite event/doc store 与快照
├── interfaces/        用户交互接口
│   ├── cli.py         命令行工具
│   ├── api/           REST API
│   └── mcp/           MCP 协议支持
├── benchmarks/        性能评测与研究
└── tests/             测试套件
```

### 设计原则

- **core**：纯领域逻辑，不包含 API、CLI 或评测编排代码
- **runtime**：负责系统组件的装配和生命周期管理
- **integrations**：provider 与本地持久化的接入点，保持算法主链解耦
- **interfaces**：用户可直接调用的入口，与 core 保持独立

### 关键目录

```
aurora/
├── __init__.py                    # 公共 API 导出
├── core/memory/engine.py         # 记忆主引擎
├── core/graph/                   # MemoryGraph、VectorIndex、FAISSIndex
├── runtime/runtime.py            # 单用户运行时与重放
├── runtime/bootstrap.py          # provider 装配
├── integrations/storage/         # SQLite event/doc store 与 snapshot
├── interfaces/api/               # FastAPI 接口
├── interfaces/mcp/               # MCP 接口
├── benchmarks/                   # 评测与研究适配器
└── tests/                        # 单元与集成测试
```

### 核心运行链路

1. `AuroraRuntime.ingest_interaction()` 写入本地 event log，并调用 `AuroraMemory.ingest()`
2. `AuroraMemory` 在内存中更新向量索引、关系图和 plot/story/theme 状态
3. 运行时把 plot 提取结果和 ingest 结果写入本地 SQLite doc store
4. `runtime.query()` 直接基于内存中的索引检索，再从 doc store 补摘要
5. 定期 snapshot 会把整个记忆状态保存到本地目录，供重启后回放恢复

## 开发指南

### 阅读路径

根据你的目标选择相应的阅读顺序：

**维护者主线**（理解系统运行时）
1. `aurora/__init__.py` - 公共 API 入口
2. `aurora/runtime/bootstrap.py` - 系统初始化
3. `aurora/runtime/runtime.py` - 单用户运行时
4. `aurora/core/memory/engine.py` - 记忆引擎

**核心算法开发**（修改记忆逻辑）
- 从 `aurora/core/memory/engine.py` 开始
- 关系管理：`aurora/core/memory/relations.py`
- 记忆演化：`aurora/core/memory/evolution.py`

**集成与扩展**（添加新的 provider）
- 嵌入器：`aurora/integrations/embeddings/`
- LLM：`aurora/integrations/llm/`
- 本地持久化：`aurora/integrations/storage/`

### 常见改动

| 需求 | 文件位置 |
|------|---------|
| 修改记忆编码/检索算法 | `aurora/core/memory/engine.py` |
| 修改关系图谱或演化机制 | `aurora/core/memory/` |
| 切换嵌入模型或 LLM | `aurora/runtime/bootstrap.py` |
| 修改运行时持久化策略 | `aurora/runtime/runtime.py` |
| 扩展 CLI/API/MCP 接口 | `aurora/interfaces/` |
| 添加新的 provider | `aurora/integrations/embeddings/` 或 `aurora/integrations/llm/` |

### 可选阅读

以下内容对理解单用户运行时非必需，但对研究和优化有帮助：

- `aurora/benchmarks/` - 性能评测与基准适配器
- `docs/research/` - 研究文档与论文
- `aurora/scripts/` - 工具脚本

## 接口与部署

### 命令行接口

```bash
aurora [command] [options]
```

### REST API

```bash
uvicorn aurora.interfaces.api.app:app \
  --host 0.0.0.0 \
  --port 8000
```

### MCP 协议

```python
from aurora.interfaces.mcp import create_mcp_server
```

## 许可证

Proprietary. All rights reserved.
