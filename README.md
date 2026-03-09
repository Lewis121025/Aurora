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

chat = runtime.respond(
    session_id="chat_main",
    user_message="你记得我偏向什么样的记忆系统吗？",
)

print(chat.reply)
print(chat.rendered_memory_brief)
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
├── core/graph/                   # MemoryGraph 与本地精确 VectorIndex
├── runtime/runtime.py            # 单用户运行时与重放
├── runtime/bootstrap.py          # provider 装配
├── integrations/storage/         # SQLite event/doc store 与 snapshot
├── interfaces/api/               # FastAPI 接口
├── interfaces/mcp/               # MCP 接口
├── benchmarks/                   # 评测与研究适配器
└── tests/                        # 单元与集成测试
```

### 核心运行链路

1. `AuroraRuntime.respond()` 先检索证据，并构造结构化 `Memory Brief`
2. 运行时只把结构化 brief 和当前用户输入发送给 LLM，不把原始历史对话直接塞进 prompt
3. 回复生成后，`AuroraRuntime.ingest_interaction()` 把本轮交互写入本地 event log 并调用 `AuroraMemory.ingest()`
4. `AuroraMemory` 在内存中更新向量索引、关系图和 plot/story/theme 状态
5. 运行时把 plot 提取结果和 ingest 结果写入本地 SQLite doc store；定期 snapshot 保存整个记忆状态

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

### 终端观测脚本

```bash
aurora
```

默认会直接进入实时对话/观测模式，并在 full 模式下显示每轮最终发给 LLM 的 prompt。LLM 只看到结构化 `Memory Brief`，不会直接看到原始历史对话；终端额外展示 evidence/debug 面板。内建命令用于观察系统内部状态：

- `/query <text>`：只做检索，不生成回复
- `/events 5`：看最近写入的事件日志
- `/inspect <id>`：看 plot/story/theme/doc 的摘要
- `/coherence`：看一致性检查结果
- `/narrative`：看自我叙事摘要
- `/pipeline`：看每轮对话的处理链路
- `/context`：回看上一次结构化 memory context
- `/prompt`：回看上一次发给 LLM 的最终 prompt
- `/observe off|brief|full`：切换观测输出级别

如果你想显式指定模式，也可以：

```bash
aurora observe --observe full
```

或者直接运行脚本入口：

```bash
python scripts/observe_runtime.py --observe full
```

### REST API

```bash
uvicorn aurora.interfaces.api.app:app \
  --host 0.0.0.0 \
  --port 8000
```

核心对话入口是 `POST /v1/memory/respond`，返回：
- `reply`
- `memory_context`
- `rendered_memory_brief`
- `system_prompt` / `user_prompt`
- `retrieval_trace_summary`
- `ingest_result`
- `timings`

### MCP 协议

```python
from aurora.interfaces.mcp import create_mcp_server
```

## 许可证

Proprietary. All rights reserved.
