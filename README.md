# Aurora

**面向 AI Agent 的叙事记忆系统**

Aurora 是一个为 AI Agent 设计的长期记忆管理框架。通过语义嵌入、关系图谱和记忆演化机制，Aurora 使 Agent 能够有效地存储、检索和理解对话历史、用户偏好和交互模式，从而实现更具连贯性和个性化的交互体验。

## 核心特性

- **语义记忆检索**：基于语义相似度的高效记忆查询
- **关系图谱**：自动构建和维护记忆间的关系网络
- **记忆演化**：支持记忆的动态更新和压力衰减
- **多租户支持**：隔离的租户管理和持久化
- **灵活集成**：支持多种嵌入器、LLM 和存储后端

## 快速开始

### 安装

```bash
pip install -e .
```

### 基础示例

```python
from aurora import AuroraMemory, MemoryConfig

# 初始化记忆系统
config = MemoryConfig(dim=64, max_plots=100)
memory = AuroraMemory(cfg=config, seed=42)

# 记录对话
memory.ingest(
    "用户：你好。助理：你好，有什么可以帮你？",
    actors=("user", "assistant")
)

# 查询相关记忆
results = memory.query("刚才聊了什么？", k=5)
```

**默认配置**：使用 `LocalSemanticEmbedding` 作为嵌入器，无需外部依赖。

## 架构设计

### 分层结构

```
aurora/
├── core/              核心算法与领域逻辑
│   ├── memory/        记忆引擎、关系管理、演化机制
│   ├── embedding/     嵌入接口与实现
│   └── exceptions/    异常定义
├── runtime/           运行时编排与生命周期管理
│   ├── bootstrap.py   系统初始化与 provider 配置
│   ├── tenant.py      多租户管理与持久化
│   └── hub.py         中央协调器
├── integrations/      第三方系统集成
│   ├── embeddings/    多种嵌入模型支持
│   ├── llm/           LLM 集成
│   └── storage/       存储后端适配
├── interfaces/        用户交互接口
│   ├── cli/           命令行工具
│   ├── api/           REST API
│   └── mcp/           MCP 协议支持
├── benchmarks/        性能评测与研究
└── tests/             测试套件
```

### 设计原则

- **core**：纯领域逻辑，不包含 API、CLI 或评测编排代码
- **runtime**：负责系统组件的装配和生命周期管理
- **integrations**：所有外部系统的接入点，保持解耦
- **interfaces**：用户可直接调用的入口，与 core 保持独立

### 详细目录结构

```
aurora/
├── __init__.py                    # 公共 API 导出
├── core/
│   ├── memory/
│   │   ├── engine.py             # 记忆存储与检索引擎
│   │   ├── relations.py          # 关系图谱管理
│   │   ├── evolution.py          # 记忆演化与衰减
│   │   └── contracts.py          # 嵌入契约定义
│   ├── embedding/
│   │   ├── base.py               # 嵌入器基类
│   │   └── local.py              # 本地语义嵌入实现
│   └── exceptions.py             # 异常定义
├── runtime/
│   ├── bootstrap.py              # 系统初始化与 provider 配置
│   ├── tenant.py                 # 多租户管理与持久化
│   ├── hub.py                    # 中央协调器
│   ├── settings.py               # 配置管理
│   └── cqrs.py                   # 命令查询职责分离
├── integrations/
│   ├── embeddings/
│   │   ├── openai.py             # OpenAI 嵌入
│   │   └── huggingface.py        # HuggingFace 嵌入
│   ├── llm/
│   │   ├── openai.py             # OpenAI LLM
│   │   └── anthropic.py          # Anthropic LLM
│   └── storage/
│       ├── memory.py             # 内存存储
│       └── sqlite.py             # SQLite 存储
├── interfaces/
│   ├── cli/
│   │   ├── __main__.py           # CLI 入口
│   │   └── commands.py           # 命令定义
│   ├── api/
│   │   ├── app.py                # FastAPI 应用
│   │   └── routes.py             # API 路由
│   └── mcp/
│       ├── __init__.py           # MCP 服务器
│       └── tools.py              # MCP 工具定义
├── benchmarks/
│   ├── adapters/                 # 基准适配器
│   └── metrics.py                # 评测指标
├── scripts/                       # 工具脚本
└── config/                        # 配置文件模板

tests/
├── unit/                          # 单元测试
├── integration/                   # 集成测试
└── fixtures/                      # 测试数据

docs/
├── research/                      # 研究文档
└── api/                           # API 文档
```

### 模块职责详解

#### core/memory
- **engine.py**：记忆的核心存储与检索逻辑，实现向量化存储和语义查询
- **relations.py**：维护记忆间的关系图谱，支持关系查询和遍历
- **evolution.py**：实现记忆的动态演化、重要性评分和压力衰减
- **contracts.py**：定义嵌入契约，确保嵌入器的一致性

#### core/embedding
- **base.py**：嵌入器的抽象接口，定义统一的嵌入 API
- **local.py**：本地语义嵌入实现，无需外部服务

#### runtime
- **bootstrap.py**：系统启动流程，初始化 provider 和依赖注入
- **tenant.py**：租户隔离、会话管理和持久化恢复
- **hub.py**：中央协调器，协调各组件间的交互
- **settings.py**：配置管理，支持环境变量和配置文件
- **cqrs.py**：命令查询职责分离，实现事件驱动架构

#### integrations
- **embeddings/**：多种嵌入模型的适配器（OpenAI、HuggingFace 等）
- **llm/**：LLM 服务的适配器（OpenAI、Anthropic 等）
- **storage/**：存储后端的适配器（内存、SQLite、数据库等）

#### interfaces
- **cli/**：命令行工具，提供交互式操作接口
- **api/**：REST API，支持 HTTP 调用
- **mcp/**：MCP 协议支持，与 Claude 等工具集成

## 开发指南

### 阅读路径

根据你的目标选择相应的阅读顺序：

**维护者主线**（理解系统运行时）
1. `aurora/__init__.py` - 公共 API 入口
2. `aurora/runtime/bootstrap.py` - 系统初始化
3. `aurora/runtime/tenant.py` - 租户管理
4. `aurora/core/memory/engine.py` - 记忆引擎

**核心算法开发**（修改记忆逻辑）
- 从 `aurora/core/memory/engine.py` 开始
- 关系管理：`aurora/core/memory/relations.py`
- 记忆演化：`aurora/core/memory/evolution.py`

**集成与扩展**（添加新的 provider）
- 嵌入器：`aurora/integrations/embeddings/`
- LLM：`aurora/integrations/llm/`
- 存储：`aurora/integrations/storage/`

### 常见改动

| 需求 | 文件位置 |
|------|---------|
| 修改记忆编码/检索算法 | `aurora/core/memory/engine.py` |
| 修改关系图谱或演化机制 | `aurora/core/memory/` |
| 切换嵌入模型或 LLM | `aurora/runtime/bootstrap.py` |
| 修改租户持久化策略 | `aurora/runtime/tenant.py` |
| 扩展 CLI/API/MCP 接口 | `aurora/interfaces/` |
| 添加新的 provider | `aurora/integrations/` |

### 可选阅读

以下内容对理解核心运行时非必需，但对研究和优化有帮助：

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
from aurora.interfaces.mcp import MCP
mcp = MCP()
```

## 许可证

Proprietary. All rights reserved.
