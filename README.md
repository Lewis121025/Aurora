# Aurora

面向单用户 AI agent 的本地优先叙事记忆运行时。

Aurora 的主线很简单：

`interfaces -> runtime -> soul -> integrations`

其中 `aurora.soul` 是唯一生产核心。历史研究代码和旧文档已迁移到 `legacy_research_stack` 分支。

## 它做什么

Aurora 不是把整段聊天历史直接塞进 prompt。

它会把交互摄入为 `plot`，再逐步组织成 `story` 和 `theme`，在本地维护：

- 结构化长期记忆
- 可查询的向量与关系图
- 身份状态、张力、修复、梦整合与模式涌现
- 事件日志、派生文档和快照

## 当前架构

```text
aurora/
├── soul/           # 唯一生产核心：记忆摄入、检索、演化
├── runtime/        # 编排层：配置、装配、生命周期、响应流程
├── interfaces/     # CLI / terminal / API / MCP
├── integrations/   # embeddings / llm / storage
├── benchmarks/     # 基准测试与适配器
├── system/         # errors / version
└── utils/          # 通用小工具
```

几个最重要的文件：

- [aurora/runtime/runtime.py](aurora/runtime/runtime.py)
- [aurora/runtime/bootstrap.py](aurora/runtime/bootstrap.py)
- [aurora/soul/engine.py](aurora/soul/engine.py)
- [aurora/soul/retrieval.py](aurora/soul/retrieval.py)
- [aurora/soul/models.py](aurora/soul/models.py)
- [aurora/soul/query.py](aurora/soul/query.py)
- [aurora/soul/facts.py](aurora/soul/facts.py)

## 核心算法

Aurora Soul 当前的主算法簇有 4 组：

1. 记忆摄入  
   `OnlineKDE + LowRankMetric + bounded local graph linking`

2. 结构涌现  
   `Louvain communities + StoryArc/Theme materialization`

3. 场式检索  
   `mean shift attractor tracing + personalized PageRank + contradiction inhibition + query/time/fact boost`

4. 身份演化  
   `contradiction scanning + GraphRepairOperator + GraphDreamOperator + schema consolidation`

## 安装

基础安装：

```bash
pip install -e .
```

常见可选依赖：

```bash
pip install -e '.[api]'
pip install -e '.[bailian]'
pip install -e '.[ark]'
pip install -e '.[dev]'
```

## 快速开始

### 1. 最小本地示例

这套配置不依赖外部 API，适合确认运行链路。

```python
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings

runtime = AuroraRuntime(
    settings=AuroraSettings(
        data_dir=".aurora",
        embedding_provider="hash",
        axis_embedding_provider="hash",
        meaning_provider="heuristic",
        narrative_provider="heuristic",
    )
)

runtime.accept_interaction(
    event_id="evt_001",
    session_id="chat_main",
    user_message="我偏好本地优先、结构化的长期记忆。",
    agent_message="我会优先使用 plot、story、theme 的叙事结构来组织记忆。",
)

query = runtime.query(text="用户偏好什么样的记忆系统", k=5)
print(query.hits[0].snippet if query.hits else "no hits")
```

### 2. 使用真实模型

默认配置文件来自 `.env`，前缀是 `AURORA_`。常见生产配置可以这样写：

```dotenv
AURORA_DATA_DIR=.aurora

AURORA_LLM_PROVIDER=bailian
AURORA_EMBEDDING_PROVIDER=bailian
AURORA_AXIS_EMBEDDING_PROVIDER=bailian

AURORA_BAILIAN_LLM_API_KEY=your_llm_key
AURORA_BAILIAN_EMBEDDING_API_KEY=your_embedding_key

AURORA_BAILIAN_LLM_MODEL=qwen3.5-plus
AURORA_BAILIAN_EMBEDDING_MODEL=text-embedding-v4
```

如果你不想在仓库根目录生成运行数据，把 `AURORA_DATA_DIR` 指到仓库外目录即可。

## CLI

安装后入口命令是 `aurora`。

直接进入全屏终端对话模式。默认布局是左侧对话、右侧状态与 prompt 遥测：

```bash
aurora
```

常用子命令：

```bash
aurora ingest "用户消息" "Agent 回复"
aurora query "用户偏好什么"
aurora respond "你记得我刚才说什么吗？"
aurora job <job_id>
aurora identity
aurora stats
aurora serve --port 8000
aurora observe --max-hits 8
```

终端内常用命令：

- `/identity`
- `/stats`
- `/context`
- `/prompt`
- `/query <text>`
- `/events [n]`
- `/plots [n]`
- `/stories [n]`
- `/themes [n]`
- `/clear`
- `/quit`

也可以直接运行脚本入口：

```bash
python scripts/runtime/observe.py --max-hits 8
```

## HTTP API

安装 API 依赖后启动：

```bash
uvicorn aurora.interfaces.api.app:app --host 127.0.0.1 --port 8000
```

当前主接口在 `/v5/*`：

- `POST /v5/interactions`
- `POST /v5/query`
- `POST /v5/chat/replies`
- `POST /v5/feedback`
- `GET /v5/events/{event_id}`
- `GET /v5/jobs/{job_id}`
- `GET /v5/identity`
- `GET /v5/stats`
- `GET /healthz`

## MCP

Aurora 可以把内存能力暴露成 MCP server：

```python
from aurora.interfaces.mcp.server import create_mcp_server
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings

runtime = AuroraRuntime(settings=AuroraSettings())
server = create_mcp_server(runtime)
```

## 持久化

`AuroraRuntime` 默认在 `data_dir` 下维护本地状态：

- `runtime.sqlite3`
- `snapshots/`

这意味着 Aurora 默认是本地优先、单用户、可重放的运行时，而不是纯内存 demo。

## 阅读顺序

如果你第一次进仓库，推荐这样读：

1. [aurora/runtime/settings.py](aurora/runtime/settings.py)
2. [aurora/runtime/bootstrap.py](aurora/runtime/bootstrap.py)
3. [aurora/runtime/runtime.py](aurora/runtime/runtime.py)
4. [aurora/soul/engine.py](aurora/soul/engine.py)
5. [aurora/soul/retrieval.py](aurora/soul/retrieval.py)
6. [aurora/soul/models.py](aurora/soul/models.py)

如果你是改算法，优先看：

- [aurora/soul/engine.py](aurora/soul/engine.py)
- [aurora/soul/retrieval.py](aurora/soul/retrieval.py)
- [aurora/soul/query.py](aurora/soul/query.py)
- [aurora/soul/facts.py](aurora/soul/facts.py)

如果你是改 provider 或部署层，优先看：

- [aurora/runtime/bootstrap.py](aurora/runtime/bootstrap.py)
- [aurora/integrations/embeddings](aurora/integrations/embeddings)
- [aurora/integrations/llm](aurora/integrations/llm)
- [aurora/integrations/storage](aurora/integrations/storage)

## 开发约定

- 从具体模块导入，不依赖顶层聚合导出
- `aurora/__init__.py` 只保留版本信息
- `aurora.soul` 是唯一生产核心
- 历史研究代码请切到 `legacy_research_stack` 分支查看

基本回归：

```bash
pytest tests -q
```

## 文档

- 文档总览见 [docs/README.md](docs/README.md)
- 研究文档在 [docs/research](docs/research)
- 架构决策在 [docs/adr](docs/adr)
- 历史迁移/质量/旧研究文档已迁移到 `legacy_research_stack` 分支

## 许可证

Proprietary. All rights reserved.
