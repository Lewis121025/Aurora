# Aurora

**Aurora** 是一个具备连续性记忆的运行时系统，通过 `awake`/`doze`/`sleep` 三态维持行为连贯性。

Aurora 维护内部状态并累积关系历史，在外部行为之前完成内部认知过程。系统依赖 LLM 作为核心认知引擎——单次调用同时产生触碰（touch）、关系移动（relation shift）与响应（response）。

---

## 安装

```bash
pip install -e '.[dev]'
```

## 配置

### LLM 接入

复制 `.env.example` 至 `.env` 并配置：

```env
AURORA_LLM_BASE_URL=https://api.openai.com/v1
AURORA_LLM_MODEL=gpt-4o-mini
AURORA_LLM_API_KEY=your-api-key
```

支持任意 OpenAI 兼容接口（阿里云百炼、DeepSeek 等）。**缺少上述环境变量 Aurora 无法启动。**

---

## 使用方式

### 命令行

```bash
aurora turn "Hello Aurora"    # 执行一次认知循环
aurora doze                   # 进入低活跃状态
aurora sleep                  # 进入休眠状态
```

### HTTP API

| 端点         | 方法  | 说明         |
|--------------|-------|--------------|
| `/health`    | GET   | 健康检查     |
| `/state`     | GET   | 查询当前状态 |
| `/turn`      | POST  | 执行认知循环 |
| `/doze`      | POST  | 进入 doze    |
| `/sleep`     | POST  | 进入 sleep   |

**请求示例**：

```json
POST /turn
{
  "session_id": "default",
  "text": "I learned something important today"
}
```

---

## 质量保障

```bash
# 测试
uv run pytest -q

# 类型检查
uv run mypy aurora --show-error-codes --pretty

# 代码风格
uv run ruff check aurora
```

---

## 项目目录

```text
aurora/
├── __main__.py              # 包入口：python -m aurora
├── being/                   # 本体状态层
│   ├── metabolic_state.py   # 代谢状态：awake/doze/sleep 三态定义
│   └── orientation.py       # 本体定向与姿态
├── evaluation/              # 评估层
│   ├── continuity.py        # 连续性校验
│   ├── relation_dynamics.py # 关系动力学评估
│   └── sleep_effects.py     # 睡眠效应评估
├── expression/              # 表达层
│   ├── cognition.py         # 统一认知表达（触碰 + 移动 + 响应）
│   └── context.py           # 认知上下文构建
├── llm/                     # LLM 适配层
│   ├── config.py            # LLM 配置加载
│   ├── openai_compat.py     # OpenAI 兼容客户端
│   └── provider.py          # LLMProvider 协议定义
├── memory/                  # 记忆系统
│   ├── affinity.py          # 记忆亲和度计算
│   ├── association.py       # 联想机制
│   ├── doze_ops.py          # doze 状态记忆操作
│   ├── fragment.py          # 记忆片段原语
│   ├── knot.py              # 记忆结（Knot）
│   ├── recall.py            # 记忆检索
│   ├── reweave.py           # 记忆重织接口
│   ├── reweave_engine.py    # 重织引擎实现
│   ├── sediment.py          # 记忆沉积层
│   ├── store.py             # 记忆存储后端
│   ├── tags.py              # 记忆标签系统
│   ├── thread.py            # 记忆线程（Thread）
│   └── trace.py             # 记忆轨迹（Trace）
├── persistence/             # 持久化层
│   ├── migrations.py        # 数据库迁移
│   └── store.py             # SQLite 持久化存储
├── phases/                  # 生命周期相位
│   ├── awake.py             # awake 阶段逻辑
│   ├── doze.py              # doze 阶段逻辑
│   ├── outcomes.py          # 阶段产出定义
│   ├── sleep.py             # sleep 阶段逻辑
│   └── transitions.py       # 状态转移编排
├── relation/                # 关系系统
│   ├── formation.py         # 关系形成机制
│   ├── moment.py            # 关系时刻（RelationMoment）
│   └── store.py             # 关系存储
├── runtime/                 # 运行时核心
│   ├── contracts.py         # 运行时合约定义
│   ├── engine.py            # AuroraEngine 核心实现
│   ├── projections.py       # 状态投影
│   └── state.py             # 运行时状态机
└── surface/                 # 对外接口层
    ├── api.py               # HTTP API 端点
    ├── cli.py               # 命令行入口
    └── schemas.py           # 请求/响应 Schema
```

---

## 架构文档

| 文档 | 说明 |
|------|------|
| [`docs/aurora-architecture-principles.md`](docs/aurora-architecture-principles.md) | 北极星原则 |
| [`docs/aurora-final-architecture-blueprint.md`](docs/aurora-final-architecture-blueprint.md) | 目标系统形态 |
| [`docs/aurora-module-map.md`](docs/aurora-module-map.md) | 模块边界定义 |
| [`docs/aurora-rewrite-roadmap.md`](docs/aurora-rewrite-roadmap.md) | 迭代路线图 |
| [`docs/aurora-api-contract.md`](docs/aurora-api-contract.md) | HTTP API 合约 |

---

## 许可证

Proprietary
