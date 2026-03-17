# Aurora

Aurora 是一个基于第一性原理的自发自适应记忆架构。核心思想：**记忆的本质不是存储过去，而是用过去的数据雕刻现在的 Prompt。**

系统通过三个正交层实现类人的记忆演化：
1. **冷事实层 (ObjectiveLedger)** — 原子事实持久化（SQLite + 384d 向量），纯追加日志
2. **主观状态层 (RelationalState)** — 关系状态 JSON，每次对话全量挂载到 System Prompt，O(1) 直觉投影
3. **认知摩擦与张力队列 (TensionQueue)** — 未解决悬案的优先队列，半衰期衰减，打破 LLM 的完美顺从

---

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

支持任意 OpenAI 兼容接口（阿里云百炼、DeepSeek 等）。**缺少上述环境变量 Aurora 无法启动。**

---

## 使用方式

### 命令行

```bash
aurora turn "Hello Aurora"    # 执行一次认知循环
aurora status                 # 查看引擎状态
```

### HTTP API

| 端点      | 方法 | 说明     |
|-----------|------|----------|
| `/health` | GET  | 健康检查 |
| `/turn`   | POST | 认知循环 |

```json
POST /turn
{
  "session_id": "default",
  "text": "I learned something important today"
}
```

---

## 运行时流转

1. **唤醒 (Wake)** — 挂载 RelationalState + TensionQueue 头部到 context，O(1) 投影
2. **对话 (Chat)** — 实时交互，偶尔从 ObjectiveLedger 补充冷事实
3. **蒸馏 (Background)** — 会话结束或达到 20 轮阈值时，LLM 分析对话，提取认知 Diff：
   - 更新 RelationalState（亲密度 / 氛围 / 交互规则）
   - 提取原子事实沉淀到 ObjectiveLedger
   - 检测认知摩擦（事实矛盾）→ 生成 Tension 悬案

---

## 质量保障

```bash
uv run pytest -q
uv run mypy aurora --show-error-codes --pretty
uv run ruff check aurora
```

---

## 项目目录

```text
aurora/
├── __main__.py              # 包入口：python -m aurora
├── expression/              # 表达层
│   ├── cognition.py         # 统一认知（move + response）
│   └── context.py           # 认知上下文
├── llm/                     # LLM 适配层
│   ├── config.py            # 配置加载
│   ├── openai_compat.py     # OpenAI 兼容客户端
│   └── provider.py          # LLMProvider 协议
├── memory/                  # 记忆系统
│   ├── ledger.py            # 冷事实账本（SQLite + 向量）
│   └── store.py             # 会话内记忆图（Node + Edge）
├── phases/                  # 交互阶段
│   └── awake.py             # awake 阶段
├── pipelines/               # 处理管道
│   └── distillation.py      # 认知蒸馏
├── relation/                # 关系系统
│   ├── state.py             # RelationalState（灵魂载体）
│   └── tension.py           # TensionQueue（悬案队列）
├── runtime/                 # 运行时核心
│   ├── contracts.py         # 核心类型
│   ├── engine.py            # AuroraEngine
│   ├── projections.py       # 状态投影
│   └── state.py             # 运行时状态
└── surface/                 # 对外接口
    ├── api.py               # HTTP API
    └── cli.py               # CLI
```

---

## 架构文档

[`docs/aurora-architecture-blueprint.md`](docs/aurora-architecture-blueprint.md) — 基于第一性原理的架构方案

---

## 许可证

Proprietary