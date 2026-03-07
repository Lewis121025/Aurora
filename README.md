# AURORA

Agent 记忆系统，基于叙事心理学原理设计。

## 设计思想

传统记忆系统将记忆视为信息存储。AURORA 采用不同的视角：

**将记忆视为身份的持续构建过程，而非过去的存档。**

这一视角带来三个设计决策：

1. **关系优先**：记忆围绕"我与他人的关系"组织，而非语义相似性
2. **矛盾共存**：区分需要解决的矛盾与应当保留的矛盾
3. **意义演化**：同一事件的意义可随理解深化而改变

## 架构

```
Plot (原子记忆)
  ├─ 事实层：发生了什么
  ├─ 关系层：我在这段关系中是谁
  └─ 身份层：这对"我是谁"意味着什么

     ↓ 聚合

Story (关系叙事)
  └─ 围绕特定关系组织的记忆序列

     ↓ 涌现

Theme (身份维度)
  └─ 对"我是谁"的部分回答
```

## 安装

```bash
pip install -e .
```

## 使用

```python
from aurora import AuroraMemory, MemoryConfig

mem = AuroraMemory(cfg=MemoryConfig(), seed=42)

# 摄入
mem.ingest("用户问了一个关于记忆的问题，我给出了解释")

# 查询（可指定关系上下文）
trace = mem.query("记忆是什么？", asker_id="user_123")

# 演化（整合、反思、遗忘）
mem.evolve()
```

## 目录结构

```
aurora/
├── algorithms/          # 核心算法
│   ├── aurora_core.py   # 主入口
│   ├── tension.py       # 矛盾管理
│   ├── coherence.py     # 一致性
│   ├── self_narrative.py # 自我叙事
│   ├── entity_tracker.py # 实体追踪
│   ├── relationship.py   # 关系管理
│   ├── evolution.py      # 记忆演化
│   ├── pressure.py       # 压力机制
│   ├── causal.py         # 因果推理
│   ├── abstention.py     # 弃权机制
│   ├── knowledge_classifier.py # 知识分类
│   ├── components/       # 算法组件
│   ├── graph/            # 图结构
│   ├── models/           # 数据模型
│   ├── narrator/         # 叙事生成
│   └── retrieval/        # 检索算法
├── embeddings/          # 嵌入实现
│   ├── base.py          # 基础接口
│   ├── ark.py           # Ark 嵌入
│   ├── bailian.py       # 百炼嵌入
│   ├── hash.py          # 哈希嵌入
│   └── local_semantic.py # 本地语义嵌入
├── llm/                 # LLM 集成
│   ├── provider.py      # 提供商接口
│   ├── ark.py           # Ark LLM
│   ├── prompts.py       # 提示模板
│   └── schemas.py       # 数据模式
├── services/            # CQRS 服务
│   ├── ingestion.py     # 摄入服务
│   ├── query.py         # 查询服务
│   └── worker.py        # 后台任务
├── storage/             # 存储层
│   ├── vector_store.py  # 向量存储
│   ├── doc_store.py     # 文档存储
│   ├── state_store.py   # 状态存储
│   ├── event_log.py     # 事件日志
│   └── snapshot.py      # 快照管理
├── api/                 # REST API
│   ├── app.py           # FastAPI 应用
│   └── schemas.py       # API 模式
├── benchmark/           # 基准测试
│   ├── interface.py     # 测试接口
│   ├── metrics.py       # 评估指标
│   └── adapters/        # 适配器
├── mcp/                 # MCP 服务器
├── privacy/             # 隐私保护
├── utils/               # 工具函数
├── cli.py               # 命令行接口
├── service.py           # 主服务
└── hub.py               # 中心协调器
```

## 文档

- [AURORA_memory_algorithm.md](docs/AURORA_memory_algorithm.md) - 核心算法设计
- [narrative_memory_architecture.md](docs/narrative_memory_architecture.md) - 叙事记忆架构
- [AURORA_production_pack.md](docs/AURORA_production_pack.md) - 生产环境部署
- [benchmark_guide.md](docs/benchmark_guide.md) - 基准测试指南
- [longmemeval_sota_analysis.md](docs/longmemeval_sota_analysis.md) - LongMemEval 分析
- [lessons_learned/](docs/lessons_learned/) - 经验总结

## 许可

Proprietary. All rights reserved.
