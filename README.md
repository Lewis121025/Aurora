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
│   └── self_narrative.py
├── embeddings/          # 嵌入
├── services/            # CQRS 服务
├── storage/             # 存储
└── api/                 # REST API
```

## 文档

- `docs/AURORA_memory_algorithm.md` - 算法设计
- `docs/narrative_memory_architecture.md` - 架构说明

## 许可

Proprietary. All rights reserved.
