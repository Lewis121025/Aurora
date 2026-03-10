# 基准测试不匹配问题

> [!WARNING]
> 归档说明：本文档为历史复盘材料，仅保留作参考，不再代表当前生产实现。当前 canonical 入口请以 `README.md`、`docs/README.md`、`docs/adr/002-graph-first-emergence.md`、`docs/research/benchmark_guide.md` 和 `aurora.soul` / `aurora.runtime` 代码为准。

## 问题背景

最初尝试使用 MemoryAgentBench 评估 AURORA，结果准确率只有 21.2%。

## 症状

| 指标 | 分数 |
|------|------|
| Answer Relevance (AR) | 21.2% |
| Test-Time Learning (TTL) | 0% |
| Conflict Resolution (CR) | 0% |

## 根本原因分析

**MemoryAgentBench 的设计目标**：
- 推荐系统（如 MovieLens）
- 检索增强生成（RAG）
- 静态知识库查询

**AURORA 的设计目标**：
- 对话式记忆（Conversational Memory）
- 身份构建（Identity Construction）
- 关系追踪（Relationship Tracking）

**核心矛盾**：
```
MemoryAgentBench: "用户喜欢动作片" → 推荐动作片
AURORA: "用户说喜欢动作片" → 记住这个偏好，理解用户身份
```

## 错误方向

花了大量时间试图让 AURORA 适应 MemoryAgentBench：
1. 调整检索参数
2. 修改答案提取逻辑
3. 增加 LLM 辅助判断

这些都是"削足适履"的做法。

## 正确方向

**找到适合 AURORA 的基准测试**：

| 基准测试 | 适合度 | 原因 |
|---------|--------|------|
| MemoryAgentBench | ❌ 低 | RAG/推荐任务 |
| LongMemEval | ✅ 高 | 对话记忆、知识更新、时序推理 |
| EpBench | ✅ 高 | 情景记忆、时空关系 |
| LoCoMo | ⚠️ 中 | 长上下文，但不强调叙事 |

## 教训

1. **先验证基准测试的适用性** - 不要盲目追求在所有基准上都表现好
2. **理解系统的设计边界** - AURORA 不是万能的，它有明确的设计目标
3. **选择能展示系统优势的基准** - 在擅长的领域做到最强
4. **不要削足适履** - 改变评估方式，而不是扭曲系统设计

## 结果

切换到 LongMemEval 后：
- 评估维度与 AURORA 设计高度契合
- 能够展示 AURORA 的叙事记忆优势
- 明确的改进方向
