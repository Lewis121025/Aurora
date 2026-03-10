# 07 批量导入与评估问题

> [!WARNING]
> 归档说明：本文档为历史复盘材料，仅保留作参考，不再代表当前生产实现。当前 canonical 入口请以 `README.md`、`docs/README.md`、`docs/adr/002-graph-first-emergence.md`、`docs/research/benchmark_guide.md` 和 `aurora.soul` / `aurora.runtime` 代码为准。

## 问题发现

### 1. 性能瓶颈：串行 Embedding

**问题**：LongMemEval 每个样本有 300-500 个对话轮次，每次 embedding API 调用 0.4-0.8s，导致一个样本需要 200+ 秒。

**根本原因**：`AuroraSoul.ingest()` 是为实时在线场景设计的（一次处理一条），不适合基准测试场景（需要批量导入大量历史数据）。

**解决方案**：
- 添加 `ingest_batch()` 方法
- 利用 `BailianEmbedding.embed_batch()` 的批量能力
- 性能提升 ~3x

### 2. 日期信息缺失

**问题**：temporal-reasoning 问题需要知道事件发生的日期，但系统只索引对话内容，不包含日期信息。

**示例**：
- 问题："MoMA 和 Metropolitan 之间相隔几天？"
- 需要：找到两次访问的日期 → 计算天数差
- 实际：日期在 `haystack_dates` 中但没有被索引

**解决方案**：
- 修改 `ingest_batch()` 支持 `date` 字段
- 日期作为前缀附加到文本：`[2023/01/08 (Sun) 12:49] user: ...`

### 3. 弃权机制误触发

**问题**：在检索结果中检测否定词（如 "didn't mention"）会导致误触发弃权。

**根本原因**：弃权检测逻辑检查检索结果中的否定词，但这些否定词可能与当前问题无关。

**解决方案**：
- 只检查查询本身是否是"存在性查询"
- Benchmark 模式下完全禁用弃权

### 4. 问题类型感知

**问题**：不同问题类型需要不同的回答策略，但系统使用统一的 prompt。

**分类**：
| 类型 | 需求 | 策略 |
|------|------|------|
| single-session-user | 提取用户事实 | 直接检索 |
| single-session-assistant | 提取助手回复 | 直接检索 |
| single-session-preference | 推理用户偏好 | 偏好推理 |
| multi-session | 聚合多会话 | 聚合统计 |
| knowledge-update | 追踪知识变化 | 时间线查询 |
| temporal-reasoning | 时间推理 | 时间计算 |

**解决方案**：
- 为每种类型创建专用 prompt 模板
- 在 `detect_question_type()` 中正确分类
- 传递 `question_type_hint` 到 prompt 生成

### 5. 评估方式局限

**问题**：当前评估是简单的字符串包含匹配：
```python
return str(expected).lower() in predicted.lower()
```

对于某些问题类型不合适：
- preference 类型的"答案"是偏好描述，不是直接事实
- temporal 类型需要数值匹配（如 "7 days"）

**需要改进**：考虑更智能的评估方式

## 当前准确率

| 类型 | 准确率 | 分析 |
|------|--------|------|
| single-session-user | 80% | 直接事实检索，效果好 |
| single-session-assistant | 40% | 中等 |
| multi-session | 20-33% | 需要聚合 |
| knowledge-update | 20% | 需要追踪变化 |
| temporal-reasoning | 20% | 时间推理困难 |
| single-session-preference | 0% | 偏好推理 + 评估问题 |

## 下一步

1. 优化 temporal-reasoning：增强时间计算能力
2. 改进 preference 评估：考虑语义相似度而非精确匹配
3. 提升检索质量：分析为什么某些问题检索不到相关内容
