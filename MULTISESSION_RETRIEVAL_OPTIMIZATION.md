# Multi-Session 检索策略优化总结

## 优化目标

提升 multi-session 类型问题的准确率（当前 53%），这类问题需要聚合多个会话的信息。

## 问题分析

根据数据分析，multi-session 问题的特征：
- **50%** 包含 "how many"（计数聚合）
- **19%** 包含 "total"（汇总聚合）
- **4%** 包含 "count"（计数）
- **27%** 其他类型

典型问题示例：
- "How many items of clothing do I need to pick up or return from a store?" → 需要从 3 个会话中计数
- "How many projects have I led or am currently leading?" → 需要从 4 个会话中聚合
- "How many days did I spend on camping trips?" → 需要从多个会话中汇总天数

## 实施的优化

### 1. 添加聚合关键词检测 ✅

**文件**: `aurora/algorithms/constants.py`

添加了 `AGGREGATION_KEYWORDS` 常量，包含中英文聚合关键词：
- 中文: "多少", "几个", "总数", "总共", "合计", "一共", "累计", "汇总", "总计", "所有", "全部", "都", "每", "各"
- 英文: "how many", "how much", "total", "sum", "count", "all", "every", "each", "aggregate", "combined", "together", "altogether", "in total", "in all", "number of", "amount of", "quantity of"

添加了 `AGGREGATION_K_MULTIPLIER = 3.0`，用于聚合查询的 k 值调整。

### 2. 实现聚合问题检测 ✅

**文件**: `aurora/algorithms/retrieval/field_retriever.py`

添加了 `_is_aggregation_query()` 方法，用于检测查询是否需要聚合：
```python
def _is_aggregation_query(self, query_text: str) -> bool:
    """Detect if a query requires aggregation across multiple sessions."""
    query_lower = query_text.lower()
    for keyword in AGGREGATION_KEYWORDS:
        if keyword in query_lower:
            return True
    return False
```

更新了 `_classify_query()` 方法，将聚合关键词识别为 `MULTI_HOP` 类型（用于检索策略）。

### 3. 自动调整检索数量 ✅

**文件**: `aurora/algorithms/aurora_core.py`

在 `query()` 方法中添加了聚合检测和 k 值自动调整：

```python
# Check if this is an aggregation query
is_aggregation = self.retriever._is_aggregation_query(text)

# Adjust k based on aggregation detection
if self.benchmark_mode:
    if is_aggregation:
        # Aggregation queries need the most results
        effective_k = max(k, BENCHMARK_AGGREGATION_K)  # 25
    elif detected_type == QueryType.MULTI_HOP:
        effective_k = max(k, BENCHMARK_MULTI_SESSION_K)  # 20
    else:
        effective_k = max(k, BENCHMARK_DEFAULT_K)  # 15
elif is_aggregation:
    # Non-benchmark mode: use multiplier
    effective_k = int(k * AGGREGATION_K_MULTIPLIER)  # k * 3.0
```

**效果**:
- Benchmark 模式：聚合查询自动使用 k=25（`BENCHMARK_AGGREGATION_K`）
- 普通模式：聚合查询使用 k * 3.0（`AGGREGATION_K_MULTIPLIER`）

### 4. 优化聚合 Prompt ✅

**文件**: `aurora/llm/prompts.py`

增强了 multi-session prompt，更明确地强调聚合需求：

```python
'multi-session': """Based on the conversation history from MULTIPLE SESSIONS below, 
answer the question by AGGREGATING information across sessions.

CRITICAL: This question requires information from MULTIPLE CONVERSATIONS/SESSIONS.
You MUST:
1. Search through ALL sessions in the context
2. Extract EVERY relevant piece of information from each session
3. AGGREGATE the information:
   - For "how many" questions: COUNT all instances across all sessions
   - For "total" questions: SUM all values across all sessions  
   - For "all" questions: LIST all items mentioned across all sessions
4. Provide a COMPLETE answer that includes information from all relevant sessions

IMPORTANT: Do NOT miss information from any session. Count carefully and comprehensively.
...
"""
```

更新了 `detect_question_type()` 方法，使用更全面的聚合关键词列表进行自动检测。

### 5. 更新 Benchmark 脚本 ✅

**文件**: `run_longmemeval_baseline.py`

移除了硬编码的 k=25，改为依赖系统的自动检测和调整：
```python
# Query - system will automatically detect aggregation queries and adjust k
k = 10  # Base k, will be auto-adjusted by query() based on query type
trace = memory.query(question, k=k)
```

## 优化效果预期

### 检索层面
- **增加检索量**: 聚合查询从 k=10 提升到 k=25（2.5倍），覆盖更多会话
- **自动检测**: 无需手动指定问题类型，系统自动识别聚合需求

### Prompt 层面
- **明确指令**: LLM 收到更清晰的聚合指令，强调"搜索所有会话"、"计数所有实例"
- **类型感知**: 根据问题类型（how many/total/all）提供针对性指导

### 预期提升
基于之前的分析：
- **检索范围增加**: +5-10% 准确率提升
- **Prompt 优化**: +5-8% 准确率提升
- **综合预期**: +10-18% 准确率提升（从 53% 提升到 63-71%）

## 技术细节

### 关键词匹配策略
- 使用简单的字符串包含匹配（`keyword in query_lower`）
- 支持中英文双语
- 优先级：聚合检测 > 多跳检测 > 事实查询

### k 值调整策略
- **Benchmark 模式**: 使用固定阈值（`BENCHMARK_AGGREGATION_K=25`）
- **普通模式**: 使用倍数调整（`k * AGGREGATION_K_MULTIPLIER`）
- **向后兼容**: 如果手动指定 query_type，仍会使用原有逻辑

### 检索流程
1. 检测查询类型（包括聚合检测）
2. 根据类型和模式调整 k 值
3. 执行检索（使用调整后的 k）
4. 构建 prompt（使用类型感知模板）
5. LLM 生成答案

## 后续优化方向

1. **会话级聚合**: 利用 Story 结构进行会话级别的信息聚合
2. **显式计数**: 在检索结果中标记会话边界，帮助 LLM 更好地聚合
3. **数值提取**: 增强数值提取和验证，确保计数准确性
4. **反馈学习**: 根据错误案例调整关键词和 k 值

## 测试建议

1. 运行 benchmark 测试，对比优化前后的准确率
2. 分析失败案例，检查是否仍有检索不足或聚合失败的问题
3. 调整 `AGGREGATION_K_MULTIPLIER` 和 `BENCHMARK_AGGREGATION_K` 的值，找到最优配置
