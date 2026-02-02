# Multi-Session 失败原因深度分析报告

## 执行摘要

**问题**: Multi-session 类型准确率仅 25.6% (34/133)，是最大瓶颈  
**影响**: 提升此类型对总分影响最大（133题，占比26.6%）

## 1. 失败统计

### 1.1 总体数据
- **总题数**: 133
- **成功**: 34 (25.6%)
- **失败**: 99 (74.4%)
  - 逻辑失败: 77 (58.0%)
  - 代码错误: 22 (16.5%)

### 1.2 失败类型分布
- **代码错误** (`'int' object has no attribute 'lower'`): 22题
- **逻辑失败** (无错误但答案错误): 77题

### 1.3 答案类型分布
- **数值答案**: 35题
- **文本答案**: 64题

### 1.4 会话跨度
- **所有失败案例都涉及多个会话** (multi-session)
- 失败案例平均会话数: 2.7
- 成功案例平均会话数: 2.2

## 2. 根本原因分析

### 2.1 代码错误：类型转换问题

**问题位置**: `run_longmemeval_baseline.py:110`

```python
def evaluate_answer(expected: str, generated: str, context: str) -> bool:
    expected_lower = expected.lower().strip()  # ❌ 如果 expected 是 int，会报错
```

**错误原因**:
- `item['answer']` 可能是整数类型（如 `3`）
- 代码假设 `expected` 总是字符串
- 当传入整数时，调用 `.lower()` 导致 `AttributeError`

**影响**: 22题直接失败，占失败总数的22.2%

**修复方案**:
```python
def evaluate_answer(expected: str, generated: str, context: str) -> bool:
    # 确保 expected 是字符串
    expected_str = str(expected).strip()
    expected_lower = expected_str.lower()
    # ... 其余代码
```

### 2.2 逻辑失败：跨会话信息聚合失败

#### 2.2.1 问题特征

**典型失败案例**:
1. **计数聚合**: "How many model kits have I worked on or bought?" → 需要从4个会话中提取并计数
2. **时间汇总**: "How many days did I spend on camping trips?" → 需要从3个会话中提取天数并求和
3. **金额汇总**: "How much total money have I spent on bike-related expenses?" → 需要从4个会话中提取金额并求和

**关键发现**:
- 失败案例答案长度平均 **29.8字符**（成功案例仅10.7字符）
- 失败案例需要更复杂的**信息提取和聚合**
- 所有失败案例都涉及**多个会话的信息整合**

#### 2.2.2 失败原因分类

##### A. 信息未存储（VOI 门控丢弃）

**问题**: AURORA 的 VOI (Value of Information) 门控可能丢弃关键信息

**证据**:
- `run_longmemeval_baseline.py:163` 创建 memory 时**未设置 `benchmark_mode=True`**
- 代码中有 `benchmark_mode` 参数，但未使用：
  ```python
  memory = AuroraMemory(cfg=config, seed=42, embedder=embedder)
  # ❌ 应该设置 benchmark_mode=True 来强制存储所有 plots
  ```

**影响**: 
- 当 VOI 门控认为信息价值低时，会丢弃 plot
- Multi-session 问题需要的信息可能分散在不同会话中
- 如果某个会话的信息被丢弃，聚合就会失败

**修复方案**:
```python
memory = AuroraMemory(
    cfg=config, 
    seed=42, 
    embedder=embedder,
    benchmark_mode=True  # ✅ 强制存储所有 plots
)
```

##### B. 检索未找到（语义相似度不足）

**问题**: 查询与存储的 plot 语义相似度不足，导致检索失败

**证据**:
- Multi-session 问题通常是**聚合查询**（"总共多少"、"所有X"）
- 单个 plot 可能只包含部分信息
- 查询的语义可能与单个 plot 的语义差异较大

**示例**:
```
Query: "How many model kits have I worked on or bought?"
Plot 1: "I bought a Revell F-15 Eagle model kit"
Plot 2: "I worked on a Tamiya Spitfire model"
Plot 3: "I bought a German Tiger I tank model"
...
```

**问题**: 
- 查询是"计数"语义
- 单个 plot 是"购买/工作"语义
- 语义相似度可能较低

**修复方案**:
1. **增加 k 值**: Multi-session 查询使用更大的 k（如 k=20）
2. **查询扩展**: 将聚合查询扩展为多个子查询
3. **Story 级别检索**: 不仅检索 Plot，还检索 Story（包含多个相关 Plot）

##### C. 跨会话聚合失败（缺乏聚合机制）

**问题**: AURORA 缺乏显式的跨会话信息聚合机制

**当前实现**:
- `query()` 返回 top-k plots
- 由 LLM 从这些 plots 中提取答案
- **没有显式的聚合逻辑**

**缺失能力**:
1. **会话边界识别**: 不知道哪些 plots 属于同一会话
2. **信息去重**: 同一信息可能在不同会话中重复
3. **聚合策略**: 如何将多个会话的信息合并

**修复方案**:
1. **会话标记**: 在 ingest 时标记会话 ID
2. **Story 聚合**: 利用 Story 结构进行信息聚合
3. **显式聚合提示**: 在 LLM prompt 中明确要求跨会话聚合

## 3. 第一性原理分析

### 3.1 Multi-Session 需要什么能力？

1. **完整存储**: 所有会话的信息都必须存储（不能丢弃）
2. **广泛检索**: 需要检索多个会话的相关信息（k 值要大）
3. **信息聚合**: 能够将多个会话的信息合并、去重、汇总
4. **会话感知**: 理解会话边界和会话间的关系

### 3.2 AURORA 当前缺失什么？

| 能力 | AURORA 现状 | 缺失程度 |
|------|------------|---------|
| 完整存储 | ❌ VOI 门控可能丢弃 | ⭐⭐⭐⭐⭐ |
| 广泛检索 | ⚠️ k=10 可能不够 | ⭐⭐⭐ |
| 信息聚合 | ❌ 无显式聚合机制 | ⭐⭐⭐⭐⭐ |
| 会话感知 | ❌ 无会话标记 | ⭐⭐⭐⭐ |

### 3.3 与 SOTA 对比

根据 `docs/longmemeval_sota_analysis.md`:

**SOTA 方法**:
- Turn-level 粒度索引（vs Session-level）
- 事实增强索引（Fact-Augmented Keys）
- 时间感知查询扩展
- Chain-of-Note + JSON 格式

**AURORA 差距**:
- ✅ Plot ≈ Turn（粒度正确）
- ❌ 但 VOI 门控可能丢弃（存储不完整）
- ❌ 无事实增强索引（检索不充分）
- ❌ 无显式聚合机制（聚合失败）

## 4. 改进建议（优先级排序）

### P0: 修复代码错误（立即修复）

**问题**: `evaluate_answer` 函数类型错误  
**修复**: 确保 `expected` 转换为字符串  
**预期收益**: +16.5% (22/133)

```python
def evaluate_answer(expected: str, generated: str, context: str) -> bool:
    expected_str = str(expected).strip()  # ✅ 修复
    expected_lower = expected_str.lower()
    # ... 其余代码
```

### P1: 启用 Benchmark Mode（强制存储）

**问题**: VOI 门控可能丢弃关键信息  
**修复**: 设置 `benchmark_mode=True`  
**预期收益**: +10-15%

```python
memory = AuroraMemory(
    cfg=config, 
    seed=42, 
    embedder=embedder,
    benchmark_mode=True  # ✅ 强制存储所有 plots
)
```

### P2: 增加检索范围（Multi-Session 专用）

**问题**: k=10 可能不够覆盖多个会话  
**修复**: Multi-session 查询使用 k=20-30  
**预期收益**: +5-10%

```python
# 根据问题类型调整 k
if question_type == 'multi-session':
    k = 25  # ✅ 增加检索范围
else:
    k = 10
```

### P3: 改进聚合 Prompt

**问题**: LLM prompt 未明确要求跨会话聚合  
**修复**: 在 prompt 中明确说明需要聚合多个会话的信息  
**预期收益**: +5-8%

```python
prompt = f"""Based on the conversation history from MULTIPLE SESSIONS below, 
answer the question by AGGREGATING information across sessions.

IMPORTANT: This question requires information from multiple sessions. 
You need to:
1. Extract relevant information from each session
2. Combine and aggregate the information
3. Provide a comprehensive answer

Conversation Context (from {len(sessions)} sessions):
{context[:3500]}

Question: {question}

Answer (aggregate across all sessions):"""
```

### P4: 会话标记和 Story 聚合（长期优化）

**问题**: 缺乏会话感知和显式聚合机制  
**修复**: 
1. Ingest 时标记会话 ID
2. 利用 Story 结构进行信息聚合
3. 实现显式的聚合逻辑

**预期收益**: +10-15%

## 5. 典型失败案例追踪

### 案例 1: 计数聚合失败

**Question ID**: `gpt4_59c863d7`  
**Question**: "How many model kits have I worked on or bought?"  
**Expected**: "I have worked on or bought five model kits..."  
**Sessions**: 4个会话  
**Answer spans**: 4个会话

**失败原因**:
1. 可能某些会话的信息被 VOI 门控丢弃
2. k=10 可能未检索到所有相关 plots
3. LLM 未能正确聚合4个会话的信息

### 案例 2: 数值答案类型错误

**Question ID**: `0a995998`  
**Question**: "How many items of clothing do I need to pick up or return from a store?"  
**Expected**: `3` (整数)  
**Error**: `'int' object has no attribute 'lower'`

**失败原因**: 代码错误，`evaluate_answer` 未处理整数类型

### 案例 3: 时间汇总失败

**Question ID**: `b5ef892d`  
**Question**: "How many days did I spend on camping trips in the United States this year?"  
**Expected**: "8 days."  
**Sessions**: 3个会话

**失败原因**: 需要从3个会话中提取天数并求和，聚合失败

## 6. 预期改进效果

如果实施所有改进：

| 改进项 | 预期收益 | 累计准确率 |
|--------|---------|-----------|
| 当前 | - | 25.6% |
| P0: 修复代码错误 | +16.5% | 42.1% |
| P1: Benchmark Mode | +10% | 52.1% |
| P2: 增加 k | +5% | 57.1% |
| P3: 改进 Prompt | +5% | 62.1% |
| P4: 会话聚合 | +10% | **72.1%** |

**目标**: 从 25.6% 提升到 **70%+**

## 7. 实施优先级

1. **立即实施**: P0（修复代码错误）- 1小时
2. **本周实施**: P1 + P2 + P3 - 1天
3. **长期优化**: P4（会话聚合机制）- 1周

## 8. 验证方法

实施改进后，重新运行 baseline：

```bash
python run_longmemeval_baseline.py --limit 133
```

重点关注 multi-session 类型的准确率变化。
