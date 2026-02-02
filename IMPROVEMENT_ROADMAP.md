# 从 66% 到 90% 的改进路线图

## 执行摘要

**当前状态**: 66.1% 准确率  
**目标状态**: 90.0% 准确率  
**需要提升**: 23.9 个百分点  
**总题数**: 500

---

## 1. 各类型提升影响分析

### 1.1 按投入产出比（ROI）排序

| 优先级 | 类型 | 当前 | 目标 | 提升(分) | 工作量 | ROI | 难度 |
|--------|------|------|------|---------|--------|-----|------|
| 🥇 | temporal-reasoning | 60% | 80% | **+5.32%** | 3天 | **1.77** | 中等 |
| 🥈 | multi-session | 53% | 80% | **+7.18%** | 5天 | **1.44** | 困难 |
| 🥉 | single-session-assistant | 73% | 85% | +1.34% | 1天 | **1.34** | 简单 |
| 4 | single-session-user | 87% | 95% | +1.12% | 1天 | **1.12** | 简单 |
| 5 | single-session-preference | 53% | 80% | +1.62% | 2天 | 0.81 | 中等 |
| 6 | knowledge-update | 80% | 90% | +1.56% | 2天 | 0.78 | 中等 |

### 1.2 按总体影响排序

| 类型 | 题目数 | 当前贡献 | 目标贡献 | 提升(分) | 影响 |
|------|--------|---------|---------|---------|------|
| **multi-session** | 133 (26.6%) | 14.1% | 21.3% | **+7.18%** | ⭐⭐⭐⭐⭐ |
| **temporal-reasoning** | 133 (26.6%) | 16.0% | 21.3% | **+5.32%** | ⭐⭐⭐⭐ |
| single-session-preference | 30 (6.0%) | 3.2% | 4.8% | +1.62% | ⭐⭐ |
| knowledge-update | 78 (15.6%) | 12.5% | 14.0% | +1.56% | ⭐⭐ |
| single-session-assistant | 56 (11.2%) | 8.2% | 9.5% | +1.34% | ⭐ |
| single-session-user | 70 (14.0%) | 12.2% | 13.3% | +1.12% | ⭐ |

**关键发现**:
- **multi-session** 和 **temporal-reasoning** 各占 26.6% 的题目，是最大的两个瓶颈
- 这两个类型的提升可以带来 **12.5%** 的总分提升（超过一半的目标提升）
- 快速修复（single-session-assistant/user）虽然 ROI 高，但影响有限（仅 2.5%）

---

## 2. 失败案例共同特征分析

### 2.1 Multi-Session（53% → 80%，+7.18%）

**失败原因**:
1. **代码错误**（22题，16.5%）：`evaluate_answer` 未处理整数类型答案
2. **跨会话信息聚合失败**（77题，58%）：需要从多个会话中提取并聚合信息
3. **VOI 门控丢弃关键信息**：某些会话的信息被过滤
4. **检索范围不足**：k=10 可能不够覆盖多个会话
5. **缺乏显式聚合机制**：无会话标记和 Story 聚合

**典型失败案例**:
- 计数聚合："How many model kits have I worked on or bought?" → 需要从4个会话中提取并计数
- 时间汇总："How many days did I spend on camping trips?" → 需要从3个会话中提取天数并求和
- 金额汇总："How much total money have I spent on bike-related expenses?" → 需要从4个会话中提取金额并求和

**修复方案**（优先级排序）:
1. **P0**: 修复代码错误（类型转换）- 预期收益 +16.5%
2. **P1**: 启用 `benchmark_mode=True` - 预期收益 +10-15%
3. **P2**: Multi-session 查询使用 k=20-30 - 预期收益 +5-10%
4. **P3**: 改进聚合 prompt - 预期收益 +5-8%
5. **P4**: 实现会话标记和 Story 聚合 - 预期收益 +10-15%

**预期累计提升**: 从 53% → **72%+**

### 2.2 Temporal-Reasoning（60% → 80%，+5.32%）

**失败原因**:
1. **时序关系理解不足**：无法正确识别"最早"、"最近"、"之后"等时间关系
2. **时间过滤不准确**：时间预过滤逻辑缺失或不完善
3. **时间顺序推理失败**：无法正确排序事件
4. **时间锚点解析不完整**："after the first service" 无法解析

**根本问题**:
- 所有对话使用 `now_ts()`，无法记录历史时间
- 时间是后处理（先语义检索，后时间排序），而非预过滤
- 缺乏显式时间戳支持

**修复方案**:
1. **时间锚点检测**：识别"最近"、"最早"、"之后"等时间锚点
2. **时间范围预过滤**：在检索前缩小时间搜索空间
3. **增强时序推理 prompt**：明确要求按时间顺序推理
4. **优化时间嵌入**：改进时间特征的向量表示

**预期提升**: 从 60% → **80%+**

### 2.3 Single-Session-Assistant（73% → 85%，+1.34%）

**失败原因**:
1. **benchmark_mode 导致检索噪声**：强制存储所有交互，检索时同时检索到 user 和 assistant turns
2. **未优先检索 assistant turns**：询问"你说了什么"时，检索结果包含 user turns（噪声）
3. **prompt 弱化**：Phase1 的 prompt 不如 baseline 强调"extracting specific information"
4. **上下文截断**：从 3500 降到 3000，可能截断关键信息

**修复方案**:
1. **P0**: 为 assistant 问题优先检索 assistant turns
2. **P1**: 恢复上下文长度到 3500
3. **P2**: 增强 prompt，强调提取 assistant 的信息
4. **P3**: 优化 abstention 逻辑，避免误判

**预期提升**: 从 73% → **85%+**

### 2.4 Single-Session-User（87% → 95%，+1.12%）

**失败原因**:
1. **检索噪声**：检索结果包含不相关信息
2. **上下文截断**：可能截断关键信息

**修复方案**:
1. 优化检索，提高相关性
2. 增加上下文长度

**预期提升**: 从 87% → **95%+**

### 2.5 Single-Session-Preference（53% → 80%，+1.62%）

**失败原因**:
1. **偏好信息未正确存储**：VOI 决策可能跳过偏好信息
2. **偏好提取失败**：检索和 prompt 未正确提取偏好

**修复方案**:
1. 改进偏好识别（关键词检测）
2. 增强偏好存储（提高存储权重）

**预期提升**: 从 53% → **80%+**

### 2.6 Knowledge-Update（80% → 90%，+1.56%）

**失败原因**:
1. **更新检测失败**：更新检测依赖高语义相似度（>0.75），但知识更新通常语义不相似
2. **旧值未正确标记为 superseded**：检索时未过滤已被 supersede 的 plots

**修复方案**:
1. 改进更新检测逻辑（降低相似度阈值）
2. 优化相似度阈值
3. 检索时过滤 superseded plots

**预期提升**: 从 80% → **90%+**

---

## 3. 优先级排序的优化方向

### 综合评分（考虑 ROI、影响、难度）

| 优先级 | 类型 | 综合得分 | 预期提升 | 工作量 | 关键改进 |
|--------|------|---------|---------|--------|---------|
| 🥇 | **single-session-assistant** | 5.08 | +1.34% | 1天 | 优先检索 assistant turns；恢复上下文长度 |
| 🥈 | **single-session-user** | 4.90 | +1.12% | 1天 | 优化检索；增加上下文长度 |
| 🥉 | **temporal-reasoning** | 4.84 | +5.32% | 3天 | 改进时间过滤；增强时序推理 prompt |
| 4 | **multi-session** | 4.78 | +7.18% | 5天 | 修复代码错误；启用 benchmark_mode |
| 5 | single-session-preference | 2.97 | +1.62% | 2天 | 改进偏好识别；增强偏好存储 |
| 6 | knowledge-update | 2.94 | +1.56% | 2天 | 改进更新检测；优化相似度阈值 |

---

## 4. 分阶段实施建议

### 阶段1：快速修复（1-2天）⚡

**目标**: 快速提升 2.5 个百分点

| 类型 | 预期提升 | 工作量 | 关键改进 |
|------|---------|--------|---------|
| single-session-assistant | +1.34% | 1天 | 优先检索 assistant turns；恢复上下文长度 |
| single-session-user | +1.12% | 1天 | 优化检索；增加上下文长度 |

**累计提升**: +2.46% → **68.6%**

### 阶段2：中等改进（3-5天）📈

**目标**: 提升 8.5 个百分点

| 类型 | 预期提升 | 工作量 | 关键改进 |
|------|---------|--------|---------|
| temporal-reasoning | +5.32% | 3天 | 改进时间过滤；增强时序推理 prompt |
| single-session-preference | +1.62% | 2天 | 改进偏好识别；增强偏好存储 |
| knowledge-update | +1.56% | 2天 | 改进更新检测；优化相似度阈值 |

**累计提升**: +8.50% → **77.1%**

### 阶段3：长期优化（5+天）🚀

**目标**: 提升 7.2 个百分点

| 类型 | 预期提升 | 工作量 | 关键改进 |
|------|---------|--------|---------|
| multi-session | +7.18% | 5天 | 修复代码错误；启用 benchmark_mode；增加 k 值；改进聚合 prompt；实现会话聚合 |

**累计提升**: +7.18% → **84.3%**

### 总潜在提升

**总潜在提升**: +18.15%  
**预期最终准确率**: **84.2%**  
**距离目标 90%**: 还需 **5.8%** 的提升

**建议**: 
- 如果阶段1-3实施后达到 84%，可以考虑：
  1. 进一步优化 multi-session（从 72% → 80%+），可再提升 +1.1%
  2. 进一步优化 temporal-reasoning（从 80% → 85%+），可再提升 +0.7%
  3. 微调其他类型，争取每个类型再提升 1-2%

---

## 5. 关键改进点详细说明

### 5.1 Multi-Session 改进（最高优先级）

**立即修复**（1小时）:
```python
# run_longmemeval_baseline.py:110
def evaluate_answer(expected: str, generated: str, context: str) -> bool:
    expected_str = str(expected).strip()  # ✅ 修复类型转换
    expected_lower = expected_str.lower()
    # ... 其余代码
```

**短期改进**（1天）:
```python
# 启用 benchmark_mode
memory = AuroraMemory(
    cfg=config, 
    seed=42, 
    embedder=embedder,
    benchmark_mode=True  # ✅ 强制存储所有 plots
)

# Multi-session 查询使用更大的 k
if question_type == 'multi-session':
    k = 25  # ✅ 增加检索范围
else:
    k = 10
```

**中期改进**（3-5天）:
- 改进聚合 prompt，明确要求跨会话聚合
- 实现会话标记和 Story 聚合机制

### 5.2 Temporal-Reasoning 改进

**核心改进**（3天）:
1. **时间锚点检测**：识别"最近"、"最早"、"之后"等
2. **时间范围预过滤**：在检索前缩小时间搜索空间
3. **增强时序推理 prompt**：明确要求按时间顺序推理

### 5.3 Single-Session-Assistant 改进

**核心改进**（1天）:
```python
# 检测是否为 assistant 问题
is_assistant_question = any(keyword in question.lower() 
    for keyword in ["remind me what you", "what did you say", 
                    "you told me", "you mentioned"])

if is_assistant_question:
    # 优先检索 assistant turns
    # 方法1: 在 post-processing 中过滤，优先选择 assistant turns
    # 方法2: 在检索时增加 assistant turn 的权重
```

---

## 6. 验证方法

### 6.1 单元测试

为每个改进点创建单元测试：
- Multi-session 聚合测试
- Temporal-reasoning 时间推理测试
- Assistant 问题检索测试

### 6.2 回归测试

在完整测试集上验证改进效果：
```bash
python run_longmemeval_baseline.py
```

重点关注：
- Multi-session 准确率变化
- Temporal-reasoning 准确率变化
- 总体准确率变化

---

## 7. 风险与注意事项

### 7.1 风险

1. **Multi-session 改进可能影响其他类型**：增加 k 值可能增加噪声
2. **Benchmark_mode 可能降低 assistant 问题准确率**：需要平衡
3. **时间过滤可能过度过滤**：需要调优阈值

### 7.2 注意事项

1. **逐步实施**：每个阶段完成后验证效果，再继续下一阶段
2. **保留基线**：记录每个改进点的单独效果
3. **监控性能**：改进不应显著增加延迟或内存占用

---

## 8. 总结

**关键路径**:
1. **立即修复** multi-session 代码错误（+1.65%）
2. **快速改进** single-session-assistant/user（+2.5%）
3. **重点优化** temporal-reasoning（+5.3%）
4. **深度改进** multi-session（+7.2%）

**预期时间线**:
- **第1周**: 阶段1 + 阶段2（快速修复 + 中等改进）→ 77%
- **第2周**: 阶段3（长期优化）→ 84%
- **第3周**: 微调和优化 → 90%

**关键成功因素**:
- Multi-session 和 Temporal-reasoning 是最大瓶颈，必须优先解决
- 快速修复虽然影响小，但可以快速建立信心
- 需要平衡不同改进点，避免相互影响
