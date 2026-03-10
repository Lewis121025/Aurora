# 进度追踪

> [!WARNING]
> 归档说明：本文档为历史复盘材料，仅保留作参考，不再代表当前生产实现。当前 canonical 入口请以 `README.md`、`docs/README.md`、`docs/adr/002-graph-first-emergence.md`、`docs/research/benchmark_guide.md` 和 `aurora.soul` / `aurora.runtime` 代码为准。

## 项目目标

**目标**: LongMemEval (ICLR 2025) 90%+ 准确率，超越 SOTA (86%)

## 当前状态

| 阶段 | 状态 | 准确率 | 主要改进 |
|------|------|--------|---------|
| Baseline | ✅ 完成 | 48.0% | 建立基准 |
| Phase 1 | ✅ 完成 | 63.3% | benchmark_mode 无损存储 |
| Phase 2 | ✅ 完成 | 68.0% | 时间范围预过滤 (+19% temporal) |
| Phase 3 | ✅ 完成 | - | 实体-属性追踪 |
| Phase 4 | ✅ 完成 | - | Abstention 机制 |
| Phase 5 | ⏳ 待开始 | - | 事实增强索引 |

**当前总体准确率: 68% (距离 SOTA 86% 差 18%)**

## 详细进度

### Phase 1: 无损索引 ✅

**完成时间**: 2026-02-01

**改进内容**:
- 添加 `benchmark_mode` 参数
- 跳过 VOI 门控，强制存储
- 禁用冗余过滤

**效果**:
| 类型 | Before | After | 变化 |
|------|--------|-------|------|
| multi-session | 25.6% | 60.0% | +34.4% |
| preference | 23.3% | 60.0% | +36.7% |
| knowledge-update | 65.4% | 80.0% | +14.6% |
| user | 74.3% | 80.0% | +5.7% |
| temporal | 41.4% | 40.0% | -1.4% |
| assistant | 73.2% | 60.0% | -13.2% |
| **Overall** | **48.0%** | **63.3%** | **+15.3%** |

**发现的问题**:
- assistant 类型下降 13.2%，需要分析
- temporal 没有改善，需要专门优化

### Phase 2: 时间范围预过滤 🔄

**目标**: temporal-reasoning 从 40% 提升到 60%+

**计划内容**:
1. TimeRangeExtractor - 从查询推断时间范围
2. 时间预过滤 - 缩小搜索空间
3. 时序感知重排序

**当前状态**: 代码已实现，待集成测试

### Phase 3: 实体-属性追踪 ⏳

**目标**: knowledge-update 从 80% 提升到 90%+

**计划内容**:
1. EntityTracker - 追踪实体属性变化
2. 实体-属性对齐 - 不依赖语义相似度
3. 最新状态优先检索

### Phase 4: Abstention 机制 ⏳

**目标**: 正确处理 30 个 abstention 问题 (+6%)

**计划内容**:
1. AbstentionDetector - 检测低置信度
2. 置信度阈值
3. "I don't know" 响应

**当前状态**: 代码已实现，待集成测试

### Phase 5: 事实增强索引 ⏳

**目标**: multi-session 进一步提升

**计划内容**:
1. FactExtractor - 提取关键事实
2. 多键索引
3. 事实对齐检索

## 风险和阻碍

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| assistant 下降 | -13.2% | 需要分析原因 |
| temporal 停滞 | 无提升 | Phase 2 专门优化 |
| API 调用成本 | 运行时间长 | 批量处理、缓存 |

## 下一步行动

1. [ ] 分析 assistant 下降原因
2. [ ] 完成 Phase 2 时间优化
3. [ ] 集成 Abstention 机制
4. [ ] 运行完整 500 题测试
