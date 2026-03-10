# AURORA LongMemEval 夺冠项目 - 经验教训记录

> [!WARNING]
> 归档说明：本文档及本目录下分篇均为历史复盘材料，仅保留作参考，不再代表当前生产实现。当前 canonical 入口请以 `README.md`、`docs/README.md`、`docs/adr/002-graph-first-emergence.md`、`docs/research/benchmark_guide.md` 和 `aurora.soul` / `aurora.runtime` 代码为准。

> 记录项目过程中遇到的关键问题、根本原因分析和解决方案

## 目录结构

```
lessons_learned/
├── README.md                    # 本文件 - 索引和总览
├── 01_first_principles.md       # 第一性原理分析方法论
├── 02_benchmark_mismatch.md     # 基准测试不匹配问题
├── 03_voi_gate_problem.md       # VOI 门控导致信息丢失
├── 04_superseded_semantics.md   # superseded 语义设计
├── 05_temporal_modeling.md      # 时间建模问题
└── 06_progress_tracker.md       # 进度追踪
```

## 项目目标

**目标**: 在 LongMemEval (ICLR 2025) 上达到 90%+ 准确率，超越 SOTA (86%)

**当前进度**: 
- Baseline: 48%
- Phase 1 后: 63.3% (+15.3%)
- 目标: 90%+

## 关键里程碑

| 日期 | 里程碑 | 准确率 | 关键改进 |
|------|--------|--------|---------|
| 2026-02-01 | Baseline 建立 | 48.0% | 完整 500 题评估 |
| 2026-02-01 | Phase 1 完成 | 63.3% | benchmark_mode 无损存储 |
| - | Phase 2 | - | 时间范围预过滤 |
| - | Phase 3 | - | 实体-属性追踪 |
| - | Phase 4 | - | Abstention 机制 |
| - | 最终目标 | 90%+ | - |

## 核心教训总结

1. **第一性原理优于补丁方案** - 不要针对症状打补丁，要找到根本原因
2. **基准测试选择至关重要** - AURORA 的叙事记忆设计不适合 RAG 类基准
3. **无损存储是前提** - Benchmark 场景不能丢弃任何信息
4. **时间是叙事的骨架** - 时间不是元数据，而是核心维度
