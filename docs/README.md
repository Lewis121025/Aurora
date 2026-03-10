# 文档索引

Aurora 现已收敛到单一路径的 `graph_first` 架构。为了保留设计历史，部分旧文档仍保存在原位置，但它们已经归档，不再作为当前实现或运行方式的依据。

## 当前有效文档

- [README.md](/Users/lewis/Aurora/README.md)
- [ADR 002: Graph-First Emergence](/Users/lewis/Aurora/docs/adr/002-graph-first-emergence.md)
- [基准测试指南](/Users/lewis/Aurora/docs/research/benchmark_guide.md)

当前 canonical 实现请直接以 [aurora/soul](/Users/lewis/Aurora/aurora/soul) 和 [aurora/runtime](/Users/lewis/Aurora/aurora/runtime) 为准。

## 已归档文档

以下内容仅保留作历史参考，不应再被视为现行方案：

- [配置常量模块化重组 - 完成报告](/Users/lewis/Aurora/docs/migrations/config_complete.md)
- [代码质量提升进度报告](/Users/lewis/Aurora/docs/quality/progress.md)
- [AURORA 代码质量提升路线图](/Users/lewis/Aurora/docs/quality/roadmap.md)
- [AURORA 叙事记忆算法（第一性原理版）参考实现](/Users/lewis/Aurora/docs/research/AURORA_memory_algorithm.md)
- [AURORA 叙事记忆：生产级拼图补全包](/Users/lewis/Aurora/docs/research/AURORA_production_pack.md)
- [AURORA LongMemEval Baseline Report](/Users/lewis/Aurora/docs/research/longmemeval_baseline_report.md)
- [LongMemEval SOTA 差距分析与夺冠策略](/Users/lewis/Aurora/docs/research/longmemeval_sota_analysis.md)
- [叙事涌现记忆系统（Narrative Emergent Memory System）](/Users/lewis/Aurora/docs/research/narrative_memory_architecture.md)
- [Lessons Learned 索引与各分篇](/Users/lewis/Aurora/docs/research/lessons_learned/README.md)

## 归档规则

- 旧文档默认保留在原路径，避免破坏历史引用。
- 归档文档会在文首明确标注“归档说明”。
- 如果文档描述了已删除的兼容层、双架构切换、迁移阶段脚手架，视为归档内容。
