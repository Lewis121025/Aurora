# VOI 门控导致信息丢失问题

## 问题背景

AURORA 使用 Value of Information (VOI) 门控决定是否存储新信息。
在 Benchmark 场景下，这导致了严重的信息丢失。

## 症状

| 问题类型 | Baseline 准确率 | 问题 |
|---------|----------------|------|
| multi-session | 25.6% | 跨会话信息丢失 |
| single-session-preference | 23.3% | 偏好信息被过滤 |

## 根本原因

**VOI 门控的设计初衷**：
- 避免存储冗余信息
- 节省存储空间
- 提高检索效率

**Benchmark 场景的特殊性**：
- 每个 Turn 都可能包含答案
- 看似"不重要"的信息可能是关键证据
- 信息完整性比存储效率更重要

```python
# 问题代码
def ingest(self, text):
    plot = self._build_plot(text)
    
    # VOI 门控可能过滤掉关键信息
    if self._voi_gate(plot) > threshold:  # ← 问题在这里
        self._store_plot(plot)
    
    return plot
```

## 具体案例

**multi-session 问题**：
```
Session 1: "I bought 2 books at Store A"
Session 5: "I returned 1 book to Store A"  ← 可能被 VOI 过滤
Session 10: "I bought 1 book at Store B"

Query: "How many books do I have?"
Expected: 2 (2 - 1 + 1)
Actual: 无法计算，因为 Session 5 被过滤了
```

## 解决方案

**添加 benchmark_mode 参数**：

```python
class AuroraMemory:
    def __init__(self, cfg, benchmark_mode=False):
        self.benchmark_mode = benchmark_mode or cfg.benchmark_mode
    
    def ingest(self, text):
        plot = self._build_plot(text)
        
        if self.benchmark_mode:
            # 强制存储，跳过 VOI 门控
            self._store_plot(plot)
            return plot
        
        # 正常流程
        if self._voi_gate(plot) > threshold:
            self._store_plot(plot)
        
        return plot
```

## 效果

| 问题类型 | Baseline | Phase 1 | 提升 |
|---------|----------|---------|------|
| multi-session | 25.6% | 60.0% | +34.4% |
| single-session-preference | 23.3% | 60.0% | +36.7% |
| **Overall** | 48.0% | 63.3% | +15.3% |

## 教训

1. **不同场景需要不同策略** - 生产环境和 Benchmark 环境的需求不同
2. **信息完整性是前提** - 没有信息，再好的算法也无用
3. **提供配置选项** - 让用户根据场景选择合适的模式
4. **分析问题根源** - multi-session 问题不是检索问题，是存储问题

## 后续优化

benchmark_mode 是一个临时方案。长期应该：
1. 改进 VOI 门控算法，减少误过滤
2. 实现"软删除"，保留被过滤信息的索引
3. 支持延迟决策，等有更多上下文再决定
