# 时间建模问题

> [!WARNING]
> 归档说明：本文档为历史复盘材料，仅保留作参考，不再代表当前生产实现。当前 canonical 入口请以 `README.md`、`docs/README.md`、`docs/adr/002-graph-first-emergence.md`、`docs/research/benchmark_guide.md` 和 `aurora.soul` / `aurora.runtime` 代码为准。

## 问题背景

temporal-reasoning 准确率只有 40%，是主要瓶颈之一。

## 症状

| 查询类型 | 示例 | 问题 |
|---------|------|------|
| "最早" | "What did I learn first?" | 无法确定时间顺序 |
| "最近" | "What's my latest hobby?" | 返回了旧信息 |
| "之后" | "What happened after X?" | 无法锚定参考点 |

## 根本原因

**当前实现的问题**：

1. **时间是后处理** - 先语义检索，后时间排序
2. **缺少时间预过滤** - 无法缩小搜索空间
3. **时间锚点解析不完整** - "after the first service" 无法解析

```python
# ❌ 当前：时间只是后处理
def retrieve(self, query, k):
    results = self._semantic_search(query, k)
    
    if self._is_temporal_query(query):
        results.sort(key=lambda x: x.timestamp)  # 太晚了！
    
    return results
```

## 第一性原理分析

**在叙事心理学中，时间的三重角色**：

| 角色 | 说明 | AURORA 对应 |
|------|------|------------|
| 时间点 | 事件发生的时刻 | Plot.ts |
| 时间线 | 事件的序列 | Story.plot_ids |
| 时间不变量 | 超越时间的教训 | Theme |

**核心洞察**：时间不是可选的元数据，而是叙事结构的本质维度。

## 解决方案

**1. 时间锚点检测**：

```python
class TimeAnchor(Enum):
    RECENT = "recent"      # 最近/上次
    EARLIEST = "earliest"  # 最早/第一次
    SPAN = "span"          # 一段时间
    NONE = "none"          # 无特定锚点

def _detect_time_anchor(self, query: str) -> TimeAnchor:
    if any(w in query for w in ["最近", "上次", "recently", "last"]):
        return TimeAnchor.RECENT
    if any(w in query for w in ["最早", "第一次", "first", "initially"]):
        return TimeAnchor.EARLIEST
    # ...
```

**2. 时间范围预过滤**：

```python
class TimeRangeExtractor:
    def extract(self, query: str, events: List[Tuple[str, float]]) -> TimeRange:
        """从查询推断时间范围"""
        anchor = self._detect_anchor(query)
        
        if anchor == TimeAnchor.EARLIEST:
            earliest_ts = min(ts for _, ts in events)
            return TimeRange(end=earliest_ts + 86400)
        
        # ...

def retrieve_hybrid(self, query, k):
    # 1. 检测时间范围
    time_range = self.time_extractor.extract(query, self._get_events())
    
    # 2. 预过滤候选集
    candidates = self._filter_by_time(all_plots, time_range)
    
    # 3. 在过滤后的集合上检索
    results = self._semantic_search(query, candidates, k)
    
    return results
```

**3. 时序感知重排序**：

```python
def _temporal_aware_rerank(self, ranked, time_anchor, k):
    if time_anchor == TimeAnchor.RECENT:
        ranked.sort(key=lambda x: -x.timestamp)  # 最新优先
    elif time_anchor == TimeAnchor.EARLIEST:
        ranked.sort(key=lambda x: x.timestamp)   # 最早优先
    elif time_anchor == TimeAnchor.SPAN:
        ranked = self._select_temporal_diversity(ranked, k)  # 时间多样性
    
    return ranked[:k]
```

## 预期效果

| 优化 | 预期提升 |
|------|---------|
| 时间锚点检测 | +3-5% |
| 时间范围预过滤 | +7-11% |
| 时序感知重排序 | +2-3% |
| **总计** | +12-19% |

## 教训

1. **时间是核心维度** - 不是可选的元数据
2. **预过滤比后排序更有效** - 缩小搜索空间
3. **理解查询意图** - "first" 和 "latest" 需要不同处理
4. **多信号融合** - 语义 + 时间 + 结构
