# 第一性原理分析方法论

## 问题背景

在优化 AURORA 系统时，最初采用了"补丁式"方案：
- 针对 knowledge-update 问题，过滤 superseded plots
- 针对 temporal-reasoning 问题，保留 superseded plots
- 这导致了逻辑矛盾和代码复杂性

## 错误方法

```python
# ❌ 补丁式方案 - 根据查询类型决定是否过滤
def query(self, text, k=10):
    trace = self._retrieve(text, k)
    
    if self._is_knowledge_update_query(text):
        trace.ranked = self._filter_superseded(trace.ranked)
    elif self._is_temporal_query(text):
        trace.ranked = self._keep_all(trace.ranked)
    # 问题：需要越来越多的 if-else 分支
```

## 第一性原理分析

**问题本质**：superseded 的语义是什么？

从叙事心理学角度：
- 人类记忆不会"删除"过去的事实
- "我住在北京" 被 supersede 后，变成 "我**曾经**住在北京"
- 事实本身仍然为真，只是时态变了

**核心洞察**：superseded ≠ deleted，而是 temporal repositioning

## 正确方案

```python
# ✅ 第一性原理方案 - 让检索层返回完整信息，让语义层做决策
def query(self, text, k=10):
    # 1. 检索所有相关信息（包括 superseded）
    all_relevant = self._retrieve(text, k * 2)
    
    # 2. 组织成时间线
    timelines = self._group_into_timelines(all_relevant)
    
    # 3. 返回带时态标记的结果
    return TimelineAwareTrace(
        ranked=self._filter_for_ranking(all_relevant),
        timeline_group=timelines,
        include_historical=True
    )
```

## 输出格式

```
[KNOWLEDGE EVOLUTION: User residence]
[HISTORICAL - superseded] User: I live in Beijing
  → Updated to: User: I moved to Shanghai
[CURRENT] User: I moved to Shanghai
```

## 教训

1. **不要针对症状打补丁** - 找到问题的根本原因
2. **保持架构简洁** - 一个原则解决多个问题
3. **让正确的层做正确的事** - 检索层提供信息，语义层做决策
4. **回归领域原理** - 叙事心理学给出了答案

## 影响

这个方法论应用后：
- knowledge-update 和 temporal-reasoning 不再矛盾
- 代码复杂度降低
- 系统更容易理解和维护
