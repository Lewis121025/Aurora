# Superseded 语义设计问题

> [!WARNING]
> 归档说明：本文档为历史复盘材料，仅保留作参考，不再代表当前生产实现。当前 canonical 入口请以 `README.md`、`docs/README.md`、`docs/adr/002-graph-first-emergence.md`、`docs/research/benchmark_guide.md` 和 `aurora.soul` / `aurora.runtime` 代码为准。

## 问题背景

实现知识更新追踪时，将 superseded 状态理解为"无效/过时"，
导致检索时过滤掉了时序查询需要的历史信息。

## 错误理解

```python
# ❌ 错误：把 superseded 当作垃圾
def _filter_active_results(self, ranked):
    return [r for r in ranked if r.status != "superseded"]
```

这导致：
- knowledge-update 查询正常 ✓
- temporal-reasoning 查询失败 ✗ (历史信息被过滤)

## 第一性原理分析

**从叙事心理学角度**：

人类记忆如何处理"我换了住址"？
1. 不是删除"我住在北京"
2. 而是将其重新定位为"我**曾经**住在北京"
3. 两个信息都是真实的，只是时态不同

**核心洞察**：
```
superseded ≠ invalid
superseded = temporal repositioning (时态重新定位)
```

## 正确设计

**superseded 的正确语义**：
- 不是"错误的"或"无效的"
- 而是"已成为过去时态"
- 仍然是真实信息，只是不再是"当前状态"

```python
# ✅ 正确：superseded 是时态标记，不是有效性标记
@dataclass
class Plot:
    status: Literal["active", "superseded", "corrected"]
    # active: 当前有效
    # superseded: 已成为历史（但仍然真实）
    # corrected: 被纠正（原信息是错误的）
```

## 解决方案

**Timeline-Aware Retrieval**：

```python
def query(self, text, k=10):
    # 1. 检索所有相关信息（不过滤 superseded）
    all_relevant = self._semantic_search(text, k * 2)
    
    # 2. 组织成时间线
    timelines = self._group_into_timelines(all_relevant)
    
    # 3. 返回结果时区分当前和历史
    return RetrievalTrace(
        ranked=ranked_results,        # 用于排序的活跃结果
        timeline_group=timelines,     # 完整的时间线信息
        include_historical=True       # 标记包含历史信息
    )
```

**输出格式**：

```
[KNOWLEDGE EVOLUTION: User residence]
[HISTORICAL - superseded] User: I live in Beijing
  → Updated to: User: I moved to Shanghai
[CURRENT] User: I moved to Shanghai
```

## 效果

这个设计统一解决了两类问题：

| 查询类型 | 需要的信息 | 设计如何满足 |
|---------|-----------|-------------|
| knowledge-update | 最新状态 | ranked 返回 active 优先 |
| temporal-reasoning | 历史轨迹 | timeline_group 包含完整历史 |

## 教训

1. **语义设计要精确** - "superseded" 不等于 "deleted"
2. **一个设计解决多个问题** - 不需要 if-else 分支
3. **回归领域知识** - 叙事心理学给出了正确答案
4. **保持信息完整性** - 过滤是消费者的责任，不是存储的责任
