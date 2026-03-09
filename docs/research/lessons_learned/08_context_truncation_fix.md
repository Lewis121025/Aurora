# 08 上下文截断问题修复

> [!WARNING]
> 本文档在 Aurora Soul canonical migration 之后已过时。当前生产主线请以 `aurora.soul.AuroraSoul` 和 `aurora.runtime` 为准。

## 问题发现日期
2026-02-02

## 问题描述

LongMemEval 测试中，即使检索到了包含正确答案的内容，LLM 仍然回答 "no information available"。

## 根本原因分析

### 1. 数据格式解析错误

**问题**：测试代码使用了 `haystack` 字段，但 LongMemEval-S 数据使用的是 `haystack_sessions`。

**数据格式**：
```python
# 正确的格式
{
    "haystack_sessions": [  # list of sessions
        [  # each session is a list of messages
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        ...
    ],
    "haystack_dates": ["2023/05/20 (Sat) 02:21", ...]
}
```

**解决方案**：实现正确的 `flatten_sessions()` 函数。

### 2. 向量维度不匹配

**问题**：Bailian Embedding 返回 1024 维向量，但默认配置是 384 维。

**错误信息**：
```
ValueError: vector dim mismatch: (1024,) vs (384,)
```

**解决方案**：创建正确的配置：
```python
cfg = SoulConfig(dim=1024)
memory = AuroraSoul(cfg=cfg, embedder=embedder, ...)
```

### 3. 上下文截断丢失关键信息（核心问题）

**问题**：`build_qa_prompt()` 使用简单的前 3500 字符截断，导致关键信息被丢弃。

**诊断数据**：
- 问题："What degree did I graduate with?"
- 目标 chunk 长度：17929 字符
- 关键句："I graduated with a degree in Business Administration"
- 检索排名：第 2 位（相似度 0.2126，排名正确）
- 原始截断限制：3500 字符
- 结果：关键信息不在 prompt 中

**解决方案**：实现智能上下文过滤 `_extract_relevant_context()`：
1. 将上下文分割成 chunks
2. 根据问题关键词对每个 chunk 评分
3. 如果高分 chunk 太大，提取包含关键词的行及其上下文
4. 按分数排序选择 chunks，直到达到限制
5. 保持原始顺序以维持连贯性

**关键代码改动**：
```python
# 之前
max_context_length: int = 3500
truncated_context = context[:max_context_length]

# 之后
max_context_length: int = 12000
filtered_context = _extract_relevant_context(
    context, question, max_context_length
)
```

## 修复后效果

单样本测试：
- 问题："What degree did I graduate with?"
- 期望："Business Administration"
- 修复前 LLM 回答："The user did not provide information about the degree..."
- 修复后 LLM 回答："Business Administration."
- 结果：**正确**

## 经验教训

1. **数据格式验证**：始终检查数据格式，不要假设字段名称
2. **配置一致性**：embedding 维度必须与配置匹配
3. **上下文处理**：
   - 简单截断会丢失关键信息
   - 需要基于问题关键词的智能过滤
   - 大型 chunks 需要特殊处理（提取相关行）
4. **诊断优先**：
   - 先确认检索是否正确（排名、相似度）
   - 再确认 LLM 输入是否正确（prompt 内容）
   - 最后确认 LLM 输出

## 相关文件

- `aurora/integrations/llm/Prompt/qa_prompt.py`：添加 `_extract_relevant_context()` 函数
- `SoulConfig`：需要传入正确的 `dim` 参数
- 测试脚本：需要正确解析 `haystack_sessions`
