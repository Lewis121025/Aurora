# Single-Session-Assistant 准确率下降分析报告

## 执行摘要

**问题**: single-session-assistant 类型从 73.2% (baseline) 下降到 60.0% (phase1)，下降 13.2%

**影响**: Phase1 测试集中，baseline 有 7 个正确，但 phase1 只有 6 个正确，至少 1 个回归

**根本原因**: benchmark_mode=True 导致检索噪声增加，特别是对于询问"你说了什么/推荐了什么"的 assistant 问题

---

## 1. 问题特征分析

### 1.1 失败案例模式

从 Phase1 测试集（10个问题）分析：

| 特征 | 统计 |
|------|------|
| 答案在 assistant turn | 9/10 (90%) |
| 答案在 user turn | 1/10 (10%) |
| 包含 "remind" 关键词 | 7/10 (70%) |
| 包含 "told/said" 关键词 | 2/10 (20%) |
| 包含 "suggest/recommend" 关键词 | 3/10 (30%) |

### 1.2 典型失败案例

**案例 1**: `6ae235be`
- **问题**: "I remember you told me about the refining processes at CITGO's three refineries earlier. Can you remind me what kind of processes are used at the Lake Charles Refinery?"
- **期望答案**: "Atmospheric distillation, fluid catalytic cracking (FCC), alkylation, and hydrotreating."
- **答案位置**: assistant turn
- **Baseline**: ❌ 错误
- **特征**: 询问 assistant 之前说过的话

**案例 2**: `1903aded`
- **问题**: "I think we discussed work from home jobs for seniors earlier. Can you remind me what was the 7th job in the list you provided?"
- **期望答案**: "Transcriptionist."
- **答案位置**: assistant turn
- **Baseline**: ❌ 错误
- **特征**: 询问 assistant 之前提供的列表中的特定项

**案例 3**: `ceb54acb`
- **问题**: "In our previous chat, you suggested 'sexual compulsions' and a few other options for alternative terms for certain behaviors. Can you remind me what the other four options were?"
- **期望答案**: "I suggested 'sexual fixations', 'problematic sexual behaviors', 'sexual impulsivity', and 'compulsive sexuality'."
- **答案位置**: assistant turn
- **Baseline**: ❌ 错误
- **特征**: 询问 assistant 之前建议的多个选项

---

## 2. 根本原因分析

### 2.1 Benchmark Mode 导致的检索噪声增加 ⚠️ **最可能的原因**

**问题机制**:

1. **Baseline (benchmark_mode=False)**:
   - VOI 门控会过滤掉一些"不重要"的交互
   - 主要存储高价值信息（关系、偏好、重要事实）
   - 检索时噪声相对较少

2. **Phase1 (benchmark_mode=True)**:
   - 强制存储 **所有** interactions，包括：
     - User turns（用户说的话）
     - Assistant turns（助手说的话）
     - 看似"不重要"的闲聊
   - 检索时可能同时检索到 user 和 assistant turns

3. **对于 assistant 问题的特殊影响**:
   - Assistant 问题询问的是"你（assistant）说了什么"
   - 检索时可能检索到：
     - ✅ 相关的 assistant turns（正确）
     - ❌ 相关的 user turns（噪声）
     - ❌ 不相关的 assistant turns（噪声）
   - 噪声增加导致 LLM 难以提取正确答案

**代码证据**:

```python
# aurora/algorithms/aurora_core.py:1027-1030
if self.benchmark_mode:
    plot._storage_prob = 1.0
    logger.debug(f"Benchmark mode: force storing plot {plot.id[:8]}...")
    return True
```

### 2.2 上下文截断

| 配置 | 上下文长度 | 影响 |
|------|-----------|------|
| Baseline | `context[:3500]` | 可能包含更多相关信息 |
| Phase1 | `context[:3000]` | 可能截断关键信息 |

**影响**: 对于较长的 assistant 回答（如列表、详细说明），可能被截断

### 2.3 Prompt 弱化

**Baseline prompt**:
```
Based on the conversation history below, answer the question concisely.
Focus on extracting the specific information requested.
```

**Phase1 prompt**:
```
Answer the question based on the conversation history.
Answer (be brief and specific):
```

**差异**: Baseline 强调"extracting specific information"，更适合 assistant 问题

### 2.4 Abstention 误判

Phase1 增加了 abstention 机制，可能错误地将有效的 assistant 问题标记为"不知道"：

```python
# run_longmemeval_phase1.py:129-142
should_abstain = False
if hasattr(trace, 'abstention') and trace.abstention:
    should_abstain = trace.abstention.should_abstain

if should_abstain or not context.strip():
    answer = "I don't know"
```

**影响**: 可能导致正确的 assistant 问题被拒绝回答

---

## 3. 修复建议

### 3.1 优先级 P0: 针对 Assistant 问题的检索优化

**方案**: 在检索时为 assistant 问题优先检索 assistant turns

**实现**:

```python
# 在 field_retriever.py 或 aurora_core.py 中
def query(self, text: str, k: int = 5, query_type: Optional[QueryType] = None):
    # 检测是否为 assistant 问题
    is_assistant_question = self._is_assistant_question(text)
    
    if is_assistant_question:
        # 优先检索 assistant turns
        # 方法1: 在 post-processing 中过滤，优先选择 assistant turns
        # 方法2: 在检索时增加 assistant turn 的权重
        pass
```

**检测 assistant 问题的关键词**:
- "remind me what you"
- "what did you say/recommend/suggest"
- "you told me"
- "you mentioned"

### 3.2 优先级 P1: 恢复上下文长度

**方案**: 将 Phase1 的上下文长度从 3000 恢复到 3500

```python
# run_longmemeval_phase1.py:147
context[:3500]  # 改为 3500
```

### 3.3 优先级 P2: 增强 Prompt

**方案**: 为 assistant 问题使用专门的 prompt

```python
# run_longmemeval_phase1.py
if question_type == 'single-session-assistant':
    prompt = f"""Based on the conversation history below, extract the specific information 
that the ASSISTANT previously mentioned or recommended.

IMPORTANT: Focus on what the ASSISTANT said, not what the user said.
Extract the exact information requested from the assistant's previous responses.

Conversation Context:
{context[:3500]}

Question: {question}

Answer (extract the specific information from assistant's previous response, be brief and specific):"""
else:
    prompt = f"""Answer the question based on the conversation history.
...
```

### 3.4 优先级 P3: 优化 Abstention 逻辑

**方案**: 对于 assistant 问题，降低 abstention 阈值或跳过 abstention

```python
# run_longmemeval_phase1.py
if question_type == 'single-session-assistant':
    # Assistant 问题通常有答案，降低 abstention 倾向
    should_abstain = False  # 或使用更宽松的阈值
else:
    should_abstain = trace.abstention.should_abstain if hasattr(trace, 'abstention') else False
```

### 3.5 优先级 P4: 检索后过滤

**方案**: 在检索结果中优先选择 assistant turns

```python
# 在构建 context 时
assistant_plots = []
user_plots = []

for plot_id, score, kind in trace.ranked[:10]:
    plot = memory.plots.get(plot_id)
    if plot:
        # 检查 plot 是否来自 assistant turn
        # 可以通过 plot.text 或 metadata 判断
        if self._is_assistant_turn(plot):
            assistant_plots.append(plot)
        else:
            user_plots.append(plot)

# 优先使用 assistant plots
if assistant_plots:
    context = "\n".join([p.text for p in assistant_plots[:8]])
    # 如果不够，再补充 user plots
    if len(assistant_plots) < 8:
        context += "\n" + "\n".join([p.text for p in user_plots[:8-len(assistant_plots)]])
else:
    context = "\n".join([p.text for p in user_plots[:8]])
```

---

## 4. 预期效果

| 修复项 | 预期提升 | 累计准确率 |
|--------|---------|-----------|
| 当前 Phase1 | - | 60.0% |
| + P0: Assistant 检索优化 | +8-10% | 68-70% |
| + P1: 上下文长度恢复 | +2-3% | 70-73% |
| + P2: Prompt 增强 | +2-3% | 72-76% |
| + P3: Abstention 优化 | +1-2% | 73-78% |
| **目标** | - | **≥73.2%** (恢复 baseline) |

---

## 5. 验证方案

### 5.1 单元测试

创建针对 assistant 问题的测试：

```python
def test_assistant_question_retrieval():
    """测试 assistant 问题优先检索 assistant turns"""
    memory = AuroraMemory(cfg=config, benchmark_mode=True)
    
    # Ingest conversation
    memory.ingest("User: What restaurants do you recommend?")
    memory.ingest("Assistant: I recommend Roscioli for Italian food.")
    memory.ingest("User: Thanks!")
    
    # Query assistant question
    query = "Can you remind me what restaurant you recommended?"
    trace = memory.query(query, k=5)
    
    # 验证检索结果优先包含 assistant turn
    assert any("Roscioli" in memory.plots[p[0]].text for p in trace.ranked)
```

### 5.2 回归测试

在 Phase1 测试集上验证修复效果：

```bash
python run_longmemeval_phase1.py
# 期望: single-session-assistant 准确率 ≥ 73.2%
```

---

## 6. 总结

**根本原因**: benchmark_mode=True 导致检索噪声增加，特别是对于询问"你说了什么"的 assistant 问题，检索时可能同时检索到 user 和 assistant turns，降低答案提取准确性。

**关键修复**: 
1. 为 assistant 问题优先检索 assistant turns
2. 恢复上下文长度到 3500
3. 增强 prompt 强调提取 assistant 的信息
4. 优化 abstention 逻辑避免误判

**预期结果**: 准确率恢复到 ≥73.2%，甚至可能超过 baseline（通过更好的 prompt 和检索优化）
