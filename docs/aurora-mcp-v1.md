# Aurora MCP vNext Draft

## Position

Aurora 不应该退化成一组松散的 MCP tools。

正确形态是：

- `AuroraKernel` 继续作为唯一核心内核
- MCP 作为给其他 agent 使用的薄适配层
- MCP server 不持有额外业务状态，不绕开内核语义

MCP vNext 的目标不是暴露所有内部实现，而是把 Aurora 最稳定、最可复用的读写闭环收口成少量标准接口。

## Goals

- 让别的 agent 能把 Aurora 当作 subject-scoped memory service 使用
- 暴露最小但完整的读写闭环
- 保持 human-memory kernel 边界，不回退到 relation-first
- 默认暴露当前可见的 `SubjectMemoryState`

## Non-Goals

- 不提供任意 trace 写接口
- 不提供跨 subject 的全局检索
- 不提供 compiler / reducer 控制面
- 不为了 MCP 再造一套独立 memory 模型

## Tools

### `aurora_turn`

用途：执行一轮 subject-scoped turn。

输入：

```json
{
  "subject_id": "string",
  "text": "string",
  "now_ts": 0.0
}
```

输出：

```json
{
  "turn_id": "atom_evidence_xxx",
  "subject_id": "subject-alice",
  "response_text": "string",
  "recall_used": true,
  "applied_atom_ids": ["atom_xxx"]
}
```

### `aurora_recall`

用途：按 subject 查询 scope-aware recall，只返回 hits。

输入：

```json
{
  "subject_id": "string",
  "query": "string",
  "limit": 5,
  "mode": "blended",
  "temporal_scope": "current"
}
```

输出：

```json
{
  "subject_id": "subject-alice",
  "query": "string",
  "mode": "blended",
  "temporal_scope": "current",
  "hits": [
    {
      "memory_kind": "semantic",
      "content": "我现在住在杭州",
      "score": 0.82,
      "why_recalled": "lexical+vector+current"
    }
  ]
}
```

## Resources

### `aurora://subject/{subject_id}/state`

返回当前可见的 `SubjectMemoryState`。这里只暴露公开状态视图，不返回 evidence archive，也不返回内部 atom 总表。

`state` 的返回形状只包含这些 projection views：

- `semantic_self_model`
- `semantic_world_model`
- `procedural_memory`
- `active_cognition`
- `affective_state`
- `narrative_state`
- `recent_episodes`

## What Must Not Be Exposed

以下内容不进入 MCP vNext：

- 任意 atom mutation tool
- 内部 distillation helper
- 生命周期强制改写接口
- 跨 subject 的批量浏览接口

原因很简单：

- 这些接口会把 Aurora 从有语义边界的 memory kernel 拉回任意存取层
- 一旦 host agent 可以直接写 atom，`turn()` 就不再是唯一受约束的长期记忆入口

## Transport Choice

vNext 优先使用 `stdio`。

原因：

- Aurora 的第一目标是“给别的 agent 用”
- 本地 agent 集成优先于远程部署
- 远程鉴权、多租户和 streamable transport 可以留到后续版本
