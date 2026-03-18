# Aurora MCP vNext Draft

## Position

Aurora 不应该退化成一组松散的 MCP tools。

正确形态是：

- `AuroraKernel` 继续作为唯一核心内核
- MCP 作为给其他 agent 使用的薄适配层
- MCP server 不持有额外业务状态，不绕开 memory field 语义
- 对外描述统一使用 `memory field` 语义，不再把激活图说成当前真值集合

MCP vNext 的目标不是暴露所有内部实现，而是把 Aurora 最稳定、最可复用的 memory field write / read 闭环收口成少量标准接口。

## Goals

- 让别的 agent 能把 Aurora 当作 subject-scoped memory field service 使用
- 暴露最小但完整的自然语言写入与 field 读取闭环
- 保持 human-memory kernel 边界

## Non-Goals

- 不提供任意 field mutation tool
- 不提供跨 subject 的全局检索
- 不提供 compiler / evolution 内核控制面
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
  "created_atom_ids": ["atom_xxx"],
  "created_edge_ids": ["edge_xxx"]
}
```

### `aurora_recall`

用途：按 subject 查询当前 memory field 上的局部切片。

输入：

```json
{
  "subject_id": "string",
  "query": "string",
  "limit": 8
}
```

输出：

```json
{
  "subject_id": "subject-alice",
  "query": "string",
  "summary": "[QUERY_MEMORY_FIELD] ...",
  "atoms": [
    {
      "atom_id": "atom_xxx",
      "atom_kind": "memory",
      "text": "我现在住在杭州",
      "activation": 0.93,
      "confidence": 0.92,
      "salience": 0.88,
      "created_at": 1.0
    }
  ],
  "edges": [
    {
      "source_atom_id": "atom_xxx",
      "target_atom_id": "atom_yyy",
      "influence": 0.38,
      "confidence": 0.75
    }
  ]
}
```

## Resources

### `aurora://subject/{subject_id}/memory-field`

返回当前主体的 memory field 视图：

- `summary`
- `atoms`
- `edges`

这里只暴露当前高激活 memory field 切片，不暴露任意底层 field mutation 能力。

## What Must Not Be Exposed

以下内容不进入 MCP vNext：

- 任意 atom / edge 写接口
- 内部 distillation helper
- 手动 lifecycle 改写接口
- 跨 subject 的批量浏览接口

原因很简单：

- 这些接口会把 Aurora 从有语义边界的 memory kernel 拉回任意存取层
- 一旦 host agent 可以直接改 field 真相，`turn()` 就不再是唯一受约束的长期记忆入口

## Transport Choice

vNext 优先使用 `stdio`。

原因：

- Aurora 的第一目标是“给别的 agent 用”
- 本地 agent 集成优先于远程部署
- 远程鉴权、多租户和 streamable transport 可以留到后续版本
