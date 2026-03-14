# Aurora V2 代码示例包

这个包里的示例是围绕你当前工程形状写的，不是完全脱离现有项目的“新系统”。

文件说明：

- `models_v2_example.py`
  - 兼容式对象升级示例
- `relation_store_v2_example.py`
  - 按 `relation_id` 保存双向关系状态
- `memory_reweave_v2_example.py`
  - 叙事重织的最小可运行示例
- `sleep_bridge_example.py`
  - 保留 `self/world/open` 三元组的桥接版 sleep
- `aurora_patch_examples_cn.md`
  - 如何把这些示例接到你当前 `engine.py / store.py / sleep.py`

建议你先落地这三件事：

1. 先修 `handle_turn()` 的 safety / commit 顺序
2. 再把 `RelationStore` 改成按 `relation_id` 聚合
3. 最后把 `run_sleep_reweave()` 从单一 delta 升级成 `ReweaveResult`
