# Micro-Aurora TDD 实战重构训练营 (Bootcamp)

> **"What I cannot create, I do not understand." —— Richard Feynman**

欢迎来到硅谷级工业强度的“测试驱动开发 (TDD)”系统化闯关训练营！
从这里开始，我不会再让你“阅读源码大纲”，也不会让你“执行我写好的成品”。
你将成为一台“人肉编译器”，你必须手写出 Aurora 系统中最核心的几段算法和架构枢纽，让你的代码经受住极为严苛的单元测试（Unit Test）的拷打。只有看到全屏绿色 `PASSED` 的那一刻，你才算真正“系统掌握”了这个组件的灵魂。

---

## ⚔️ 你的破壁通关规则

对于每一个 Lab（实验室），提交流程极其残酷且固定：

1. **进入对应的 Lab 文件夹**（例如 `cd Learning/Bootcamp/Lab_01_Foundation`）
2. **观察护城河**：你绝对不能修改 `test_xxx.py`！这是系统判定你是否存活的唯一法官。
3. **直面死亡终端**：在根目录毫不犹豫地运行测试命令 `uv run pytest Learning/Bootcamp/Lab_01_Foundation/test_models.py`。
   - 看到漫天的红色 `AssertionError` 和 `FAILED`？恭喜你，你的训练开始了！
4. **填补空缺灵魂**：打开同目录下的 `task_xxx.py` 文件。
   - 找到所有带有大写 `# TODO` 标记的留空区域。
   - 结合原版 Aurora 的源码逻辑和你对基建或高维算法的理解，亲自敲下每一行代码。
5. **涅槃绿灯**：反复执行测试，直到终端打出高亮耀眼的 `100% PASSED`！你彻底征服了这一关。

---

## 🏆 训练营总关卡一览
(*通关后请手动打勾 `[x]`*)

- [ ] **Lab 01: 坚不可摧的底层基石** (`Lab_01_Foundation`)
      *目标：徒手捏出一个防篡改的 Dataclass，并配置带有类型强校验拦截的 Pydantic 环境雷达。*
- [ ] **Lab 02: 高并发的心脏除颤** (`Lab_02_Engine`)
      *目标：手写多线程状态锁 (Lock)，以及 Event Sourcing 无法逆转的追加流。*
- [ ] **Lab 03: 降维打击与空间几何** (`Lab_03_Vector`)
      *目标：不依赖大模型，基于 Numpy 手写大语言模型最核心的余弦相似度测距 (Cosine Similarity)。*
- [ ] **Lab 04: AI 的心智骰子** (`Lab_04_Probability`)
      *目标：手撸出主导大模型废话丢弃机制和性格摇号的随机概率与运算推演。*
- [ ] **Lab 05: 穿越时空的回忆迷宫** (`Lab_05_Graph`)
      *目标：完成基于广度优先搜索 (BFS) 与软概率阈值的记忆提取链路。*

---
> ⚠️ **温馨警告**：如果你卡住了超过 15 分钟，请不要硬撑！你随时能在当前会话呼叫我，或者参考原版 `aurora/` 下的对应源码进行思维借鉴。

现在，深吸一口气，打开 `Learning/Bootcamp/Lab_01_Foundation/task_models.py`，写下属于你的第一行架构级代码吧！
