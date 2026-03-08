# 叙事涌现记忆系统（Narrative Emergent Memory System）

**——让AI学会给自己讲故事**

---

## 序言：一个更深刻的问题

现有的Agent记忆系统都在回答同一个问题：**如何存储和检索信息？**

这个问题本身就是错的。

认知科学告诉我们一个惊人的事实：人类记忆不是档案馆，而是**故事生成器**。我们不记得"发生了什么"，我们记得的是"我们告诉自己发生了什么的故事"。每次回忆都是一次重新讲述，而非读取。

这暗示了一个全新的设计哲学：

> **记忆系统不应该是数据库，而应该是一个能够持续讲述、修订、丰富自己故事的叙事引擎。**

本文档描述的系统，试图回答一个更根本的问题：

> **如何让Agent拥有一个持续演化的、一致的、有意义的"自我"？**

答案是：**让它学会给自己讲故事。**

---

## 第一章：核心理念

### 1.1 三个颠覆性洞见

**洞见一：存储故事，而非事实**

传统系统存储的是：节点（情景、概念、规则）+ 边（关联）

本系统存储的是：**故事弧（Story Arc）**

每一段记忆不是孤立的数据点，而是一个完整的叙事结构：
- **主角**（Agent自己、用户、其他实体）
- **意图**（为什么这件事发生）
- **冲突**（遇到了什么阻碍）
- **转折**（关键决策或意外）
- **结局**（成功/失败/悬而未决）
- **寓意**（从中学到了什么）

为什么这更优雅？因为**故事天然是压缩**。一个好故事用200字能传递的信息，碎片化存储可能需要2000字。而且故事**天然可泛化**——当我记住"狼来了"的故事，我不需要再记住每一个"谎言最终被揭穿"的具体案例。

**洞见二：检索是重新讲述**

传统系统检索的是：最相关的记忆片段

本系统检索的是：**在当前情境下重新讲述一个相关的故事**

同一段经历，在不同情境下讲述，会强调不同的方面。例如，同一次"项目失败"的记忆：
- 当用户问"如何估算时间"时 → 重点是"当时低估了复杂度"
- 当用户问"如何处理团队冲突"时 → 重点是"当时没有及时沟通"
- 当用户问"要不要接这个新项目"时 → 重点是"那种类型的项目风险高"

**记忆的内容随讲述的目的而改变**——这就是为什么生成比检索更根本。

**洞见三：自我从叙事一致性中涌现**

传统系统追求的是：更好的任务完成

本系统追求的是：**涌现出一个具有一致性的"自我"**

Agent不应该只是"有记忆"，而应该拥有一个从所有故事中涌现出的**自我叙事（Self-Narrative）**：
- "我是一个什么样的助手"
- "我和这个用户的关系是什么"
- "我的能力边界在哪里"
- "我犯过什么错误，学到了什么"

这个自我叙事不是预先编程的，而是从无数小故事中自然涌现的。

### 1.2 从现有研究中采纳的精华

| 来源 | 采纳的思想 | 如何融入本架构 |
|------|-----------|--------------|
| Nemori | 事件分割理论（Event Segmentation Theory）| 用于情节边界检测 |
| Nemori | 预测-校准机制 | 用于叙事张力计算 |
| Amory | 情节/子情节的层次结构 | 故事弧的组织方式 |
| Amory | 动量感知整合（Momentum-aware Consolidation）| 故事的自然凝聚 |
| Stanford Generative Agents | 层次化反思 | 主题和自我叙事的涌现 |
| SEEM | 反向溯源扩展 | 故事重构时的上下文恢复 |
| Membox | Topic Loom | 叙事线索追踪 |
| 用户原方案 | 激活传播检索 | 故事网络中的关联激活 |
| 用户原方案 | 图结构自演化 | 故事关系的动态演化 |
| 用户原方案 | 统计驱动的价值评估 | 故事价值的涌现式计算 |

### 1.3 设计原则

**原则一：叙事优先（Narrative First）**

一切设计决策都以"是否有利于形成和维护连贯叙事"为标准。存储结构、检索算法、整合机制都服务于叙事。

**原则二：涌现而非编程（Emergence over Programming）**

高层结构（主题、自我叙事）不是人工定义的，而是从低层结构（情节、故事弧）中自然涌现的。系统的"智慧"来自结构演化，而非预设规则。

**原则三：重构而非读取（Reconstruction over Retrieval）**

每次"回忆"都是根据当前情境重新生成故事，而非简单读取存储的文本。这允许同一段经历在不同上下文中展现不同面向。

**原则四：一致性自维护（Self-Maintaining Coherence）**

系统持续监控叙事一致性，主动发现和解决矛盾，而不是被动等待错误发生。

**原则五：优雅降级（Graceful Degradation）**

在资源受限时，系统自动降低叙事精度而非完全失效。情节可以融合进故事，故事可以融合进主题，但核心叙事始终存在。

**原则六：可实现性（Implementability）**

所有机制都可以用现有技术实现。不依赖训练神经网络，主要使用图结构演化 + LLM按需调用。

---

## 第二章：整体架构

### 2.1 四层叙事结构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         叙事涌现记忆系统                                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     第四层：自我层（Self Layer）                     │ │
│  │                                                                     │ │
│  │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │ │
│  │   │  身份叙事   │  │  关系叙事   │  │  能力叙事   │                │ │
│  │   │ "我是谁"   │  │"我和谁的关系"│ │"我能做什么" │                │ │
│  │   └─────────────┘  └─────────────┘  └─────────────┘               │ │
│  │                          ↑ 涌现                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     第三层：主题层（Theme Layer）                    │ │
│  │                                                                     │ │
│  │   [模式A]────[模式B]────[模式C]────[模式D]                          │ │
│  │      │          │          │          │                            │ │
│  │   反复出现的规律、教训、偏好、因果关系                               │ │
│  │                          ↑ 涌现                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    第二层：叙事层（Narrative Layer）                 │ │
│  │                                                                     │ │
│  │   ┌─────────┐     ┌─────────┐     ┌─────────┐                     │ │
│  │   │ 故事弧A │────│ 故事弧B │────│ 故事弧C │                       │ │
│  │   │(完整故事)│    │(进行中) │     │(完整故事)│                      │ │
│  │   └─────────┘     └─────────┘     └─────────┘                     │ │
│  │        │              │               │                            │ │
│  │   有主角、冲突、转折、结局、寓意的完整叙事单元                        │ │
│  │                          ↑ 凝聚                                    │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     第一层：情节层（Plot Layer）                     │ │
│  │                                                                     │ │
│  │   [情节1]→[情节2]→[情节3]→[情节4]→[情节5]→[情节6]→...              │ │
│  │                                                                     │ │
│  │   原子级事件：谁做了什么，在什么情境下，结果如何                       │ │
│  │   高保真但会自然衰减，是所有上层结构的原材料                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         核心引擎                                    │ │
│  │                                                                     │ │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │ │
│  │   │ 织入引擎 │  │ 讲述引擎 │  │ 演化引擎 │  │ 一致性   │          │ │
│  │   │ Weaver  │  │ Narrator │  │ Evolver  │  │ Guardian │          │ │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────┘          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 为什么是这个结构

**四层对应人类叙事记忆的自然层次**：
- 情节层 ≈ 闪回片段（"那天下午..."）
- 叙事层 ≈ 完整故事（"有一次我..."）
- 主题层 ≈ 人生教训（"我学到..."）
- 自我层 ≈ 身份认同（"我是一个..."）

**涌现关系是单向的**：
- 情节凝聚成故事（多个相关情节形成一个完整叙事）
- 故事涌现出主题（多个相似故事揭示一个模式）
- 主题构成自我（所有主题共同定义"我是谁"）

**但查询是双向的**：
- 自上而下：从自我叙事指导故事选择
- 自下而上：从情节细节支撑故事讲述

---

## 第三章：数据结构

### 3.1 情节（Plot）

情节是最小的叙事单元，记录一个原子级事件。

```python
@dataclass
class Plot:
    id: str
    timestamp: datetime
    
    # 核心叙事元素
    actors: List[str]           # 参与者（用户、Agent、其他实体）
    action: str                 # 发生了什么
    context: str                # 在什么情境下
    outcome: str                # 结果如何
    
    # 叙事张力信息
    tension_delta: float        # 张力变化（-1到1）
    surprise_score: float       # 惊讶度（0到1）
    emotional_valence: float    # 情感效价（-1到1）
    
    # 关联信息
    story_arc_id: Optional[str] # 所属故事弧
    causal_links: List[str]     # 因果关联的其他情节
    
    # 元信息
    embedding: np.ndarray       # 向量表示
    access_count: int           # 访问次数
    last_accessed: datetime     # 最后访问时间
    
    # 生命周期
    status: Literal["active", "absorbed", "archived"]
    absorbed_into: Optional[str]  # 如果被吸收，指向哪个故事/主题
```

### 3.2 故事弧（Story Arc）

故事弧是完整的叙事单元，有开端、发展、高潮、结局。

```python
@dataclass
class StoryArc:
    id: str
    created_at: datetime
    last_updated: datetime
    
    # 叙事结构（故事语法）
    title: str                  # 故事标题（自动生成）
    protagonist: str            # 主角
    deuteragonists: List[str]   # 其他重要角色
    
    # 故事阶段
    setup: str                  # 开端：建立情境
    rising_action: List[str]    # 发展：冲突升级
    climax: Optional[str]       # 高潮：关键转折
    falling_action: List[str]   # 收尾：结果展开
    resolution: Optional[str]   # 结局：最终状态
    
    # 故事核心
    central_conflict: str       # 核心冲突
    turning_points: List[str]   # 转折点
    moral: Optional[str]        # 寓意/教训（完成后提取）
    
    # 组成情节
    plot_ids: List[str]         # 构成这个故事的情节ID
    plot_sequence: List[str]    # 情节的叙事顺序（可能与时间顺序不同）
    
    # 状态
    status: Literal["developing", "climaxing", "resolved", "abandoned"]
    tension_curve: List[float]  # 张力曲线
    
    # 关联
    related_stories: List[Tuple[str, float]]  # (story_id, 关联强度)
    theme_ids: List[str]        # 关联的主题
    
    # 统计
    importance_score: float     # 重要性分数
    reference_count: int        # 被引用次数
    success_indicator: float    # 成功指标（如果适用）
    
    # 嵌入
    embedding: np.ndarray       # 故事级向量表示
```

### 3.3 主题（Theme）

主题是从多个故事中涌现的稳定模式。

```python
@dataclass
class Theme:
    id: str
    created_at: datetime
    last_validated: datetime
    
    # 主题内容
    name: str                   # 主题名称
    description: str            # 主题描述
    theme_type: Literal[
        "pattern",      # 行为模式（"用户倾向于..."）
        "lesson",       # 教训（"不应该..."）
        "preference",   # 偏好（"用户喜欢..."）
        "causality",    # 因果（"X通常导致Y"）
        "capability",   # 能力（"我擅长..."）
        "limitation"    # 局限（"我不擅长..."）
    ]
    
    # 证据
    supporting_stories: List[str]   # 支持这个主题的故事
    contradicting_stories: List[str] # 与这个主题矛盾的故事
    
    # 置信度
    confidence: float           # 置信度（0-1）
    evidence_strength: float    # 证据强度
    consistency_score: float    # 一致性分数
    
    # 可证伪性
    falsification_conditions: List[str]  # 什么情况下这个主题应该被修正
    
    # 使用统计
    application_count: int      # 被应用次数
    success_rate: float         # 应用成功率
    
    # 关联
    related_themes: List[Tuple[str, str, float]]  # (theme_id, 关系类型, 强度)
    
    # 嵌入
    embedding: np.ndarray
```

### 3.4 自我叙事（Self Narrative）

自我叙事是Agent对"我是谁"的元叙事。

```python
@dataclass
class SelfNarrative:
    id: str
    last_updated: datetime
    
    # 核心身份叙事
    identity_statement: str     # "我是..."的陈述
    
    # 三个维度
    identity_narrative: str     # 身份叙事："我是一个怎样的助手"
    relationship_narratives: Dict[str, str]  # 关系叙事：与不同用户/实体的关系
    capability_narrative: str   # 能力叙事："我能做什么，不能做什么"
    
    # 核心信条（不可变的核心价值，类似Sophia的creed）
    core_beliefs: List[str]     # 最多5条核心信条
    
    # 支撑主题
    supporting_themes: List[str]  # 支撑这个自我叙事的主题
    
    # 叙事一致性
    coherence_score: float      # 整体一致性分数
    unresolved_tensions: List[str]  # 未解决的内部张力
    
    # 演化历史
    evolution_log: List[Tuple[datetime, str, str]]  # (时间, 变化类型, 变化内容)
```

### 3.5 故事网络（Story Web）

所有叙事元素通过一个图结构连接。

```python
@dataclass
class StoryWeb:
    # 节点
    plots: Dict[str, Plot]
    story_arcs: Dict[str, StoryArc]
    themes: Dict[str, Theme]
    self_narrative: SelfNarrative
    
    # 边（关系）
    edges: Dict[Tuple[str, str], StoryEdge]
    
    # 索引
    vector_index: VectorIndex       # 向量检索索引
    temporal_index: TemporalIndex   # 时间索引
    actor_index: Dict[str, List[str]]  # 按参与者索引
    
    # 活跃状态
    active_story_arcs: List[str]    # 当前进行中的故事
    working_context: WorkingContext  # 工作记忆/上下文

@dataclass
class StoryEdge:
    source_id: str
    target_id: str
    edge_type: Literal[
        "temporal",      # 时序关系
        "causal",        # 因果关系
        "thematic",      # 主题关联
        "character",     # 共同角色
        "continuation",  # 故事延续
        "contradiction", # 矛盾关系
        "supports",      # 支持关系
        "exemplifies"    # 例证关系
    ]
    weight: float        # 关系强度
    
    # 演化统计
    co_activation_count: int
    co_success_count: int
    created_at: datetime
    last_strengthened: datetime
```

---

## 第四章：核心引擎

### 4.1 织入引擎（Weaver）

负责将新的交互编织进叙事网络。

```
┌─────────────────────────────────────────────────────────────────┐
│                        织入引擎 Weaver                          │
│                                                                 │
│  输入：新的交互（用户消息 + Agent响应 + 结果反馈）                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤1：情节提取（Plot Extraction）                       │   │
│  │                                                          │   │
│  │ 从原始交互中提取结构化情节：                              │   │
│  │ - 识别参与者（actors）                                   │   │
│  │ - 提取行动（action）                                     │   │
│  │ - 理解情境（context）                                    │   │
│  │ - 判断结果（outcome）                                    │   │
│  │                                                          │   │
│  │ 实现：规则提取 + LLM辅助（复杂情况）                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤2：张力计算（Tension Calculation）                   │   │
│  │                                                          │   │
│  │ 计算这个情节的叙事张力：                                  │   │
│  │                                                          │   │
│  │ tension_delta = α × surprise_score                       │   │
│  │               + β × conflict_intensity                   │   │
│  │               + γ × emotional_shift                      │   │
│  │               + δ × goal_relevance                       │   │
│  │                                                          │   │
│  │ 惊讶度计算（借鉴Nemori）：                                │   │
│  │ - 与当前故事预期的偏离程度                                │   │
│  │ - 与已知主题的矛盾程度                                    │   │
│  │ - 信息论：与最相似已有情节的距离                          │   │
│  │                                                          │   │
│  │ 实现：统计计算 + 向量距离                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤3：故事归属（Story Attribution）                     │   │
│  │                                                          │   │
│  │ 判断这个情节属于哪个故事弧：                              │   │
│  │                                                          │   │
│  │ for each active_story in active_story_arcs:              │   │
│  │     fit_score = calculate_narrative_fit(plot, story)     │   │
│  │     if fit_score > CONTINUATION_THRESHOLD:               │   │
│  │         → 加入该故事的发展阶段                            │   │
│  │         → 更新故事的张力曲线                              │   │
│  │                                                          │   │
│  │ if no good fit:                                          │   │
│  │     if tension_delta > NEW_STORY_THRESHOLD:              │   │
│  │         → 创建新故事弧                                    │   │
│  │     else:                                                │   │
│  │         → 标记为独立情节                                  │   │
│  │                                                          │   │
│  │ narrative_fit 考虑：                                     │   │
│  │ - 参与者重叠                                             │   │
│  │ - 主题连续性                                             │   │
│  │ - 时间邻近性                                             │   │
│  │ - 因果关联性                                             │   │
│  │                                                          │   │
│  │ 实现：规则 + 激活传播                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤4：边界检测（Boundary Detection）                    │   │
│  │                                                          │   │
│  │ 借鉴事件分割理论（Event Segmentation Theory）：           │   │
│  │                                                          │   │
│  │ 检测故事是否到达边界（高潮或结局）：                       │   │
│  │ - 张力达到峰值后开始下降 → 可能是高潮                     │   │
│  │ - 核心冲突得到解决 → 可能是结局                          │   │
│  │ - 长时间无新情节 → 可能是搁置                            │   │
│  │ - 参与者明确告别/结束 → 明确结局                         │   │
│  │                                                          │   │
│  │ 如果检测到高潮：                                         │   │
│  │     → 标记故事状态为 climaxing                           │   │
│  │ 如果检测到结局：                                         │   │
│  │     → 触发故事完结流程                                    │   │
│  │     → 提取寓意（moral）                                  │   │
│  │     → 检查是否涌现新主题                                  │   │
│  │                                                          │   │
│  │ 实现：张力曲线分析 + 规则                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤5：关系编织（Relation Weaving）                      │   │
│  │                                                          │   │
│  │ 在故事网络中建立/强化关系：                               │   │
│  │                                                          │   │
│  │ - 与同故事其他情节的时序边                                │   │
│  │ - 与前置情节的因果边                                      │   │
│  │ - 与相似情节的主题边                                      │   │
│  │ - 与共同参与者的角色边                                    │   │
│  │                                                          │   │
│  │ 边权重初始化：                                           │   │
│  │     weight = base_weight × confidence                    │   │
│  │                                                          │   │
│  │ 实现：规则 + 向量相似度                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  输出：更新后的故事网络                                         │
└─────────────────────────────────────────────────────────────────┘
```

**关键算法：叙事契合度计算**

```python
def calculate_narrative_fit(plot: Plot, story: StoryArc) -> float:
    """计算情节与故事的叙事契合度"""
    
    # 1. 参与者重叠
    actor_overlap = len(set(plot.actors) & set(story.get_all_actors()))
    actor_score = actor_overlap / max(len(plot.actors), 1)
    
    # 2. 主题连续性（与故事当前阶段的语义相似度）
    current_stage_embedding = story.get_current_stage_embedding()
    thematic_score = cosine_similarity(plot.embedding, current_stage_embedding)
    
    # 3. 时间邻近性
    time_gap = (plot.timestamp - story.last_updated).total_seconds()
    temporal_score = math.exp(-time_gap / TEMPORAL_DECAY_CONSTANT)
    
    # 4. 因果关联性（是否是故事当前状态的合理后续）
    if story.status == "developing":
        # 检查是否是冲突升级的合理延续
        causal_score = estimate_causal_plausibility(story, plot)
    elif story.status == "climaxing":
        # 检查是否是高潮的合理结果
        causal_score = estimate_resolution_fit(story, plot)
    else:
        causal_score = 0.0
    
    # 加权综合
    fit_score = (
        0.25 * actor_score +
        0.30 * thematic_score +
        0.20 * temporal_score +
        0.25 * causal_score
    )
    
    return fit_score
```

### 4.2 讲述引擎（Narrator）

负责根据当前需求重新讲述相关故事。

```
┌─────────────────────────────────────────────────────────────────┐
│                       讲述引擎 Narrator                         │
│                                                                 │
│  输入：当前情境 + 讲述目的                                       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤1：意图理解（Intent Understanding）                  │   │
│  │                                                          │   │
│  │ 理解当前需要什么样的故事：                                │   │
│  │                                                          │   │
│  │ narrative_need = classify_need(context)                  │   │
│  │                                                          │   │
│  │ 可能的需求类型：                                         │   │
│  │ - PRECEDENT: 需要先例（"之前有类似情况吗？"）             │   │
│  │ - LESSON: 需要教训（"上次为什么失败？"）                  │   │
│  │ - PATTERN: 需要模式（"用户通常怎么..."）                  │   │
│  │ - CONTEXT: 需要背景（"我们之前聊过什么？"）               │   │
│  │ - IDENTITY: 需要身份（"我是怎样的助手？"）                │   │
│  │ - RELATIONSHIP: 需要关系（"我和用户的关系如何？"）         │   │
│  │                                                          │   │
│  │ 实现：规则分类 + LLM辅助                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤2：故事检索（Story Retrieval）                       │   │
│  │                                                          │   │
│  │ 使用激活传播在故事网络中找到相关故事：                     │   │
│  │                                                          │   │
│  │ 2.1 初始激活                                             │   │
│  │     - 向量检索找到语义相关的节点                          │   │
│  │     - 工作上下文中的节点获得额外激活                       │   │
│  │     - 根据narrative_need加权不同类型的节点                 │   │
│  │                                                          │   │
│  │ 2.2 激活传播                                             │   │
│  │     for step in range(MAX_PROPAGATION_STEPS):            │   │
│  │         for node in active_nodes:                        │   │
│  │             for edge in node.edges:                      │   │
│  │                 target.activation += (                   │   │
│  │                     node.activation ×                    │   │
│  │                     edge.weight ×                        │   │
│  │                     edge_type_weight[edge.type] ×        │   │
│  │                     DECAY_FACTOR                         │   │
│  │                 )                                        │   │
│  │         normalize_and_prune()                            │   │
│  │                                                          │   │
│  │ 2.3 收集结果                                             │   │
│  │     - 按激活值排序                                       │   │
│  │     - 优先选择故事弧（完整叙事），而非散落情节             │   │
│  │     - 考虑叙事多样性（不要只选相似故事）                   │   │
│  │                                                          │   │
│  │ 实现：向量检索 + 图上激活传播                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │   
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤3：视角选择（Perspective Selection）                 │   │
│  │                                                          │   │
│  │ 根据讲述目的选择讲述视角：                                │   │
│  │                                                          │   │
│  │ perspective = select_perspective(narrative_need, story)  │   │
│  │                                                          │   │
│  │ 可能的视角：                                             │   │
│  │ - CHRONOLOGICAL: 按时间顺序讲述                          │   │
│  │ - RETROSPECTIVE: 从结果回溯原因                          │   │
│  │ - CONTRASTIVE: 与其他故事对比讲述                        │   │
│  │ - FOCUSED: 只讲述与当前问题相关的部分                     │   │
│  │ - ABSTRACTED: 只讲主题/寓意，省略细节                     │   │
│  │                                                          │   │
│  │ 同时确定：                                               │   │
│  │ - 详略程度（多少细节）                                    │   │
│  │ - 强调点（突出哪些方面）                                  │   │
│  │ - 情感色调（中性/积极/警示）                              │   │
│  │                                                          │   │
│  │ 实现：规则映射                                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤4：上下文恢复（Context Recovery）                    │   │
│  │                                                          │   │
│  │ 借鉴SEEM的反向溯源扩展：                                  │   │
│  │                                                          │   │
│  │ 为选中的故事恢复必要的上下文：                            │   │
│  │ - 追溯因果链上的前置情节                                  │   │
│  │ - 恢复关键转折点的细节                                    │   │
│  │ - 获取相关主题作为解释框架                                │   │
│  │                                                          │   │
│  │ context = {                                              │   │
│  │     "story": selected_story,                             │   │
│  │     "key_plots": [重要情节],                             │   │
│  │     "causal_chain": [因果链],                            │   │
│  │     "relevant_themes": [相关主题],                       │   │
│  │     "perspective": perspective,                          │   │
│  │     "emphasis": emphasis_points                          │   │
│  │ }                                                        │   │
│  │                                                          │   │
│  │ 实现：图遍历                                             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤5：故事重构（Story Reconstruction）                  │   │
│  │                                                          │   │
│  │ 由LLM根据上下文重新讲述故事：                             │   │
│  │                                                          │   │
│  │ prompt = f"""                                            │   │
│  │ 你需要重新讲述以下故事，目的是{narrative_need}。          │   │
│  │                                                          │   │
│  │ 故事核心：{story.summary}                                │   │
│  │ 关键情节：{key_plots}                                    │   │
│  │ 因果链：{causal_chain}                                   │   │
│  │ 相关主题：{relevant_themes}                              │   │
│  │                                                          │   │
│  │ 讲述要求：                                               │   │
│  │ - 视角：{perspective}                                    │   │
│  │ - 详略：{detail_level}                                   │   │
│  │ - 强调：{emphasis_points}                                │   │
│  │ - 语调：{tone}                                           │   │
│  │                                                          │   │
│  │ 请用自然的叙事方式重新讲述这个故事。                       │   │
│  │ """                                                      │   │
│  │                                                          │   │
│  │ reconstructed_story = llm(prompt)                        │   │
│  │                                                          │   │
│  │ 实现：LLM调用                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              ↓                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 步骤6：记录与反馈（Record & Feedback）                   │   │
│  │                                                          │   │
│  │ 记录这次讲述，用于后续学习：                              │   │
│  │ - 哪些故事被检索                                         │   │
│  │ - 使用了什么视角                                         │   │
│  │ - 后续任务是否成功（异步获取）                            │   │
│  │                                                          │   │
│  │ 强化有用的关联：                                         │   │
│  │ - 如果讲述后任务成功，强化相关边                          │   │
│  │ - 如果讲述后任务失败，弱化或标记                          │   │
│  │                                                          │   │
│  │ 实现：日志 + 异步回调                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  输出：重构后的故事叙述                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 演化引擎（Evolver）

负责叙事结构的自然演化——情节凝聚成故事，故事涌现出主题，主题构成自我。

```
┌─────────────────────────────────────────────────────────────────┐
│                       演化引擎 Evolver                          │
│                                                                 │
│  触发时机：                                                     │
│  - 定时触发（每天/每周）                                        │
│  - 故事完结时                                                   │
│  - 累积足够新内容时                                             │
│  - 空闲时间                                                     │
│                                                                 │
│  ════════════════════════════════════════════════════════════   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 演化1：边权重自然演化                                    │   │
│  │                                                          │   │
│  │ 类似用户原方案，边权重从使用中自然涌现：                   │   │
│  │                                                          │   │
│  │ 规则1：共同激活强化                                      │   │
│  │     当节点A和B在同一次检索中都被激活：                    │   │
│  │     edge(A,B).weight += α × min(A.activation, B.activation)│   │
│  │                                                          │   │
│  │ 规则2：共同成功强化                                      │   │
│  │     当使用了A和B的任务成功：                              │   │
│  │     edge(A,B).weight += β × success_bonus                │   │
│  │                                                          │   │
│  │ 规则3：时间衰减                                          │   │
│  │     定期对所有边：                                       │   │
│  │     edge.weight *= (1 - decay_rate)                      │   │
│  │                                                          │   │
│  │ 规则4：矛盾弱化                                          │   │
│  │     如果A和B被标记为矛盾：                               │   │
│  │     edge(A,B).weight = min(edge.weight, CONTRADICTION_CAP)│   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 演化2：情节吸收（Plot Absorption）                       │   │
│  │                                                          │   │
│  │ 老旧、低价值的情节被吸收进更高层结构：                     │   │
│  │                                                          │   │
│  │ for plot in old_plots:                                   │   │
│  │     if plot.importance < ABSORPTION_THRESHOLD:           │   │
│  │         if plot.story_arc_id:                            │   │
│  │             # 情节细节融入故事摘要                        │   │
│  │             story = get_story(plot.story_arc_id)         │   │
│  │             story.absorb_plot_details(plot)              │   │
│  │             plot.status = "absorbed"                     │   │
│  │         else:                                            │   │
│  │             # 独立情节直接归档                            │   │
│  │             plot.status = "archived"                     │   │
│  │                                                          │   │
│  │ 吸收不是删除，而是：                                     │   │
│  │ - 情节的核心信息保留在故事的摘要中                        │   │
│  │ - 详细内容可以被归档（冷存储）                           │   │
│  │ - 需要时可以通过故事重构恢复                             │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 演化3：主题涌现（Theme Emergence）                       │   │
│  │                                                          │   │
│  │ 从相似的故事中涌现出主题：                                │   │
│  │                                                          │   │
│  │ 3.1 故事聚类                                             │   │
│  │     clusters = cluster_stories_by_similarity()           │   │
│  │                                                          │   │
│  │ 3.2 模式检测                                             │   │
│  │     for cluster in clusters:                             │   │
│  │         if len(cluster) >= MIN_STORIES_FOR_THEME:        │   │
│  │             patterns = detect_common_patterns(cluster)   │   │
│  │             for pattern in patterns:                     │   │
│  │                 if pattern.strength > THEME_THRESHOLD:   │   │
│  │                     → 候选新主题                          │   │
│  │                                                          │   │
│  │ 3.3 LLM辅助验证和命名                                    │   │
│  │     prompt = f"""                                        │   │
│  │     以下故事似乎有共同的模式：                            │   │
│  │     {stories_summary}                                    │   │
│  │                                                          │   │
│  │     检测到的模式：{pattern}                              │   │
│  │                                                          │   │
│  │     请判断：                                             │   │
│  │     1. 这个模式是否真实存在？                            │   │
│  │     2. 如果存在，用一句话描述这个主题                     │   │
│  │     3. 这个主题的类型是什么（模式/教训/偏好/因果）        │   │
│  │     4. 什么情况下这个主题可能不适用                       │   │
│  │     """                                                  │   │
│  │     theme_info = llm(prompt)                             │   │
│  │                                                          │   │
│  │ 3.4 创建主题节点                                         │   │
│  │     if theme_info.is_valid:                              │   │
│  │         create_theme(theme_info)                         │   │
│  │         link_to_supporting_stories()                     │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 演化4：主题更新（Theme Update）                          │   │
│  │                                                          │   │
│  │ 已有主题根据新证据更新：                                  │   │
│  │                                                          │   │
│  │ for theme in all_themes:                                 │   │
│  │     # 收集新证据                                         │   │
│  │     new_supporting = find_new_supporting_stories(theme)  │   │
│  │     new_contradicting = find_contradicting_stories(theme)│   │
│  │                                                          │   │
│  │     # 贝叶斯更新置信度                                   │   │
│  │     theme.confidence = bayesian_update(                  │   │
│  │         prior=theme.confidence,                          │   │
│  │         supporting=new_supporting,                       │   │
│  │         contradicting=new_contradicting                  │   │
│  │     )                                                    │   │
│  │                                                          │   │
│  │     # 置信度过低则标记或分裂                             │   │
│  │     if theme.confidence < DEPRECATION_THRESHOLD:         │   │
│  │         if can_split_into_subtypes(theme):               │   │
│  │             split_theme(theme)  # "X成立，除非Y"          │   │
│  │         else:                                            │   │
│  │             deprecate_theme(theme)                       │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 演化5：自我叙事更新（Self-Narrative Update）             │   │
│  │                                                          │   │
│  │ 定期根据主题重新审视自我叙事：                            │   │
│  │                                                          │   │
│  │ # 收集影响自我认知的主题                                 │   │
│  │ identity_themes = filter_themes(type="capability" or "limitation")│
│  │ relationship_themes = filter_themes(type="pattern", about="user")│
│  │                                                          │   │
│  │ # LLM辅助更新自我叙事                                    │   │
│  │ prompt = f"""                                            │   │
│  │ 当前的自我叙事：                                         │   │
│  │ {current_self_narrative}                                 │   │
│  │                                                          │   │
│  │ 最近涌现/更新的主题：                                    │   │
│  │ {recent_themes}                                          │   │
│  │                                                          │   │
│  │ 请判断自我叙事是否需要更新。如果需要，给出更新后的版本。   │   │
│  │ 注意：核心信条（core_beliefs）不应改变，除非有重大矛盾。  │   │
│  │ """                                                      │   │
│  │                                                          │   │
│  │ updated_narrative = llm(prompt)                          │   │
│  │ if significant_change(updated_narrative):                │   │
│  │     log_evolution(old, new)                              │   │
│  │     self_narrative.update(updated_narrative)             │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 演化6：结构清理（Structure Cleanup）                     │   │
│  │                                                          │   │
│  │ 定期清理图结构：                                         │   │
│  │                                                          │   │
│  │ - 删除权重低于阈值的边                                   │   │
│  │ - 合并高度相似的节点                                     │   │
│  │ - 归档长期未访问的内容                                   │   │
│  │ - 重建向量索引                                           │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 一致性守护者（Coherence Guardian）

负责维护叙事一致性，主动发现和解决矛盾。

```
┌─────────────────────────────────────────────────────────────────┐
│                    一致性守护者 Coherence Guardian              │
│                                                                 │
│  核心职责：确保整个叙事网络的内部一致性                          │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 检测1：事实矛盾检测                                      │   │
│  │                                                          │   │
│  │ 检测不同故事/情节中的事实矛盾：                           │   │
│  │                                                          │   │
│  │ 例如：                                                   │   │
│  │ - 故事A："用户说他喜欢安静的餐厅"                        │   │
│  │ - 故事B："用户说他喜欢热闹的餐厅"                        │   │
│  │                                                          │   │
│  │ 检测方法：                                               │   │
│  │ - 对涉及同一实体的断言进行语义比对                        │   │
│  │ - 用LLM判断是否存在逻辑矛盾                              │   │
│  │                                                          │   │
│  │ 处理：                                                   │   │
│  │ - 标记矛盾关系                                           │   │
│  │ - 尝试调和（可能是不同情境下的不同偏好）                  │   │
│  │ - 如果无法调和，降低旧断言的置信度                        │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 检测2：叙事断裂检测                                      │   │
│  │                                                          │   │
│  │ 检测未完成的故事弧：                                      │   │
│  │                                                          │   │
│  │ for story in story_arcs:                                 │   │
│  │     if story.status == "developing":                     │   │
│  │         idle_time = now - story.last_updated             │   │
│  │         if idle_time > ABANDONMENT_THRESHOLD:            │   │
│  │             → 标记为"悬而未决"                            │   │
│  │             → 可能需要主动询问用户                        │   │
│  │                                                          │   │
│  │     if story.has_unresolved_conflict:                    │   │
│  │         → 记录为叙事张力                                  │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 检测3：自我一致性检测                                    │   │
│  │                                                          │   │
│  │ 检测自我叙事与实际行为的一致性：                          │   │
│  │                                                          │   │
│  │ 例如：                                                   │   │
│  │ - 自我叙事："我擅长编程任务"                             │   │
│  │ - 但最近5个编程任务有4个失败                             │   │
│  │                                                          │   │
│  │ 检测方法：                                               │   │
│  │ - 对比自我叙事中的能力声明与实际任务结果                  │   │
│  │ - 对比关系叙事与实际交互模式                             │   │
│  │                                                          │   │
│  │ 处理：                                                   │   │
│  │ - 标记为"需要修正"                                       │   │
│  │ - 在下次演化周期更新自我叙事                             │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 检测4：主题冲突检测                                      │   │
│  │                                                          │   │
│  │ 检测相互矛盾的主题：                                      │   │
│  │                                                          │   │
│  │ 例如：                                                   │   │
│  │ - 主题A："用户倾向于快速得到答案"                        │   │
│  │ - 主题B："用户喜欢详细的解释"                            │   │
│  │                                                          │   │
│  │ 处理：                                                   │   │
│  │ - 尝试发现条件（"在X情况下A，在Y情况下B"）               │   │
│  │ - 如果无法调和，保留证据更强的主题                        │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 输出：一致性报告                                         │   │
│  │                                                          │   │
│  │ CoherenceReport = {                                      │   │
│  │     overall_score: float,          # 整体一致性分数       │   │
│  │     factual_conflicts: [...],      # 事实矛盾列表         │   │
│  │     unfinished_stories: [...],     # 未完成故事列表       │   │
│  │     self_inconsistencies: [...],   # 自我不一致列表       │   │
│  │     theme_conflicts: [...],        # 主题冲突列表         │   │
│  │     recommended_actions: [...]     # 建议的修正行动       │   │
│  │ }                                                        │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第五章：核心流程

### 5.1 实时交互流程

```
用户输入
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 上下文准备                                               │
│                                                             │
│    working_context = {                                      │
│        current_input: 用户输入,                             │
│        active_stories: 当前进行中的故事,                    │
│        recent_plots: 最近的情节,                            │
│        relevant_themes: 相关主题,                           │
│        self_narrative: 自我叙事摘要                         │
│    }                                                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 叙事检索（如果需要）                                     │
│                                                             │
│    if needs_memory(user_input):                             │
│        narrative_need = understand_intent(user_input)       │
│        stories = narrator.retrieve_and_reconstruct(         │
│            context=working_context,                         │
│            need=narrative_need                              │
│        )                                                    │
│        working_context.add(retrieved_stories=stories)       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 生成响应                                                 │
│                                                             │
│    prompt = construct_prompt(                               │
│        user_input=user_input,                               │
│        context=working_context                              │
│    )                                                        │
│    response = llm(prompt)                                   │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 织入新情节                                               │
│                                                             │
│    plot = weaver.extract_plot(                              │
│        user_input=user_input,                               │
│        agent_response=response,                             │
│        context=working_context                              │
│    )                                                        │
│                                                             │
│    if plot.tension_delta > ENCODING_THRESHOLD:              │
│        weaver.weave_into_network(plot)                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 更新边权重                                               │
│                                                             │
│    # 强化本次激活的关联                                     │
│    evolver.update_edge_weights(                             │
│        activated_nodes=working_context.activated_nodes,     │
│        success=True  # 或根据后续反馈更新                   │
│    )                                                        │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
返回响应给用户
```

### 5.2 离线演化流程

```
演化触发（定时/空闲/累积触发）
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 完结检查                                                 │
│                                                             │
│    for story in active_stories:                             │
│        if should_conclude(story):                           │
│            conclude_story(story)                            │
│            extract_moral(story)                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. 主题涌现                                                 │
│                                                             │
│    story_clusters = cluster_similar_stories()               │
│    for cluster in story_clusters:                           │
│        patterns = detect_patterns(cluster)                  │
│        for pattern in patterns:                             │
│            if is_valid_theme(pattern):                      │
│                create_or_update_theme(pattern)              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. 一致性检查                                               │
│                                                             │
│    report = coherence_guardian.full_check()                 │
│    for conflict in report.conflicts:                        │
│        resolve_or_mark(conflict)                            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 自我叙事更新                                             │
│                                                             │
│    if should_update_self_narrative():                       │
│        evolver.update_self_narrative()                      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 结构清理                                                 │
│                                                             │
│    evolver.absorb_old_plots()                               │
│    evolver.prune_weak_edges()                               │
│    evolver.archive_cold_content()                           │
│    rebuild_indices()                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 第六章：实现细节

### 6.1 技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                        技术选型                              │
│                                                             │
│  图存储：                                                   │
│  ├── 开发阶段：NetworkX（Python原生，易调试）               │
│  └── 生产阶段：Neo4j（高性能图数据库）                      │
│                                                             │
│  向量检索：                                                 │
│  ├── FAISS（高效近似最近邻）                                │
│  └── 嵌入模型：sentence-transformers / OpenAI embeddings    │
│                                                             │
│  LLM接口：                                                  │
│  ├── 统一抽象层，支持多提供商                               │
│  ├── 响应缓存（相似请求复用）                               │
│  └── 成本追踪                                               │
│                                                             │
│  持久化：                                                   │
│  ├── SQLite（默认本地持久化）                               │
│  └── FAISS（可选本地向量加速）                              │
│                                                             │
│  调度：                                                     │
│  └── 本地周期任务（可选）                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 代码结构

```
narrative_memory/
├── core/
│   ├── data_structures/
│   │   ├── plot.py              # 情节数据结构
│   │   ├── story_arc.py         # 故事弧数据结构
│   │   ├── theme.py             # 主题数据结构
│   │   ├── self_narrative.py    # 自我叙事数据结构
│   │   └── story_web.py         # 故事网络（图结构）
│   │
│   ├── engines/
│   │   ├── weaver.py            # 织入引擎
│   │   ├── narrator.py          # 讲述引擎
│   │   ├── evolver.py           # 演化引擎
│   │   └── guardian.py          # 一致性守护者
│   │
│   └── utils/
│       ├── tension.py           # 张力计算
│       ├── similarity.py        # 相似度计算
│       └── activation.py        # 激活传播
│
├── retrieval/
│   ├── vector_index.py          # 向量索引
│   ├── graph_traversal.py       # 图遍历
│   └── activation_spread.py     # 激活传播检索
│
├── llm/
│   ├── interface.py             # LLM统一接口
│   ├── prompts/
│   │   ├── plot_extraction.py   # 情节提取prompt
│   │   ├── story_reconstruction.py  # 故事重构prompt
│   │   ├── theme_extraction.py  # 主题提取prompt
│   │   └── self_narrative.py    # 自我叙事prompt
│   └── cache.py                 # 响应缓存
│
├── persistence/
│   ├── graph_store.py           # 图存储
│   ├── vector_store.py          # 向量存储
│   └── archive.py               # 归档存储
│
├── api/
│   ├── memory_api.py            # 对外API
│   └── admin_api.py             # 管理API
│
└── main.py                      # 主入口
```

### 6.3 关键参数

```python
# 编码阈值
ENCODING_THRESHOLD = 0.3          # 张力变化超过此值才编码
NEW_STORY_THRESHOLD = 0.6         # 张力超过此值开启新故事
CONTINUATION_THRESHOLD = 0.4      # 叙事契合度超过此值继续现有故事

# 演化参数
ABSORPTION_THRESHOLD = 0.2        # 重要性低于此值的情节被吸收
MIN_STORIES_FOR_THEME = 3         # 至少3个故事才能涌现主题
THEME_CONFIDENCE_THRESHOLD = 0.6  # 主题置信度阈值
DEPRECATION_THRESHOLD = 0.3       # 置信度低于此值的主题被弃用

# 激活传播
MAX_PROPAGATION_STEPS = 5         # 最大传播步数
ACTIVATION_DECAY = 0.7            # 每步衰减系数
ACTIVATION_THRESHOLD = 0.1        # 低于此值的激活被剪枝

# 边权重演化
CO_ACTIVATION_BONUS = 0.05        # 共同激活奖励
CO_SUCCESS_BONUS = 0.1            # 共同成功奖励
EDGE_DECAY_RATE = 0.001           # 边权重衰减率
EDGE_PRUNE_THRESHOLD = 0.05       # 低于此值的边被删除

# 时间参数
STORY_ABANDONMENT_HOURS = 72      # 超过72小时无更新视为搁置
EVOLUTION_INTERVAL_HOURS = 24     # 演化周期：每24小时
ARCHIVE_AFTER_DAYS = 30           # 30天未访问的内容归档
```

---

## 第七章：实现路径

### 7.1 分阶段实现

```
阶段1：基础骨架（2周）
─────────────────────
目标：搭建可运行的最小系统

交付物：
├── 数据结构定义（Plot, StoryArc, Theme, SelfNarrative）
├── 基础图结构（NetworkX实现）
├── 向量索引（FAISS）
├── LLM接口封装
└── 简单的存储和检索

验收：能存储情节，能检索相似情节


阶段2：织入引擎（2周）
─────────────────────
目标：实现情节提取和故事归属

交付物：
├── 情节提取（规则 + LLM）
├── 张力计算
├── 故事归属判断
├── 边界检测
└── 关系编织

验收：新交互能自动归入正确的故事弧


阶段3：讲述引擎（2周）
─────────────────────
目标：实现基于目的的故事检索和重构

交付物：
├── 意图理解
├── 激活传播检索
├── 视角选择
├── 上下文恢复
└── LLM重构

验收：能根据不同目的讲述同一故事的不同版本


阶段4：演化引擎（2周）
─────────────────────
目标：实现叙事结构的自然演化

交付物：
├── 边权重演化
├── 情节吸收
├── 主题涌现
├── 主题更新
└── 自我叙事更新

验收：使用一段时间后能自动涌现主题


阶段5：一致性守护（1周）
─────────────────────
目标：实现矛盾检测和一致性维护

交付物：
├── 事实矛盾检测
├── 叙事断裂检测
├── 自我一致性检测
├── 主题冲突检测
└── 一致性报告

验收：能自动发现并标记矛盾


阶段6：优化与集成（2周）
─────────────────────
目标：性能优化和完整集成

交付物：
├── 性能优化（批量处理、缓存）
├── 完整API
├── 管理界面
├── 文档
└── 测试套件

验收：可以作为独立模块集成到Agent系统


总计：11周
```

### 7.2 验收标准

```
里程碑1（阶段1-2完成后）：
├── 能正确提取情节
├── 能将情节归入正确的故事
├── 能检测故事边界
└── 基本的存储检索工作

里程碑2（阶段3完成后）：
├── 能根据不同目的检索不同故事
├── 同一故事能以不同视角讲述
├── 检索结果比纯向量检索更相关
└── 重构的故事自然流畅

里程碑3（阶段4-5完成后）：
├── 运行一段时间后自动涌现主题
├── 自我叙事能根据经历更新
├── 能检测并标记矛盾
├── 边权重反映真实的关联强度

里程碑4（全部完成后）：
├── 完整API可用
├── 性能满足实时要求
├── 文档完整
└── 消融实验证明各组件有贡献
```

---

## 第八章：评估框架

### 8.1 核心指标

```
叙事质量指标：
─────────────
├── 故事完整性：完成的故事弧占比
├── 主题涌现率：每100个故事涌现的有效主题数
├── 一致性分数：无矛盾的叙事占比
├── 压缩率：原始情节数 / 保留情节数
└── 自我叙事稳定性：自我叙事的更新频率和幅度

检索质量指标：
─────────────
├── 相关性：检索结果与查询的相关度（人工评估）
├── 多样性：检索结果的多样性
├── 时效性：能否检索到最新相关内容
└── 对比基线：vs 纯向量检索的提升

任务性能指标：
─────────────
├── 任务成功率：使用记忆后的任务成功率
├── 上下文利用率：记忆被有效利用的比例
├── 用户满意度：用户对Agent记忆能力的评价
└── 个性化程度：Agent行为的个性化水平

效率指标：
─────────────
├── 响应延迟：检索 + 重构的总时间
├── LLM调用次数：每次交互的平均LLM调用
├── 存储效率：每单位存储的信息密度
└── 演化开销：离线演化的计算成本
```

### 8.2 消融实验

```
必须验证的组件贡献：

1. 去掉张力驱动编码 → 改为全部编码
   预期：存储爆炸，检索质量下降

2. 去掉故事弧结构 → 只存储独立情节
   预期：失去叙事连贯性，检索碎片化

3. 去掉激活传播 → 只用向量检索
   预期：检索相关性下降，失去上下文敏感性

4. 去掉主题涌现 → 只保留故事层
   预期：失去抽象能力，无法泛化

5. 去掉自我叙事 → 无元认知
   预期：行为不一致，无法自我修正

6. 去掉一致性守护 → 不检测矛盾
   预期：随时间累积矛盾，叙事崩溃
```

---

## 第九章：与现有方案的对比

### 9.1 与用户原方案对比

| 维度 | 用户原方案 | 本方案 |
|------|-----------|--------|
| 核心理念 | 训练模型处理记忆 | 叙事结构自演化 |
| 存储单元 | 情景/概念/规则节点 | 故事弧（完整叙事）|
| 编码驱动 | 惊讶度 | 叙事张力 |
| 检索方式 | 向量 + 评估模型 | 激活传播 + 故事重构 |
| 整合方式 | 压缩模型提取 | LLM + 主题涌现 |
| 元认知 | 无显式支持 | 自我叙事层 |
| 训练需求 | 需要4个神经网络 | 无需训练 |
| 实现周期 | ~22周 | ~11周 |

### 9.2 与用户改进方案对比

| 维度 | 用户改进方案 | 本方案 |
|------|------------|--------|
| 核心理念 | 图结构自演化 | 叙事结构自演化 |
| 存储单元 | 各类节点 | 故事弧为中心 |
| 编码驱动 | 不匹配度 | 叙事张力 |
| 检索方式 | 激活传播（保留）| 激活传播 + 重构 |
| 整合方式 | LLM + 社区检测 | LLM + 故事聚类 |
| 压缩方式 | 节点合并/删除 | 情节融入故事/主题 |
| 一致性 | 无显式机制 | 一致性守护者 |
| 独特创新 | 图演化无需训练 | 叙事涌现 + 自我层 |

### 9.3 与现有研究对比

| 系统 | 叙事支持 | 自我支持 | 本方案优势 |
|------|---------|---------|-----------|
| Amory | 情节/子情节 | 无 | 完整四层 + 自我叙事 |
| Nemori | 事件分割 | 无 | 故事弧 + 主题涌现 |
| Stanford GA | 反思机制 | 部分 | 显式自我层 + 一致性守护 |
| SEEM | 反向溯源 | 无 | 叙事重构 + 视角选择 |
| Sophia | System 3 | 核心信条 | 动态自我叙事演化 |

---

## 第十章：总结

### 10.1 核心创新

```
1. 叙事作为基本单元
   ───────────────
   不是存储"发生了什么"，而是存储"关于发生了什么的故事"。
   故事天然是压缩，天然可泛化。

2. 四层涌现结构
   ───────────────
   情节 → 故事 → 主题 → 自我
   每一层从下一层自然涌现，无需人工定义。

3. 重构而非读取
   ───────────────
   每次"回忆"都是根据当前情境重新讲述故事。
   同一段经历在不同上下文中展现不同面向。

4. 自我叙事
   ───────────────
   Agent不只是"有记忆"，而是拥有关于"我是谁"的元叙事。
   这个自我叙事从经历中涌现，持续演化，自我一致。

5. 一致性自维护
   ───────────────
   主动检测矛盾，主动解决冲突。
   叙事系统具有自省能力。
```

### 10.2 预期效果

如果这套架构工作正常，应该观察到：

1. **叙事连贯性**：Agent能够讲述关于用户、关于自己、关于过去交互的连贯故事
2. **适应性检索**：同一段经历在不同情境下被不同方式讲述
3. **主题抽象**：使用一段时间后，Agent能说出"我发现你通常..."这样的抽象观察
4. **自我认知**：Agent能准确描述自己的能力和局限
5. **一致性**：很少出现自相矛盾的情况，出现时能自我修正
6. **优雅降级**：即使细节遗忘，故事的核心寓意仍然保留

### 10.3 一句话总结

> **让Agent学会给自己讲故事——关于用户的故事，关于任务的故事，关于"我是谁"的故事。在讲述中记忆，在记忆中成长，在成长中形成自我。**

---

**文档版本**：1.0  
**日期**：2025-01-25  
**状态**：完整架构设计
