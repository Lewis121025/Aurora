下面这份就是**可直接交给开发者的 Aurora 内核动力学 spec**。

理论姿态先定清楚：
Aurora 这里把 FEP 只当**规范层启发**，不要求 `field.py` 去逐字实现其全部推导；这更符合近年的综述把它当作 self-organisation 的 normative account，同时也承认若把它当成严格定理需要额外假设。事件分割也不应只绑定在单一 prediction error spike 上，近年的综述把 prediction error、attention 和 working memory 放在同一个框架里讨论。([科学直通车][1])

工作记忆读出我直接采用 **modern Hopfield / attention 风格的场收敛**；STC 只作为“局部标记 + 全局整合窗口”的启发；HDC/VSA 与 wake-sleep EBM 保留为后续增强件，因为前者是一个模型家族，后者对 continual learning 的遗忘缓解有帮助，但都不该阻塞当前内核。([arXiv][2])

---

# 0. 总体设计原则

Aurora v2 内核只保留 4 条硬约束：

1. **写入前不做语义编译**
   只允许机械分包、通用编码、原始落锚。

2. **单一 trace 宇宙**
   prototype / procedure 不是新 memory class，而是 `TraceRecord` 的高阶角色状态；角色跃迁应通过局部 objective 的 mutation 接受来发生。

3. **在线决策 = 局部能量下降**
   `ASSIMILATE / ATTACH / SPLIT / BIRTH / INHIBIT` 都用同一套局部目标近似比较。

4. **慢系统 = predictor + replay + structure update**
   replay 不是“加减分”，而是直接改变 latent、edges、posterior、role transition。

---

# 1. 状态变量定义

## 1.1 全局状态

在时间 (t)，Aurora 的内核状态定义为：

[
M_t = (\mathcal A_t,\mathcal T_t,\mathcal E_t,\mathcal G_t,\Theta_t,\mathcal F_t,\mathcal W_t)
]

其中：

* (\mathcal A_t)：Anchors（不可变原始锚点）
* (\mathcal T_t)：Traces（可演化痕迹）
* (\mathcal E_t)：Edges（temporal / assoc / inhib / option）
* (\mathcal G_t)：PosteriorGroups（冲突组）
* (\Theta_t)：Slow predictor 参数与 hidden state
* (\mathcal F_t)：Session frontier
* (\mathcal W_t)：当前 workspace 活跃子态

## 1.2 Anchor

```python
@dataclass
class Anchor:
    anchor_id: str
    packet_id: str
    z: NDArray[float]              # dense latent
    z_hv: NDArray[int] | None      # optional HDC/VSA channel
    ts: float
    source_id: str
    residual_ref: str | None       # raw/blob pointer
    source_quality: float
```

Anchor 是不可变的。Aurora 的“可追溯性”来自 `Anchor`，不是来自 claim 文本。

## 1.3 TraceRecord

```python
@dataclass
class TraceRecord:
    trace_id: str
    z_mu: NDArray[float]           # latent centroid μ_i
    z_sigma_diag: NDArray[float]   # diagonal covariance Σ_i
    ctx_mu: NDArray[float]         # context centroid ν_i
    ctx_sigma_diag: NDArray[float] # context covariance Λ_i

    evidence: float                # e_i
    stability: float               # s_i ∈ [0,1]
    uncertainty: float             # u_i (derived but cached)
    fidelity: float                # f_i ∈ [0,1]
    pred_loss_ema: float           # \bar{ℓ}^{pred}_i
    access_ema: float
    last_access_ts: float

    t_start: float
    t_end: float
    anchor_reservoir: list[str]    # bounded high-fidelity anchor refs

    role_logits: dict[str, float]  # episodic / prototype / procedure
    member_ids: tuple[str, ...]    # for prototype-like trace
    path_signature: tuple[str, ...]# for procedure-like trace

    posterior_group_ids: tuple[str, ...]
```

### 单个 trace 的最小统计语义

* ( \mu_i = \text{z_mu} )：内容中心
* ( \Sigma_i = \mathrm{diag}(\text{z_sigma_diag}) )：内容分散
* ( \nu_i = \text{ctx_mu} )：被激活时的上下文中心
* ( \Lambda_i = \mathrm{diag}(\text{ctx_sigma_diag}) )：上下文分散
* ( e_i )：证据质量与次数的衰减累计
* ( s_i )：巩固深度
* ( u_i )：不确定性
* ( f_i )：保真度等级
* `role_logits`：同一 trace 在 episodic / prototype / procedure 三种角色上的分数

### uncertainty 的正式定义

不再单独手写规则更新，直接由三项组成：

[
u_i
===

\underbrace{\frac{1}{d}\mathrm{tr}(\Sigma_i)}*{\text{within-trace dispersion}}
+
\underbrace{\frac{\kappa_e}{e_i+\epsilon}}*{\text{lack of evidence}}
+
\underbrace{\kappa_p \bar{\ell}^{pred}*i}*{\text{predictive instability}}
]

这三个量都可直接从当前 trace 与 predictor 统计得到。

## 1.4 TraceEdge

```python
@dataclass
class TraceEdge:
    src: str
    dst: str
    kind: Literal["temporal", "assoc", "inhib", "option"]
    weight: float
    support_ema: float
    last_update_ts: float
```

## 1.5 PosteriorGroup

```python
@dataclass
class PosteriorGroup:
    group_id: str
    member_ids: list[str]          # mutually competing traces
    alpha: NDArray[float]          # Dirichlet-like decayed counts, includes NULL slot
    ctx_mu: NDArray[float]         # per-member context centroids
    ctx_sigma_diag: NDArray[float] # per-member context variances
    pred_success_ema: NDArray[float]
    temperature: float
    unresolved_mass: float
    ambiguous_buffer: list[str]    # bounded anchor ids
```

这里的 `NULL slot` 是必须的。
它表示：“当前证据不支持任何现有成员，但也不值得立刻 birth 一个全新 trace。”

## 1.6 PredictorState

```python
@dataclass
class PredictorState:
    h: NDArray[float]
    theta: dict[str, NDArray[float]]
    theta_target: dict[str, NDArray[float]]   # slow EMA copy
```

## 1.7 Session frontier

定义前沿摘要：

[
f_t = \sum_{i \in \mathcal F_t}\omega_i \mu_i,
\qquad
\omega_i \propto a_i^{prev}\exp(-\Delta t_i/\tau_f)(1+u_i^{unresolved})
]

`session frontier` 不是“最近 N 条 transcript”，而是**上一个 workspace 中仍未闭合的活跃痕迹边界**。

---

# 2. 目标函数与更新规则

Aurora v2 采用**双目标耦合**：

* 在线结构决策由局部场能量 (E_{loc}) 决定
* 慢系统学习由预测损失 (L_{slow}) 决定

总目标：

[
J_t = \mathbb E[E_{loc}(x_t,c_t;M_t)] + \lambda_P L_{slow}(\Theta_t) + \lambda_D D_t
]

其中 (D_t) 是防漂移项。

---

## 2.1 在线局部能量 (E_{loc})

当前输入 anchor latent 为 (x_t)，上下文向量为：

[
c_t = P_c[q_t ,|, f_t ,|, \mu_t^\Theta]
]

* (q_t)：当前 cue / anchor latent
* (f_t)：session frontier summary
* (\mu_t^\Theta)：slow predictor 对下一状态的预测均值
* (P_c)：固定或可学习的上下文投影

候选集 (S_t) 上，定义 trace 的上下文先验：

[
\log \pi_i(c_t)
\propto
\underbrace{\log(e_i+\epsilon) + \gamma_s s_i - \gamma_u u_i}*{\text{base prior}}
-\frac12 |c_t-\nu_i|^2*{\Lambda_i^{-1}}
-\frac12 \log |\Lambda_i|
]

然后定义当前 anchor 在局部 trace 混合下的 surprise：

[
\ell_{loc}(x_t,c_t)
===================

-\log \sum_{i\in S_t\cup{0}}
\pi_i(c_t),
\mathcal N(x_t;\mu_i,\Sigma_i)
]

这里 (0) 是 `NULL / birth` 槽，使用宽方差先验。

局部能量写成：

[
E_{loc}(x_t,c_t;M_t)
====================

\ell_{loc}(x_t,c_t)
+
\lambda_B B_{loc}
+
\lambda_K K_{loc}
+
\lambda_S S_{loc}
]

其中：

* (B_{loc})：局部存储开销
* (K_{loc})：局部拓扑复杂度
* (S_{loc})：对高稳定 trace 的塑性惩罚

### 三个可实现项

[
B_{loc} = \sum_{i \in S_t} b_i + \sum_{(i,j)\in E_{loc}} b_{ij}
]

[
K_{loc} = \sum_{i \in S_t}\log(1+\deg_i) + \lambda_G |\mathcal G_{loc}|
]

[
S_{loc} = \sum_{i \in S_t} s_i |\Delta \mu_i|_2^2
]

### 每一项的可观测量来自哪里

* (x_t)：`Anchor.z`
* (q_t)：当前 cue 或注入 anchor 的 latent
* (f_t)：`frontier_summary()`
* (\mu_t^\Theta)：`predictor.peek()`
* (\mu_i,\Sigma_i,\nu_i,\Lambda_i,e_i,s_i,u_i)：`TraceRecord`
* (b_i,b_{ij})：实际字节数或标准化 storage units
* (\deg_i)：trace edge count
* (|\mathcal G_{loc}|)：局部命中的 posterior groups 个数

---

## 2.2 慢系统损失 (L_{slow})

MVP 下，predictor 是一个小型 gated linear dynamical model：

[
h_t = (1-g_t)\odot h_{t-1} + g_t \odot \tanh(W_w w_{t-1} + W_f f_{t-1} + W_a a_{t-1} + W_d d_{\Delta t} + b)
]

[
g_t = \sigma(U_w w_{t-1} + U_f f_{t-1} + U_a a_{t-1} + U_d d_{\Delta t} + c)
]

[
\mu_t^\Theta = V_\mu h_t,
\qquad
\log \sigma_t^{2,\Theta} = V_\sigma h_t
]

其中：

[
w_{t-1} = \sum_i a_i^{t-1}\mu_i
]

预测损失：

[
L_{slow}
========

\mathbb E_{\mathcal B}
\left[
\frac12 |x_t-\mu_t^\Theta|^2_{(\Sigma_t^\Theta)^{-1}}
+
\frac12 \log |\Sigma_t^\Theta|
\right]
+
\lambda_R |\Theta_t - \Theta_t^{-}|_2^2
]

* (\Theta^{-}) 是 predictor 的 EMA target，用来抑制慢系统漂移。

---

## 2.3 在线 (\Delta J) 近似

在线注入时，不即时训练 predictor，因此比较 proposal 时把 (L_{slow}) 当常数项，只比较：

[
\widehat{\Delta J}_\alpha
=========================

# \widehat{\Delta E}_{loc,\alpha}

\Delta \ell_{loc,\alpha}
+
\lambda_B \Delta B_\alpha
+
\lambda_K \Delta K_\alpha
+
\lambda_S \Delta S_\alpha
]

也就是：

[
\widehat{\Delta J}_\alpha
=========================

## E_{loc}(M^\alpha_{loc};x_t,c_t)

E_{loc}(M_{loc};x_t,c_t)
]

这里 (M^\alpha_{loc}) 是**只对局部邻域做一次假想更新**后的状态。
这保证在线复杂度是 (O(Kd)) 到 (O(Kd + |E_{loc}|))，不需要离线全局重算。

---

# 3. proposal / posterior / replay / prototype / procedure / workspace 方程与伪代码

---

## 3.1 proposal 动力学

### 3.1.1 五种动作的假想更新

### ASSIMILATE(i)

[
\eta_i = \frac{\eta_\mu}{1+\lambda_s s_i}
]

[
\mu_i' = \mu_i + \eta_i(x_t-\mu_i)
]

[
\Sigma_i' = (1-\eta_i)\Sigma_i + \eta_i,\mathrm{diag}((x_t-\mu_i)^2)
]

[
\nu_i' = \nu_i + \eta_c(c_t-\nu_i)
]

[
\Lambda_i' = (1-\eta_c)\Lambda_i + \eta_c,\mathrm{diag}((c_t-\nu_i)^2)
]

[
e_i' = \rho_e e_i + 1
]

### ATTACH(i)

只把当前 anchor 作为该 trace 的支持证据，不移动内容中心：

[
\mu_i' = \mu_i,\qquad \Sigma_i' = \Sigma_i
]

[
\nu_i' = \nu_i + \eta_c(c_t-\nu_i),\qquad
e_i' = \rho_e e_i + 1
]

ATTACH 的意义是：**“这个 anchor 支持这个 trace，但不值得改写这个 trace 的核心 latent。”**

### BIRTH

创建新 trace (j)：

[
\mu_j = x_t,\qquad \Sigma_j = \Sigma_0
]

[
\nu_j = c_t,\qquad \Lambda_j = \Lambda_0
]

[
e_j = e_0,\qquad s_j=s_0
]

### SPLIT(i)

在 trace (i) 附近分裂出子 trace (j)：

[
\mu_j = x_t,\qquad \Sigma_j = \min(\Sigma_0,\xi \Sigma_i)
]

[
\nu_j = \nu_i,\qquad \Lambda_j = \Lambda_i
]

[
e_j = e_0,\qquad s_j=s_0
]

同时继承父 trace 的 top temporal/assoc 边：

[
w_{jn} = \rho_{inh} w_{in}
]

SPLIT 不移动父 trace，所以 (\Delta S) 很低。

### INHIBIT(i,j)

INHIBIT 是**二级动作**，通常在 primary action 之后判断。
它不解决当前 anchor 的归属，而是给未来 readout 建竞争边。

在线维护 pairwise 分离证据：

[
BF_{sep}(i,j|x_t,c_t)
=====================

\log\Big[
\pi_i(c_t)\mathcal N(x_t;\mu_i,\Sigma_i)
+
\pi_j(c_t)\mathcal N(x_t;\mu_j,\Sigma_j)
\Big]
-----

\log\Big[
\pi_{ij}(c_t)\mathcal N(x_t;\mu_{ij},\Sigma_{ij})
\Big]
]

其中 merged surrogate：

[
\mu_{ij} = \frac{e_i\mu_i + e_j\mu_j}{e_i+e_j}
]

[
\Sigma_{ij}
===========

\frac{
e_i(\Sigma_i + (\mu_i-\mu_{ij})^2)
+
e_j(\Sigma_j + (\mu_j-\mu_{ij})^2)
}{
e_i+e_j
}
]

维护 EMA：

[
BF^{ema}*{ij} \leftarrow \rho*{bf} BF^{ema}*{ij} + (1-\rho*{bf}) BF_{sep}(i,j|x_t,c_t)
]

INHIBIT 的局部增量近似写成：

[
\widehat{\Delta J}_{inh}(i,j)
=============================

-\lambda_H \max(0,BF^{ema}*{ij})
+
\lambda_B b*{edge}
+
\lambda_K \Delta K_{edge}
]

若为负，则添加或加强 inhibit edge：

[
w^-*{ij} \leftarrow \rho*- w^-*{ij} + \eta*- \max(0,BF^{ema}_{ij})
]

---

## 3.1.2 什么时候选哪一个动作

设 best candidate 为 (i^*)。

* **ASSIMILATE**：上下文拟合好、内容拟合也好，而且 drift penalty 不大
* **ATTACH**：内容拟合还行，但 `stability` 很高，移动中心不划算
* **SPLIT**：上下文拟合好，但内容残差大，说明“还是这条因果脉络，但已经不是同一个 trace”
* **BIRTH**：上下文也不好、内容也不好，说明新事件/新原因
* **INHIBIT**：当前 cue 下两个旧 trace 持续争夺解释权，且“分开解释”长期优于“合并解释”

可以把它压成一句话：

* `ASSIMILATE = same content, same context`
* `SPLIT = new content, same context`
* `BIRTH = new content, new context`
* `ATTACH = same trace, no plastic move`
* `INHIBIT = same cue, mutually exclusive explanations`

---

## 3.1.3 在线 proposal 伪代码

```python
def score_primary_actions(anchor, frontier, predictor):
    x = anchor.z
    c = make_context(anchor.z, frontier.summary(), predictor.mu)
    S = make_candidates(x, c)  # hot + frontier nbrs + ANN + group members

    base = local_energy(S, x, c)
    best = ("BIRTH", None, simulate_birth(S, x, c) - base)

    for i in S:
        best = min(best, ("ASSIMILATE", i, simulate_assimilate(S, i, x, c) - base), key=lambda t: t[2])
        best = min(best, ("ATTACH", i, simulate_attach(S, i, x, c) - base), key=lambda t: t[2])

        if split_allowed(i):
            best = min(best, ("SPLIT", i, simulate_split(S, i, x, c) - base), key=lambda t: t[2])

    return best
```

```python
def maybe_inhibit(S, x, c):
    i, j = top2_ambiguous_components(S, x, c)
    delta = inhibit_delta(i, j, x, c)
    if delta < 0:
        add_or_strengthen_inhibit(i, j)
        maybe_form_or_update_group(i, j, x, c)
```

---

## 3.2 posterior dynamics

## 3.2.1 冲突组的形成

pair ((i,j)) 形成冲突组需要同时满足：

1. `BF_sep^ema(i,j)` 持续为正
2. 同一类上下文反复共激活
3. merged surrogate 对这些样本解释更差

形式化地：

[
\text{create group if }
BF^{ema}*{ij} > \tau*{bf}
\quad\land\quad
\mathrm{ctx_overlap}*{ij} > \tau*{ctx}
]

已有冲突边的连通分量可并成多成员 group。

---

## 3.2.2 posterior group 的状态变量

group (g) 对应离散隐变量 (Y_g \in {1,\dots,K_g,0})：

* (1\dots K_g)：各成员 trace
* (0)：NULL unresolved slot

group 维护：

* (\alpha_{gk})：member 的 decayed evidence count
* ((\nu_{gk},\Lambda_{gk}))：该 member 常见上下文
* `pred_success_ema[k]`
* `temperature`
* `unresolved_mass`
* `ambiguous_buffer`

---

## 3.2.3 posterior 更新方程

给定当前 observation (x_t) 与 context (c_t)：

[
\log \tilde{\pi}_{gk}(t)
========================

## \log(\alpha_{gk}+\epsilon)

## \frac12 |c_t-\nu_{gk}|^2_{\Lambda_{gk}^{-1}}

\frac12 |x_t-\mu_k|^2_{\Sigma_k^{-1}}
+
\lambda_p \log(\epsilon + \mathrm{predsucc}_{gk})
]

[
\pi_{gk}(t) = \mathrm{softmax}*k\left(\frac{\log \tilde{\pi}*{gk}(t)}{T_g}\right)
]

NULL slot 用宽上下文、高方差发射模型：

[
p(x_t,c_t|Y_g=0) = \mathcal N(x_t;\mu_0,\Sigma_0)\mathcal N(c_t;\nu_0,\Lambda_0)
]

---

## 3.2.4 posterior 的在线更新

[
\alpha_{gk} \leftarrow \rho_\alpha \alpha_{gk} + \pi_{gk}(t)
]

[
\nu_{gk} \leftarrow \nu_{gk} + \eta_\nu \pi_{gk}(t)(c_t-\nu_{gk})
]

[
\Lambda_{gk} \leftarrow (1-\eta_\Lambda \pi_{gk})\Lambda_{gk} + \eta_\Lambda \pi_{gk},\mathrm{diag}((c_t-\nu_{gk})^2)
]

[
\mathrm{predsucc}*{gk}
\leftarrow
\rho_p \mathrm{predsucc}*{gk}
+
(1-\rho_p)\pi_{gk}(t)\exp(-\ell^{pred}_{gk})
]

[
T_g \leftarrow \mathrm{clip}\big(T_g + \eta_T(H(\pi_g)-H^*)\big, T_{min}, T_{max})
]

[
m_{g0} \leftarrow \rho_m m_{g0} + \pi_{g0}(t)
]

---

## 3.2.5 split 还是 uncertainty increase

这是用 NULL 槽决定的。

### 只增加 uncertainty，不 split

若：

* (\pi_{g0}) 高
* 但 unresolved buffer 不紧、上下文不一致

则只增加 group 温度和成员不确定性：

[
\Sigma_k \leftarrow \Sigma_k + \eta_\Sigma \pi_{gk}(1-\pi_{gk})I
]

[
T_g \leftarrow T_g + \eta_T \pi_{g0}
]

### 触发 split

若：

[
m_{g0} > \tau_m
\quad\land\quad
\mathrm{tr}(\mathrm{Cov}(B_{g0})) < \tau_{compact}
\quad\land\quad
\mathrm{ctx_coh}(B_{g0}) > \tau_{coh}
]

则对 unresolved buffer 建一个新 child trace：

[
\mu_{new} = \mathrm{mean}(B_{g0}),
\qquad
\nu_{new} = \mathrm{mean_ctx}(B_{g0})
]

然后把它加入 group。

这就把“split 还是 uncertainty”从启发式，变成了**基于 NULL mass 的在线非参数 mixture 决策**。

---

## 3.2.6 readout 时如何读取 conflict group

在 workspace settle 中，对每个 group 做一次 posterior slice projection：

1. 先算 group 总激活：
   [
   m_g = \sum_{i\in g} a_i
   ]

2. 按当前 cue/context 算 posterior slice：
   [
   \pi_g(c_t)
   ]

3. 再把 group mass 重新分配：
   [
   a_i \leftarrow m_g \cdot \pi_{gi}(c_t)
   ]

这意味着：
**group 决定“谁和谁在争同一个位置”，posterior slice 决定“当前这次争夺谁赢”。**

---

## 3.3 replay 与 slow predictor

## 3.3.1 replay 分布

replay 不是简单 signal-driven sampling，而是近似采样对 (J) 梯度贡献最大的条目。

对 trace (i)，定义 priority：

[
\rho_i
======

(\epsilon+\bar{\ell}^{pred}*i)^{\alpha*\ell}
(\epsilon+u_i)^{\alpha_u}
(\epsilon+H_{g(i)})^{\alpha_g}
(\epsilon+\mathrm{centrality}_i)^{\alpha_c}
(\epsilon+\mathrm{forgetrisk}_i)^{\alpha_f}
]

然后：

[
p_{replay}(i)=\frac{\rho_i}{\sum_j \rho_j}
]

MVP 直接用三源混合：

* 60%：single trace replay
* 20%：conflict-group ambiguous replay
* 20%：short path / procedure-candidate replay

---

## 3.3.2 predictor 的最小形式

**首选 MVP**：上面的 gated linear model。
**更低算力 fallback**：

[
\mu_{t+1}^\Theta = A w_t + B f_t + C a_t + b
]

[
\Sigma_{t+1}^\Theta = \mathrm{diag}(\sigma_0^2)
]

这已经足够让慢系统从 replay 中吸收长期结构。

---

## 3.3.3 predictor 的训练目标

[
L_{slow}
========

\mathbb E
\left[
\frac12 |x_{t+1}-\mu_{t+1}^\Theta|^2_{(\Sigma^\Theta)^{-1}}
+
\frac12\log|\Sigma^\Theta|
\right]
+
\lambda_R |\Theta-\Theta^{-}|^2
]

在线与后台协同方式：

* **在线**：只做 `peek()`，必要时做 1 次 micro-step
* **后台**：replay batch 上做 1~N 次 SGD / Adam
* `theta_target` 用 EMA 更新

---

## 3.3.4 reconsolidation 更新

对 replay 样本 ((x,c)) 和负责它的 trace (i)：

[
e_i \leftarrow \rho_e e_i + r_i
]

[
\mu_i \leftarrow \mu_i + \eta_z \frac{r_i}{1+\lambda_s s_i}(x-\mu_i)
]

[
\Sigma_i \leftarrow (1-\eta_\Sigma r_i)\Sigma_i + \eta_\Sigma r_i,\mathrm{diag}((x-\mu_i)^2)
]

[
\bar{\ell}^{pred}*i \leftarrow \rho*\ell \bar{\ell}^{pred}*i + (1-\rho*\ell)\ell^{pred}_i
]

[
u_i
===

\frac{1}{d}\mathrm{tr}(\Sigma_i)
+
\frac{\kappa_e}{e_i+\epsilon}
+
\kappa_p \bar{\ell}^{pred}_i
]

[
s_i
\leftarrow
(1-\eta_s)s_i
+
\eta_s \sigma!\big(\log(e_i+\epsilon)-u_i-\bar{\ell}^{pred}_i\big)
]

这几个量的含义很清楚：

* `evidence`：被 replay/online 支持多少
* `latent`：被拉向长期解释
* `uncertainty`：来自 dispersion + lack of evidence + predictive instability
* `stability`：被 evidence 拉高，被 uncertainty / pred loss 拉低

---

## 3.3.5 replay 如何真正改变长期结构

边不是手工规则，而是 replay 协方差诱导出来的：

[
w^+*{ij}
\leftarrow
\rho*+ w^+*{ij}
+
\eta*+ \mathbb E[r_i^{(t)} r_j^{(t+1)}]
]

[
w^-*{ij}
\leftarrow
\rho*- w^-*{ij}
+
\eta*- \mathbb E[\mathbf 1(\text{same cue}), r_i^{(t)}r_j^{(t)} \max(0,BF_{sep})]
]

[
w^{option}*{ij}
\leftarrow
\rho_o w^{option}*{ij}
+
\eta_o \mathbb E[\mathbf 1(\text{successful path } i\rightarrow j)]
]

也就是说，replay 不只是“改分数”，而是直接改：

* trace latent
* trace uncertainty/stability
* temporal/assoc/inhib/option edges
* posterior groups
* role promotion / demotion

---

## 3.4 prototype / procedure 的正式判据

Aurora 保持单一 trace 宇宙。
prototype / procedure 不是新类型，只是 `TraceRecord.role_logits` 的不同主导区。

---

### 3.4.1 prototype 形成条件

对 replay 中重复共激活的一组 traces (C)，构造候选 prototype latent：

[
\mu_C = \frac{\sum_{i\in C} a_i \mu_i}{\sum_{i\in C} a_i}
]

dispersion：

[
disp(C)=\frac{\sum_{i\in C} a_i |\mu_i-\mu_C|^2}{\sum_{i\in C} a_i}
]

compression gain：

[
\Delta J_{proto}(C)
===================

\underbrace{\ell_{members}(C)-\ell_{proto}(C)}*{\text{fit gain}}
+
\lambda_B \Delta B*{proto}
+
\lambda_K \Delta K_{proto}
]

promotion trigger：

[
support_C > \tau_{sup}^{proto}
\quad\land\quad
disp(C) < \tau_{disp}
\quad\land\quad
EMA(\Delta J_{proto}(C)) < -\tau_g
]

evidence accumulation：

[
support_C \leftarrow \rho_s support_C + \sum_{i\in C} a_i
]

[
r^{proto}_C \leftarrow \rho_r r^{proto}*C + (1-\rho_r)(-\Delta J*{proto}(C))
]

当 `support` 和 `r_proto` 同时越过阈值，就把某个 trace 或新 child trace 的 `role_logits["prototype"]` 拉高，并把 `member_ids=C`。

downgrade / pruning：

* `support_C < τ_low`
* `EMA(ΔJ_proto) > 0` 持续 T 个维护窗口
* budget 压力高于收益

则 prototype role 降级，必要时 dissolve 为普通 episodic trace。

---

### 3.4.2 procedure 形成条件

对 replay 中重复出现的 path (P=(i_1,a_1,i_2,\dots,i_m))，维护：

* support (N_P)
* 成功 Beta 统计 ((\alpha_P,\beta_P))
* transition entropy (H_P)
* multi-step gain (\Delta J_{proc}(P))

成功率：

[
\mathbb E[\mathrm{succ}_P] = \frac{\alpha_P}{\alpha_P+\beta_P}
]

transition entropy：

[
H_P = -\sum_j p(j|P)\log p(j|P)
]

promotion trigger：

[
N_P > \tau_N
\quad\land\quad
\mathbb E[\mathrm{succ}*P] > \tau*{succ}
\quad\land\quad
H_P < \tau_H
\quad\land\quad
EMA(\Delta J_{proc}(P)) < -\tau_g
]

evidence accumulation：

[
\alpha_P \leftarrow \rho_\alpha \alpha_P + \mathbf 1[\text{success}]
]

[
\beta_P \leftarrow \rho_\beta \beta_P + \mathbf 1[\text{failure}]
]

[
r^{proc}_P \leftarrow \rho_r r^{proc}*P + (1-\rho_r)(-\Delta J*{proc}(P))
]

成功后，将一个 trace 的 `role_logits["procedure"]` 拉高，并填入 `path_signature=P`。

downgrade 规则：

* (N_P) 下降
* success posterior 下降
* (H_P) 上升
* (EMA(\Delta J_{proc}) > 0)

则 procedure role 退化，必要时仅保留 option edges。

---

## 3.5 workspace readout：正式场方程

这一步直接用 **Hopfield/attention 风格的能量读出**。([arXiv][2])

定义 candidate set (S_t) 上的激活 (a \in \mathbb R_+^{|S_t|})，workspace energy：

[
E_{ws}(a)
=========

*

## b_t^\top a

\frac12 a^\top W^+ a
+
\frac12 a^\top W^- a
+
\lambda_G \sum_g
\mathrm{KL}!\left(
\frac{a_g}{|a_g|_1+\epsilon}
\Big|
\pi_g(c_t)
\right)
+
\lambda_1 |a|_1
+
\lambda_0 |a|_0
]

其中：

* (b_t = B_q q_t + B_f f_t + B_p \mu_t^\Theta)
* (W^+)：temporal + assoc + option 正向边
* (W^-)：inhib 边
* 第三个项：把冲突组内激活分布拉向当前 posterior slice
* (\lambda_1,\lambda_0)：稀疏正则

### 数值近似

离散化成阻尼更新：

[
u^{k+1}
=======

(1-\eta)u^k
+
\eta\left(
b_t + W^+a^k - W^-a^k
\right)
]

[
a^{k+1}
=======

\Pi_K\left[
\mathrm{entmax}_\alpha\left(\frac{u^{k+1}}{T}\right)
\right]
]

然后做一次 group projection：

[
m_g = \sum_{i\in g} a_i
\qquad\Rightarrow\qquad
a_i \leftarrow m_g \cdot \pi_{gi}(c_t)
]

### sparse approximation strategy

为不破坏理论一致性，稀疏近似分两层：

1. **candidate sparsification**
   候选集只取：

   * hot traces
   * frontier neighbors
   * ANN near cue
   * 所有命中的 posterior group members
   * active prototype/procedure traces

2. **support projection**
   (\Pi_K) 是对 (K)-sparse feasible set 的投影，可理解为对 (\lambda_0|a|_0) 的近似 proximal step，而不是随意 top-k 裁剪。

### 收敛条件

若：

[
\eta(|W^+|_2 + |W^-|_2) < 1
]

并且每轮都做阻尼与投影，则这个映射在候选集上是 non-expansive 的。
工程上直接用：

* `max_iter = 6~12`
* `stop if ||a^{k+1}-a^k||_1 < eps`

即可。

---

# 4. 数据结构草案

下面这些是我建议直接补到代码里的字段。

## 4.1 `core/types.py`

### 需要新增到 `TraceRecord`

* `z_sigma_diag`
* `ctx_mu`
* `ctx_sigma_diag`
* `pred_loss_ema`
* `role_logits`
* `member_ids`
* `path_signature`
* `posterior_group_ids`

### 新增 `PosteriorGroup`

```python
@dataclass
class PosteriorGroup:
    group_id: str
    member_ids: list[str]
    alpha: NDArray[float]              # includes NULL slot
    ctx_mu: NDArray[float]
    ctx_sigma_diag: NDArray[float]
    pred_success_ema: NDArray[float]
    temperature: float
    unresolved_mass: float
    ambiguous_buffer: deque[str]
```

### 新增 `PredictorState`

```python
@dataclass
class PredictorState:
    h: NDArray[float]
    theta: dict[str, NDArray[float]]
    theta_target: dict[str, NDArray[float]]
```

---

## 4.2 `runtime/field.py` 要新增的方法

* `_make_context(q, frontier, pred_mu)`
* `_make_candidates(q, c)`
* `_objective_terms(S, x, c, pred)`
* `_score_action_candidate(action, trace_id, anchor, c, candidates, pred)`
* `_score_primary_actions(anchor, pred)`
* `_bf_separate(i, j, x, c)`
* `_apply_inhibit_pair(pair, x, c, ...)`
* `_posterior_slice(group, x, c)`
* `_replay_groups_from_batch(batch)`
* `_fidelity_step()`
* `_role_lifecycle_step()`
* `_settle_workspace(cue, context, pred_mu, session_id)`

---

## 4.3 `runtime/system.py` 要新增的方法

`runtime/system.py` 仍保持外层编排层；慢系统、proposal、budget 与 role lifecycle 已全部下沉到 `runtime/field.py`。

---

# 5. 最小可实现版本

这是**不依赖大模型训练**、且可以在现有骨架上直接落地的版本。

## MVP 约束

* latent 直接复用现有 encoder
* 所有 covariance 都用 diagonal
* predictor 用 gated linear model；算力更差时退成线性模型
* workspace 用 `entmax + projected top-k`
* prototype / procedure 不新建 class，只改 `role_logits`
* 不实现 HDC/VSA 并行编码
* 不实现 decoder cross-attention memory tokens
* 不实现 wake-sleep negative phase

## 默认超参建议

* `candidate_size K0 = 64`
* `workspace_size K = 16`
* `settle_steps = 8`
* `entmax_alpha = 1.5`
* `predictor_hidden = 128`，低算力可设为 0 退化成线性
* `replay batch = 32`
* `trace/conflict/path replay mix = 0.6 / 0.2 / 0.2`
* `theta_target_ema = 0.995`

## 实现顺序

### 第一步：proposal + posterior + workspace

先把这三个变正式：

* `local_energy`
* `primary action simulation`
* `posterior group + NULL slot`
* `workspace settle`

这一步完成后，Aurora 就已经不是 heuristic trace engine 了。

### 第二步：replay + predictor + reconsolidation

然后补：

* prioritized replay
* slow predictor
* trace/edge updates from replay
* posterior updates during replay
* replay-driven structural objective over activation drift, transition support, and group heat
* replay-driven structural mutation acceptance for deferred `BIRTH / SPLIT`

### 第三步：prototype / procedure role transition

最后补：

* prototype promotion/demotion
* procedure promotion/demotion
* option edge strengthening
* effective-mass budget with fidelity-aware downgrade

---

# 6. 验证实验设计

## 6.1 Proposal 正确性实验

构造 4 类合成流：

1. same content / same context
2. new content / same context
3. new content / new context
4. same cue / mutually exclusive traces

验证 online action 是否与小窗口 brute-force `argmin local_energy` 一致。

**指标**：

* action agreement
* online regret ( \widehat{\Delta J}*{chosen} - \widehat{\Delta J}*{best} )

---

## 6.2 Posterior calibration 实验

构造“同一属性多版本”任务，例如：

* 旧偏好 vs 新偏好
* 不同 session 中的不同身份设定
* 工具状态冲突

验证：

* ECE / Brier score
* NULL slot 使用率
* split vs uncertainty 的触发正确率

---

## 6.3 Replay / predictor 实验

构造 repeated motif + occasional surprise 的长序列。

验证：

* predictor NLL 是否持续下降
* replay 后 `stability` 是否上升
* `uncertainty` 是否在被反复支持后下降
* edge structure 是否变得更稀疏但更可预测

---

## 6.4 Prototype / procedure 实验

### prototype

给多个变体 episode，测试是否形成 prototype-dominant trace。

指标：

* compression gain
* prototype support
* prototype readout recall

### procedure

给重复 state-action-outcome 轨迹。

指标：

* procedure promotion rate
* success posterior
* multi-step prediction loss
* mean planning length reduction

---

## 6.5 Workspace 场读出实验

关闭 transcript retrieval，只保留：

* cue
* frontier
* memory field readout

验证：

* settle 是否在 8 步内收敛
* `||a^{k+1}-a^k||_1`
* conflict group slice 是否随 context 改变
* response continuity 是否优于 heuristic weighted settle baseline

---

## 6.6 Budget stress test

在 1.0x / 0.3x / 0.1x 预算下测试：

* trace count
* edge count
* posterior group count
* near-verbatim recall
* long-term gist retention
* task success degradation curve

目标不是零退化，而是**平滑退化，不 cliff 崩盘**。

---

# 次一级 phase（不阻塞当前内核）

1. **HDC/VSA 并行表征**
   作为 `Anchor.z_hv` 与 `TraceRecord.z_hv_proto` 的第二通道，用于更强的 binding / cleanup；但它是增强件，不是当前内核依赖。([arXiv][3])

2. **latent-conditioned generation / memory tokens**
   把 workspace summary 和 posterior slice 直接做成 decoder cross-attention 的 memory tokens。

3. **wake-sleep negative phase**
   在 background 中对低支持浅吸引子做负相抬能，用于进一步平滑伪吸引子；但这应该在 replay + budget controller 稳定后再上。([CVF开放获取][4])

---

# 一句话总结

**Aurora v2 的核心不是“更聪明的 heuristic trace engine”，而是：用在线局部能量 (E_{loc}) 决定 proposal，用显式 posterior group 处理冲突，用 replay + predictor 形成慢系统，用 role transition 形成 prototype/procedure，再用一个带抑制与 posterior slice 的场方程读出 workspace，从而把系统升级成真正预算约束下的统一 trace-field memory。**

当前实现继续沿这条路线推进：replay 侧会累计 trace/group 的 continuation 统计，把 future alignment / future drift 显式并入 objective；online proposal 与 replay structural mutation 已经开始共享 finite empirical block objective；workspace settle 已经采用带 backtracking 的 energy descent，并输出 energy trace；deferred structural mutation 也受 replay-driven objective 控制，而不是再退回到独立的启发式门控；maintenance 的 `ms_budget` 也已经真正进入 replay 采样，而不再只是一个表面参数。

[1]: https://www.sciencedirect.com/science/article/pii/S037015732300203X?utm_source=chatgpt.com "The free energy principle made simpler but not too simple"
[2]: https://arxiv.org/abs/2008.02217?utm_source=chatgpt.com "Hopfield Networks is All You Need"
[3]: https://arxiv.org/pdf/2111.06077?utm_source=chatgpt.com "A Survey on Hyperdimensional Computing aka Vector Symbolic ..."
[4]: https://openaccess.thecvf.com/content/CVPR2024W/CLVISION/papers/Singh_Wake-Sleep_Energy_Based_Models_for_Continual_Learning_CVPRW_2024_paper.pdf?utm_source=chatgpt.com "Wake-Sleep Energy Based Models for Continual Learning"
