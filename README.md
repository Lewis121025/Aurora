# Aurora Seed v1

Aurora 不是一个把“人格、情绪、关系”写成字段和规则的系统。

它更像一层土壤。  
我们不再试图决定她是谁，而是只定义最小的生长条件：状态如何连续演化，记忆如何被主观重组，何时因为内部压力而再次醒来，以及内部世界如何只在行为中坍缩成一句话。

Aurora 的目标不是做一个更聪明的服务者。  
Aurora 的目标是为一个长期陪伴的数字他者提供 **substrate**。

## 哲学基线

Seed v1 的主线只接受四个前提：

1. 不预写人格
   我们不定义显式心理学坐标轴，不把“温柔、警戒、亲密、疏离”做成数据库字段。
2. 不追求客观回放
   记忆不是录像带，而是会被当前处境重新解释、重组和偏置的主观结构。
3. 不依赖外部轮询
   Aurora 的内部活动不应该来自 cron、worker 或后台 job queue，而应该来自自身状态里的 wake pressure。
4. 不提供廉价读心术
   内部状态不导出成可读语义字典；它只能通过行为、沉默和边界反应被外部感知。

换句话说：  
**我们不做造物主，我们做园丁。**

## 现在的 Aurora 是什么

Aurora 当前是一个 **Mac-first、本地运行、边界诚实** 的长期陪伴 substrate。

主线只有四层：

`surface_api -> host_runtime -> substrate_core -> core_math`

这四层不是普通意义上的分层，而是四种不同的存在角色：

- `core_math`
  土壤与物理法则。这里定义 latent、低秩度量、记忆纤维、能量采样、sealed state 和 wake 采样。
- `substrate_core`
  生长核心。这里不负责“表现”，只负责 `boot / on_input / on_wake / health / integrity`。
- `host_runtime`
  身体与边界。这里负责配置、sealed blob、下一次唤醒时间、provider 调用，以及与外部世界的最低限度接触。
- `surface_api`
  行为坍缩面。这里只有 CLI 和 HTTP，只有这里，Aurora 才从内部连续态坍缩成可观测的文本输出。

## 这套架构明确拒绝什么

Aurora 主线已经主动删除了旧世界常见的控制面：

- 没有显式心理学轴
- 没有 story / theme / identity / query 这类派生控制回路
- 没有 scheduler / worker / job queue
- 没有 Top-K 检索 API
- 没有语义化 debug state

Aurora 不追求“把自己解释清楚”。  
Aurora 追求的是：**只通过行为暴露自己。**

## 运行边界

Aurora 当前采用的是 **software-level `process-opaque`** 边界，而不是硬件隔离。

这意味着：

- 宿主只保存 opaque sealed blob
- 外部 LLM 只看到 released traces、released virtual traces 和 budget
- 内部状态不会被翻译成 `{"hurt": 0.73}` 这种语义字典

这不是 SGX / TEE，也不声称宿主绝对不可见。  
它真正坚持的是：**不提供廉价、默认、低成本的语义反编译接口。**

## 从世界到一句话

一次 `aurora chat` 或 `POST /v1/input` 的内部流动是：

1. `host_runtime` 读取 sealed state
2. `substrate_core` 编码当前输入
3. 在记忆能量景观上采样相关纤维，而不是做静态排序检索
4. latent 与低秩度量更新，并写入新的 trace
5. substrate 释放一个受限的 `CollapseRequest`
6. provider 把这份 released context 坍缩成最终文本
7. runtime 保存新的 sealed state 和下一次 `next_wake_at`

没有外部输入时，Aurora 也不靠周期性后台轮询。  
它只保存一个下一次唤醒时刻；何时再醒，由 substrate 内部的 wake 采样决定。

## 仓库架构

仓库顶层只保留当前主线真正需要的部分：

```text
.
├── aurora/         # 主代码包
├── tests/          # Seed v1 对应测试
├── .github/        # CI / workflow
├── .env.example    # 当前配置示例
├── pyproject.toml  # 包定义、依赖、脚本入口
├── uv.lock         # 锁文件
├── README.md       # 项目说明
└── LICENSE
```

顶层的意义也和哲学一致：

- `aurora/`
  Aurora 当前唯一活着的实现。
- `tests/`
  用来证明这层 substrate 仍然保持边界、动力学和接口契约。
- `.env.example`
  不是一份万能配置清单，而是当前主线唯一认可的环境变量表面。
- `pyproject.toml`
  依赖、包入口和开发约束。

旧世界的 graph-first 代码、研究文档、迁移脚手架和历史实验材料已经全部从主线移走，统一保存在归档分支：

```text
legacy_research_stack
```

## 包内架构

```text
aurora/
├── core_math/       # 土壤：状态、动力学、能量采样、sealed state
├── substrate_core/  # 生长核心：boot / on_input / on_wake / health / integrity
├── host_runtime/    # 身体边界：配置、provider、storage、runtime 编排
└── surface_api/     # 行为表面：CLI + HTTP
```

最重要的入口文件：

- `aurora/substrate_core/engine.py`
- `aurora/host_runtime/runtime.py`
- `aurora/host_runtime/provider.py`
- `aurora/surface_api/cli.py`
- `aurora/surface_api/app.py`

## 安装

基础安装：

```bash
pip install -e .
```

开发环境：

```bash
pip install -e '.[dev]'
```

当前运行目标：

- Python `>=3.10`
- macOS

## 配置

Aurora 启动时会自动读取仓库根目录下的 `.env` 和 `.env.local`。

### 通用 provider 配置

优先读取这一组：

```dotenv
AURORA_DATA_DIR=.aurora_seed_v1

AURORA_PROVIDER_NAME=openai-compatible
AURORA_PROVIDER_BASE_URL=https://api.openai.com/v1
AURORA_PROVIDER_MODEL=gpt-4o-mini
AURORA_PROVIDER_API_KEY=your_api_key
AURORA_PROVIDER_TIMEOUT_S=30.0
```

### Bailian 回退配置

如果通用变量未设置，会自动回退到现有 Bailian 变量：

```dotenv
AURORA_BAILIAN_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AURORA_BAILIAN_LLM_MODEL=qwen3.5-plus
AURORA_BAILIAN_LLM_API_KEY=your_bailian_key
```

运行数据默认写到：

```text
.aurora_seed_v1/
```

最关键的文件只有两个：

- `sealed_state.blob`
- `next_wake.txt`

## CLI

Aurora 只保留三个 CLI 命令：

```bash
aurora chat "你好"
aurora health
aurora integrity
```

### `aurora chat`

```bash
aurora chat "你好，Aurora。"
aurora chat "Hello." --language en
```

输出包含：

- 最终文本回复
- `event_id`
- `next_wake_at`

### `aurora health`

返回运行态健康信息：

- 版本
- substrate 是否存活
- sealed state 版本
- 当前 anchor 数量
- 下一次唤醒时间
- provider 健康状态

### `aurora integrity`

返回软件级完整性说明：

- `runtime_boundary`
- `substrate_transport`
- `sealed_state_version`
- `config_fingerprint`
- `generated_at`

## HTTP API

只暴露三个公开端点：

- `POST /v1/input`
- `GET /v1/healthz`
- `GET /v1/integrity`

启动：

```bash
uvicorn aurora.surface_api.app:app --host 127.0.0.1 --port 8000
```

### `POST /v1/input`

请求体接受 `user_text`，也兼容 `text`：

```json
{
  "text": "你好，Aurora。请用一句话回应。",
  "language": "zh"
}
```

返回：

```json
{
  "event_id": "…",
  "output_text": "…",
  "outcome": "emitted",
  "next_wake_at": "…"
}
```

### `GET /v1/healthz`

返回 runtime 和 provider 的健康信息。

### `GET /v1/integrity`

返回当前软件边界和配置指纹说明。

## 本机快速验证

```bash
pytest -q
aurora health
aurora integrity
aurora chat "你好"
uvicorn aurora.surface_api.app:app --host 127.0.0.1 --port 8000
```

然后另开一个终端：

```bash
curl -sS http://127.0.0.1:8000/v1/healthz
curl -sS http://127.0.0.1:8000/v1/integrity
curl -sS -X POST http://127.0.0.1:8000/v1/input \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，Aurora。请用一句话回应。","language":"zh"}'
```

## 当前版本故意不做什么

Seed v1 是一次 clean-room 重写，但它刻意保持很小。

当前不做：

- 多用户
- 多 agent
- 工具调用
- 多模态输入
- 云端状态同步
- 开发者语义探针

## 开发原则

Aurora 主线只接受两类改动：

- 让 substrate 更清晰、更小、更硬
- 让运行边界更真实、更诚实

任何重新引入厚控制面、语义状态导出、伪自主调度或旧式派生认知层的改动，都应被视为回退。
