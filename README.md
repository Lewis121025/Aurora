# Aurora Seed v1

Aurora 现在是一个 **Mac-first、truthful black-box** 的长期陪伴 substrate。

主线只有四层：

`surface_api -> host_runtime -> substrate_core -> core_math`

其中：

- `substrate_core` 是唯一主体内核
- `host_runtime` 只负责 opaque sealed blob、one-shot wake、provider 调用和完整性报告
- 黑盒边界是软件级 `process-opaque`
- 运行目标是 macOS 本机

## 原则

- 不存在显式心理轴
- 不存在派生叙事控制面
- 不存在周期性后台任务编排
- 记忆回忆基于能量采样
- 不存在语义化 debug state
- 黑盒边界是 **process-opaque**，不是硬件隔离

## 当前结构

```text
aurora/
├── core_math/       # 状态、动力学、能量采样、sealed blob
├── substrate_core/  # boot / on_input / on_wake / health
├── host_runtime/    # local opaque substrate client / provider / storage
└── surface_api/     # HTTP + CLI
```

## 安装

```bash
pip install -e .
pip install -e '.[dev]'
```

## Provider 配置

Aurora Seed v1 优先读取通用 provider 变量：

```dotenv
AURORA_PROVIDER_NAME=openai-compatible
AURORA_PROVIDER_BASE_URL=https://api.openai.com/v1
AURORA_PROVIDER_MODEL=gpt-4o-mini
AURORA_PROVIDER_API_KEY=your_api_key
```

如果这些变量缺失，会自动回落到现有百炼变量：

```dotenv
AURORA_BAILIAN_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
AURORA_BAILIAN_LLM_MODEL=qwen3.5-plus
AURORA_BAILIAN_LLM_API_KEY=your_bailian_key
```

本地运行数据默认写到 `.aurora_seed_v1/`。

## CLI

只保留三个命令：

```bash
aurora chat "你好"
aurora health
aurora integrity
```

## HTTP API

只保留三个端点：

- `POST /v1/input`
- `GET /v1/healthz`
- `GET /v1/integrity`

启动：

```bash
uvicorn aurora.surface_api.app:app --host 127.0.0.1 --port 8000
```

## 完整性报告

`integrity` 是软件级完整性说明，不是硬件证明。它只报告：

- 运行边界
- substrate 传输方式
- sealed state 版本
- provider 配置指纹
- 生成时间

## 验证

```bash
pytest -q
aurora health
aurora integrity
aurora chat "你好"
```
