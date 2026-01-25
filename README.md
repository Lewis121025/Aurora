# AURORA 叙事记忆系统

> **零硬编码、纯概率驱动的 Agent 记忆系统**

AURORA (Adaptive Uncertainty-Reducing, Resource-Optimal, Recursive Autobiographical) 是一个基于叙事心理学原理的 Agent 记忆系统，所有决策均基于概率/贝叶斯后验，无硬编码阈值。

## 核心特性

- **零硬编码**：Thompson Sampling 编码门、CRP 非参数聚类、可学习度量
- **叙事层级**：Plot → Story → Theme 自组织涌现
- **因果推理**：因果方向发现、do-calculus 干预、反事实推理
- **一致性守护**：概率矛盾检测、信念网络、自动冲突解决
- **自我叙事**：动态能力认知、关系建模、身份演化
- **生产就绪**：事件溯源、多租户、REST API、容器化

## 快速开始

```bash
# 安装
pip install -e .

# 运行演示
python -m aurora demo

# 启动 API 服务
pip install -e ".[api]"
aurora serve --port 8000
```

## 目录结构

```
.
├── aurora/                    # 主包
│   ├── algorithms/            # 核心算法
│   │   ├── aurora_core.py     # 主算法 (Thompson/CRP/场检索)
│   │   ├── causal.py          # 因果推理模块
│   │   ├── coherence.py       # 一致性守护模块
│   │   └── self_narrative.py  # 自我叙事模块
│   ├── api/                   # REST API (FastAPI)
│   ├── llm/                   # LLM 抽象层
│   ├── storage/               # 存储层 (事件日志/快照)
│   ├── embeddings/            # 嵌入接口
│   ├── privacy/               # 隐私保护 (PII脱敏)
│   ├── utils/                 # 工具函数
│   ├── service.py             # 服务层 (多租户)
│   ├── hub.py                 # 租户路由
│   ├── config.py              # 配置
│   └── cli.py                 # CLI 入口
├── tests/                     # 测试 (65 cases)
├── docs/                      # 文档
│   ├── AURORA_memory_algorithm.md    # 算法设计文档
│   ├── AURORA_production_pack.md     # 生产部署文档
│   └── narrative_memory_architecture.md
├── pyproject.toml             # 项目配置
├── Dockerfile                 # 容器化
└── docker-compose.yml
```

## CLI 命令

```bash
aurora demo                    # 运行完整演示
aurora ingest -u USER -m MSG -a REPLY  # 摄入交互
aurora query -u USER -q QUERY  # 查询记忆
aurora evolve -u USER          # 触发演化
aurora coherence -u USER       # 检查一致性
aurora narrative -u USER       # 获取自我叙事
aurora stats -u USER           # 获取统计
aurora serve                   # 启动 API 服务
```

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/memory/ingest` | POST | 摄入交互 |
| `/v1/memory/query` | POST | 查询记忆 |
| `/v1/memory/feedback` | POST | 记录反馈 |
| `/v1/memory/evolve` | POST | 触发演化 |
| `/v1/memory/coherence/{user_id}` | GET | 检查一致性 |
| `/v1/memory/self-narrative/{user_id}` | GET | 获取自我叙事 |
| `/v1/memory/stats/{user_id}` | GET | 获取统计 |

## 测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## 文档

详细算法设计见 `docs/AURORA_memory_algorithm.md`

## License

MIT
