# AURORA 基准测试指南

> [!NOTE]
> 当前基准适配器默认支持直接传入 `AuroraSoul`，部分场景也兼容 `AuroraRuntime`。下面的示例统一使用 `aurora.soul.AuroraSoul` 与 `SoulConfig`。

本指南说明如何使用 AURORA 记忆系统参与学术基准测试。

## 支持的基准测试

### 1. MemoryAgentBench (2025.07)

学术来源：[arXiv:2507.05257](https://arxiv.org/abs/2507.05257)

数据集：[HuggingFace ai-hyz/MemoryAgentBench](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench)

**评测维度：**

| 能力 | 实例数 | AURORA 实现 |
|------|--------|-------------|
| 准确检索 (AR) | 22 | `query()` + `FieldRetriever` |
| 测试时学习 (TTL) | 6 | `ingest()` + `evolve()` |
| 长程理解 (LRU) | 110 | Story 聚合 + Theme 涌现 + `NarratorEngine` |
| 冲突解决 (CR) | 8 | `TensionManager` + `CoherenceGuardian` |

### 2. LOCOMO (ACL 2024)

学术来源：[ACL 2024.acl-long.747](https://aclanthology.org/2024.acl-long.747/)

数据集：[GitHub snap-research/locomo](https://github.com/snap-research/locomo)

**评测任务：**

| 任务 | 推理类型 | AURORA 实现 |
|------|----------|-------------|
| 问答 | 单跳/多跳/时序/常识/世界知识 | `query()` + 关系优先检索 |
| 事件摘要 | - | Story narrative + `NarratorEngine` |
| 多模态对话 | - | （可选扩展） |

---

## 快速开始

### 安装依赖

```bash
# 安装基础依赖
pip install -e .

# 安装评测相关依赖
pip install datasets  # HuggingFace datasets

# 可选：安装 LLM 提供商
pip install -e ".[ark]"  # 火山方舟
```

### 环境配置

```bash
# 设置 API Key（用于 LLM-as-Judge 评估）
export ARK_API_KEY="your-api-key"

# 或使用 .env 文件
cp .env.example .env
# 编辑 .env 添加 API Key
```

---

## 运行 MemoryAgentBench

### 基本用法

```python
from aurora.benchmarks import MemoryAgentBenchAdapter, EvaluationConfig
from aurora.soul.engine import AuroraSoul, SoulConfig

# 1. 创建记忆实例
config = SoulConfig(dim=96, metric_rank=32, max_plots=2000)
memory = AuroraSoul(cfg=config, seed=42)

# 2. 创建适配器
adapter = MemoryAgentBenchAdapter(seed=42)

# 3. 运行评测
eval_config = EvaluationConfig(
    use_llm_judge=False,  # 设为 True 启用 LLM 评分
    verbose=True,
)
results, metrics = adapter.run_benchmark_with_config(
    memory=memory,
    source="ai-hyz/MemoryAgentBench",  # 从 HuggingFace 加载
    config=eval_config,
)

# 4. 查看结果
print(f"Overall Accuracy: {metrics.accuracy:.2%}")
print(f"AR Accuracy: {metrics.metrics_by_type.get('accurate_retrieval_accuracy', 0):.2%}")
print(f"TTL Accuracy: {metrics.metrics_by_type.get('test_time_learning_accuracy', 0):.2%}")
print(f"LRU Accuracy: {metrics.metrics_by_type.get('long_range_understanding_accuracy', 0):.2%}")
print(f"CR Accuracy: {metrics.metrics_by_type.get('conflict_resolution_accuracy', 0):.2%}")
```

### 使用本地数据集

```python
# 从本地 JSON 文件加载
results, metrics = adapter.run_benchmark_with_config(
    memory=memory,
    source="./data/mab_dataset.json",
    config=eval_config,
)
```

### 启用 LLM-as-Judge

```python
from aurora.integrations.llm.ark import ArkLLMWithFallback

# 创建 LLM 提供商
llm = ArkLLMWithFallback(api_key="your-api-key")

# 创建带 LLM 的适配器
adapter = MemoryAgentBenchAdapter(llm_provider=llm, seed=42)

# 启用 LLM 评分
eval_config = EvaluationConfig(use_llm_judge=True)
```

---

## 运行 LOCOMO

### 基本用法

```python
from aurora.benchmarks.adapters.locomo import LOCOMOAdapter
from aurora.soul.engine import AuroraSoul, SoulConfig

# 1. 创建记忆实例
config = SoulConfig(dim=96, metric_rank=32, max_plots=2000)
memory = AuroraSoul(cfg=config, seed=42)

# 2. 创建适配器
adapter = LOCOMOAdapter(seed=42)

# 3. 加载数据集
instances = adapter.load_dataset("./data/locomo/")

# 4. 运行评测
results = []
for instance in instances:
    # 准备记忆（摄入对话历史）
    adapter.prepare_memory(instance.conversation_history, memory)
    
    # 评估
    result = adapter.evaluate(instance, memory)
    results.append(result)
    
    # 重置记忆（可选）
    memory = AuroraSoul(cfg=config, seed=42)

# 5. 聚合结果
metrics = adapter.aggregate_results(results)
print(f"Overall Accuracy: {metrics['accuracy']:.2%}")
print(f"QA Accuracy: {metrics.get('qa_accuracy', 0):.2%}")
print(f"Summarization Accuracy: {metrics.get('summarization_accuracy', 0):.2%}")
```

### 按推理类型查看结果

```python
# 获取详细指标
eval_metrics = adapter.get_evaluation_metrics(results)

print(f"Single-hop QA: {eval_metrics.metrics_by_type.get('single_hop_accuracy', 0):.2%}")
print(f"Multi-hop QA: {eval_metrics.metrics_by_type.get('multi_hop_accuracy', 0):.2%}")
print(f"Temporal QA: {eval_metrics.metrics_by_type.get('temporal_accuracy', 0):.2%}")
```

---

## 评测指标

### 准确率指标

| 指标 | 说明 |
|------|------|
| `exact_match` | 精确字符串匹配 |
| `contains_match` | 答案包含关系 |
| `fuzzy_match` | 模糊匹配（SequenceMatcher） |
| `token_f1` | Token 级别 F1 分数 |
| `llm_judge` | LLM-as-a-Judge 评分 |

### 性能指标

| 指标 | 说明 |
|------|------|
| `latency_p50` | 50 百分位延迟 |
| `latency_p95` | 95 百分位延迟 |
| `latency_p99` | 99 百分位延迟 |
| `throughput` | 每秒处理数 |

## 自定义评测

### 实现自定义适配器

```python
from aurora.benchmarks.interface import BenchmarkAdapter, BenchmarkInstance, BenchmarkResult

class MyBenchmarkAdapter(BenchmarkAdapter):
    def load_dataset(self, path: str) -> List[BenchmarkInstance]:
        # 加载你的数据集
        pass
    
    def evaluate(self, instance: BenchmarkInstance, memory) -> BenchmarkResult:
        # 评估逻辑
        pass
    
    def aggregate_results(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        # 聚合结果
        pass
```

### 使用 BenchmarkRunner

```python
from aurora.benchmarks.interface import AURORABenchmarkRunner

runner = AURORABenchmarkRunner(
    memory=memory,
    adapters={
        "mab": MemoryAgentBenchAdapter(),
        "locomo": LOCOMOAdapter(),
        "custom": MyBenchmarkAdapter(),
    }
)

# 运行所有基准测试
all_results = runner.run_all({
    "mab": "ai-hyz/MemoryAgentBench",
    "locomo": "./data/locomo/",
    "custom": "./data/my_benchmark/",
})
```

---

## 结果解读

### MemoryAgentBench 基线对比

根据原论文，各系统的典型表现：

| 系统 | AR | TTL | LRU | CR | Overall |
|------|-----|-----|-----|-----|---------|
| Mem0 (graph) | 72% | 65% | 68% | 55% | 68.5% |
| Mem0 (dense) | 70% | 62% | 67% | 52% | 66.9% |
| LangMem | 60% | 55% | 60% | 48% | 58.1% |
| OpenAI Memory | 55% | 50% | 54% | 45% | 52.9% |

### LOCOMO 基线对比

| 系统 | Single-hop | Multi-hop | Temporal | Summary |
|------|------------|-----------|----------|---------|
| GPT-4 + RAG | 65% | 45% | 40% | 55% |
| Long-context LLM | 70% | 50% | 45% | 60% |
| Human | 95% | 85% | 80% | 90% |

---

## 故障排除

### 常见问题

**Q: 评测速度很慢**

A: 考虑以下优化：
- 减少 `max_plots` 限制
- 优先控制记忆规模与检索窗口；Aurora 默认使用本地精确向量检索
- 禁用 LLM-as-Judge 使用精确匹配

**Q: 内存占用过高**

A: 调整配置：
```python
from aurora.soul.engine import SoulConfig

config = SoulConfig(
    dim=64,  # 降低嵌入维度
    max_plots=500,  # 限制存储数量
    kde_reservoir=500,  # 减少 KDE 样本
)
```

**Q: LLM API 调用失败**

A: 检查：
- API Key 是否正确设置
- 网络连接是否正常
- 使用 `MockLLM` 进行本地测试

---

## 参考资料

- [AURORA 算法设计文档](./AURORA_memory_algorithm.md)
- [AURORA 架构文档](./narrative_memory_architecture.md)
- [MemoryAgentBench 论文](https://arxiv.org/abs/2507.05257)
- [LOCOMO 论文](https://aclanthology.org/2024.acl-long.747/)
