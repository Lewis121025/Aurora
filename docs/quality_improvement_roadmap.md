# AURORA 代码质量提升路线图

**目标：** 从当前 8/10 提升至 10/10

**时间估算：** 4-6 周（取决于投入程度）

---

## 当前问题诊断

### 1. 复杂度管理（影响：高）
- `AuroraMemory` 继承 4 个 Mixin，职责过多
- `constants.py` 595 行，50+ 常量缺少系统性组织
- 核心算法逻辑与运行时关注点混合

### 2. 类型安全（影响：中）
- 大量 `Dict[str, Any]`、`Optional[Any]` 弱类型
- 缺少 mypy/pyright 严格模式检查
- 运行时类型错误风险高

### 3. 运行时诊断（影响：中）
- 关键路径日志粒度还不够稳定
- 缺少统一的健康检查与状态摘要
- 快照与回放异常的定位成本偏高

### 4. 测试质量（影响：中）
- 边界条件覆盖不足
- 缺少性能基准测试
- 缺少混沌测试

### 5. 状态管理（影响：中）
- 序列化无版本控制
- 模型变更会破坏旧数据
- 缺少迁移机制

---

## 阶段 1：类型安全与静态检查

**目标：** 8/10 → 8.5/10
**时间：** 1 周
**优先级：** 高

### 任务清单

- [ ] 添加 `pyproject.toml` 类型检查配置
  ```toml
  [tool.mypy]
  python_version = "3.10"
  strict = true
  warn_return_any = true
  warn_unused_configs = true
  disallow_untyped_defs = true
  ```

- [ ] 修复核心模块的类型标注
  - `aurora/core/memory/engine.py`
  - `aurora/core/models/*.py`
  - `aurora/runtime/runtime.py`
  - `aurora/runtime/bootstrap.py`

- [ ] 替换 `Dict[str, Any]` 为 TypedDict 或 Pydantic 模型
  - `Plot.metadata` → `PlotMetadata(BaseModel)`
  - `StoryArc.context` → `StoryContext(BaseModel)`
  - `Theme.dimensions` → `ThemeDimensions(BaseModel)`

- [ ] 添加 CI 类型检查
  ```yaml
  # .github/workflows/type-check.yml
  - name: Type check
    run: mypy aurora tests
  ```

### 可交付成果
- 所有核心模块通过 `mypy --strict`
- CI 自动类型检查
- 类型覆盖率报告

---

## 阶段 2：轻量运行时诊断

**目标：** 8.5/10 → 9/10
**时间：** 1 周
**优先级：** 高

### 任务清单

- [ ] 结构化日志
  ```python
  # aurora/utils/logging.py
  import structlog

  logger = structlog.get_logger()
  logger.info("plot_ingested",
              plot_id=plot.id,
              story_id=plot.story_id,
              tension=plot.tension,
              duration_ms=elapsed)
  ```

- [ ] 统一健康检查与运行时摘要
  ```python
  # aurora/interfaces/api/app.py
  @app.get("/healthz")
  def healthz():
      return {"status": "healthy", "timestamp": time.time()}
  ```

- [ ] 快照/回放错误日志标准化
  ```python
  logger.warning(
      "aurora_snapshot_failed",
      extra={"last_seq": last_seq, "path": path},
  )
  ```

- [ ] 添加本地状态诊断命令
  ```python
  # aurora/interfaces/cli.py
  aurora stats
  aurora coherence
  ```

### 可交付成果
- 结构化日志输出
- `/healthz` 与 CLI 诊断命令保持一致
- 快照与回放异常可直接定位

---

## 阶段 3：复杂度重构

**目标：** 9/10 → 9.3/10
**时间：** 2 周
**优先级：** 中

### 任务清单

- [ ] 拆分 `AuroraMemory` God Class
  ```python
  # 当前：AuroraMemory 继承 4 个 Mixin
  # 重构后：组合优于继承

  class AuroraMemory:
      def __init__(self, cfg, seed, embedder):
          self.relationship_manager = RelationshipManager(...)
          self.pressure_manager = PressureManager(...)
          self.evolution_engine = EvolutionEngine(...)
          self.serializer = MemorySerializer(...)
  ```

- [ ] 重组 `constants.py`
  ```python
  # aurora/core/config/
  #   retrieval.py    - 检索相关常量
  #   storage.py      - 存储相关常量
  #   coherence.py    - 一致性相关常量
  #   evolution.py    - 演化相关常量
  ```

- [ ] 提取配置验证逻辑
  ```python
  # aurora/core/models/config.py
  class MemoryConfig(BaseModel):
      dim: int = Field(gt=0, le=2048)
      max_plots: int = Field(gt=0)

      @validator('metric_rank')
      def validate_metric_rank(cls, v, values):
          if v > values['dim']:
              raise ValueError('metric_rank must be <= dim')
          return v
  ```

- [ ] 添加架构决策记录（ADR）
  ```markdown
  # docs/adr/001-mixin-to-composition.md

  ## 状态
  已接受

  ## 背景
  AuroraMemory 通过 4 个 Mixin 继承功能，导致：
  - 方法解析顺序（MRO）复杂
  - 职责边界模糊
  - 测试困难

  ## 决策
  改用组合模式，将 Mixin 转为独立的管理器类

  ## 后果
  - 依赖注入更明确
  - 单元测试更容易
  - 代码行数略增加
  ```

### 可交付成果
- `AuroraMemory` 类复杂度降低 50%
- 常量按模块组织，每个文件 < 200 行
- 5+ 篇 ADR 文档

---

## 阶段 4：测试深度提升

**目标：** 9.3/10 → 9.7/10
**时间：** 1.5 周
**优先级：** 中

### 任务清单

- [ ] 边界条件测试
  ```python
  # tests/core/test_aurora_core_edge_cases.py

  def test_empty_memory_query():
      """空记忆查询应返回空结果"""
      mem = AuroraMemory(cfg=MemoryConfig(dim=64), seed=42)
      trace = mem.query("test", k=5)
      assert len(trace.ranked) == 0

  def test_single_plot_evolution():
      """单条 Plot 不应触发演化"""
      mem = AuroraMemory(cfg=MemoryConfig(dim=64), seed=42)
      mem.ingest("test")
      mem.evolve()
      assert len(mem.stories) == 0

  def test_max_capacity_overflow():
      """超过 max_plots 应触发压力机制"""
      mem = AuroraMemory(cfg=MemoryConfig(dim=64, max_plots=10), seed=42)
      for i in range(15):
          mem.ingest(f"plot {i}")
      assert len(mem.plots) <= 10
  ```

- [ ] 性能基准测试
  ```python
  # tests/benchmarks/test_performance.py
  import pytest

  @pytest.mark.benchmark
  def test_ingest_latency(benchmark):
      mem = AuroraMemory(cfg=MemoryConfig(dim=64), seed=42)
      result = benchmark(mem.ingest, "test interaction")
      assert result.stats.mean < 0.1  # < 100ms

  @pytest.mark.benchmark
  def test_query_latency_1k_plots(benchmark):
      mem = setup_memory_with_1k_plots()
      result = benchmark(mem.query, "test query", k=10)
      assert result.stats.mean < 0.05  # < 50ms
  ```

- [ ] 属性测试（Property-based testing）
  ```python
  # tests/core/test_properties.py
  from hypothesis import given, strategies as st

  @given(st.text(min_size=1, max_size=1000))
  def test_ingest_never_crashes(text):
      """任意文本输入都不应导致崩溃"""
      mem = AuroraMemory(cfg=MemoryConfig(dim=64), seed=42)
      try:
          mem.ingest(text)
      except Exception as e:
          pytest.fail(f"Ingest crashed with: {e}")

  @given(st.integers(min_value=1, max_value=100))
  def test_query_k_invariant(k):
      """查询返回的结果数 <= k"""
      mem = setup_memory_with_50_plots()
      trace = mem.query("test", k=k)
      assert len(trace.ranked) <= k
  ```

- [ ] 混沌测试
  ```python
  # tests/chaos/test_resilience.py

  def test_concurrent_ingest_query():
      """并发摄入和查询不应导致数据竞争"""
      mem = AuroraMemory(cfg=MemoryConfig(dim=64), seed=42)
      with ThreadPoolExecutor(max_workers=10) as executor:
          futures = []
          for i in range(100):
              if i % 2 == 0:
                  futures.append(executor.submit(mem.ingest, f"plot {i}"))
              else:
                  futures.append(executor.submit(mem.query, "test", k=5))
          for f in futures:
              f.result()  # 不应抛出异常
  ```

### 可交付成果
- 测试覆盖率 > 85%
- 50+ 边界条件测试
- 性能基准测试套件
- 混沌测试通过

---

## 阶段 5：生产就绪度

**目标：** 9.7/10 → 10/10
**时间：** 1 周
**优先级：** 中

### 任务清单

- [ ] 状态序列化版本控制
  ```python
  # aurora/core/models/base.py

  class SerializableMixin(ABC):
      SCHEMA_VERSION: ClassVar[str] = "1.0.0"

      def to_dict(self) -> Dict[str, Any]:
          return {
              "__schema_version__": self.SCHEMA_VERSION,
              "__class__": self.__class__.__name__,
              **self._serialize_fields()
          }

      @classmethod
      def from_dict(cls, data: Dict[str, Any]):
          version = data.get("__schema_version__", "0.0.0")
          if version != cls.SCHEMA_VERSION:
              data = cls._migrate(data, from_version=version)
          return cls._deserialize_fields(data)

      @classmethod
      def _migrate(cls, data: Dict[str, Any], from_version: str) -> Dict[str, Any]:
          """版本迁移逻辑"""
          migrations = {
              "0.0.0": cls._migrate_0_to_1,
              "1.0.0": lambda x: x,  # 当前版本
          }
          return migrations[from_version](data)
  ```

- [ ] 迁移工具
  ```python
  # aurora/scripts/migrate_state.py

  def migrate_snapshot(snapshot_path: Path, target_version: str):
      """迁移快照到目标版本"""
      with open(snapshot_path, 'rb') as f:
          snap = pickle.load(f)

      migrated = AuroraMemory.from_dict(snap.state.to_dict())

      backup_path = snapshot_path.with_suffix('.bak')
      shutil.copy(snapshot_path, backup_path)

      with open(snapshot_path, 'wb') as f:
          pickle.dump(Snapshot(last_seq=snap.last_seq, state=migrated), f)
  ```

- [ ] 错误预算与 SLO
  ```python
  # aurora/runtime/slo.py

  class SLOMonitor:
      """监控 SLO 并计算错误预算"""

      def __init__(self):
          self.slo_targets = {
              "ingest_p99_latency_ms": 200,
              "query_p99_latency_ms": 100,
              "availability": 0.999,  # 99.9%
          }
          self.error_budget = 0.001  # 0.1% 错误预算

      def check_slo_compliance(self) -> Dict[str, bool]:
          """检查是否符合 SLO"""
          ...
  ```

- [ ] 容量规划文档
  ```markdown
  # docs/capacity_planning.md

  ## 性能基准

  ### 单机配置（8 核 16GB）
  - 摄入吞吐量：500 plots/s
  - 查询延迟（P99）：< 50ms（1K plots）
  - 内存占用：~100MB（1K plots）

  ### 扩展性
  - 10K plots：查询延迟 < 100ms
  - 100K plots：需要 FAISS GPU 索引
  - 1M+ plots：需要分片存储
  ```

- [ ] 灾难恢复手册
  ```markdown
  # docs/disaster_recovery.md

  ## 场景 1：快照损坏
  1. 从事件日志重放
  2. 验证数据完整性
  3. 创建新快照

  ## 场景 2：数据库故障
  1. 切换到备用数据库
  2. 从最近快照恢复
  3. 重放增量事件
  ```

### 可交付成果
- 状态序列化版本控制
- 自动迁移工具
- SLO 监控仪表板
- 容量规划文档
- 灾难恢复手册

---

## 质量门禁

每个阶段完成后必须通过以下检查：

### 代码质量
- [ ] `mypy --strict` 无错误
- [ ] `ruff check` 无警告
- [ ] 测试覆盖率 > 85%
- [ ] 所有测试通过

### 性能
- [ ] 摄入延迟 P99 < 200ms
- [ ] 查询延迟 P99 < 100ms（1K plots）
- [ ] 内存占用 < 150MB（1K plots）

### 可观测性
- [ ] 所有关键路径有时延与错误日志
- [ ] 所有错误有结构化日志
- [ ] 诊断命令与健康检查输出一致

### 文档
- [ ] API 文档完整
- [ ] 架构决策记录（ADR）更新
- [ ] 运维手册完整

---

## 成功指标

| 维度 | 当前 | 目标 | 衡量方式 |
|------|------|------|----------|
| 类型覆盖率 | ~60% | 95% | mypy --strict |
| 测试覆盖率 | ~70% | 85% | pytest-cov |
| 代码复杂度 | 高 | 中 | radon cc |
| 文档完整度 | 70% | 90% | 人工审查 |
| 可观测性 | 20% | 90% | 关键路径覆盖 |
| 生产就绪度 | 60% | 95% | 检查清单 |

---

## 下一步行动

选择一个阶段开始：

1. **如果你关心稳定性** → 从阶段 1（类型安全）开始
2. **如果你关心可维护性** → 从阶段 2（可观测性）开始
3. **如果你关心架构** → 从阶段 3（复杂度重构）开始
4. **如果你关心质量** → 从阶段 4（测试深度）开始
5. **如果你要上生产** → 从阶段 5（生产就绪度）开始

建议顺序：1 → 2 → 3 → 4 → 5（从基础到高级）
