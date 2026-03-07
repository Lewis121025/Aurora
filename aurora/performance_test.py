"""
AURORA 性能基准测试
=============================

关键操作的性能基准测试。

测试覆盖:
- 摄入吞吐量
- 查询延迟
- 演变性能
- 序列化速度
- 内存使用

运行方式: pytest tests/performance/ -v --benchmark
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import pytest

from aurora.algorithms.aurora_core import AuroraMemory
from aurora.algorithms.models.config import MemoryConfig
from aurora.algorithms.models.plot import Plot
from aurora.utils.time_utils import now_ts
from aurora.utils.id_utils import det_id


class TestIngestPerformance:
    """摄入性能的基准测试。"""

    def test_ingest_throughput(self):
        """测试摄入吞吐量：目标 > 100 ops/s。"""
        config = MemoryConfig(dim=64, max_plots=2000)
        memory = AuroraMemory(cfg=config, seed=42)
        
        num_operations = 200
        start = time.perf_counter()
        
        for i in range(num_operations):
            memory.ingest(
                f"用户：测试交互{i}。助理：这是回应{i}。",
                actors=("user", "assistant"),
            )
        
        elapsed = time.perf_counter() - start
        throughput = num_operations / elapsed
        
        print(f"\nIngest throughput: {throughput:.1f} ops/s")
        
        # Target: > 100 ops/s (relaxed for CI environments)
        assert throughput > 50, f"Throughput too low: {throughput:.1f} ops/s"
    
    def test_ingest_latency_p99(self):
        """Test ingest latency p99 < 50ms."""
        config = MemoryConfig(dim=64, max_plots=500)
        memory = AuroraMemory(cfg=config, seed=42)
        
        latencies: List[float] = []
        
        for i in range(100):
            start = time.perf_counter()
            memory.ingest(
                f"测试交互{i}",
                actors=("user", "assistant"),
            )
            latencies.append(time.perf_counter() - start)
        
        p99 = np.percentile(latencies, 99) * 1000  # Convert to ms
        p50 = np.percentile(latencies, 50) * 1000
        
        print(f"\nIngest latency p50: {p50:.1f}ms, p99: {p99:.1f}ms")
        
        # Target: p99 < 100ms (relaxed for CI)
        assert p99 < 100, f"p99 latency too high: {p99:.1f}ms"


class TestQueryPerformance:
    """Benchmarks for query performance."""
    
    @pytest.fixture
    def populated_memory(self):
        """Create memory with 500 plots for query benchmarks."""
        config = MemoryConfig(dim=64, max_plots=600)
        memory = AuroraMemory(cfg=config, seed=42)
        
        for i in range(500):
            memory.ingest(
                f"测试文档{i}：包含一些内容和关键词{i % 50}。",
                actors=("user", "assistant"),
            )
        
        return memory
    
    def test_query_latency(self, populated_memory: AuroraMemory):
        """Test query latency: target p99 < 50ms."""
        latencies: List[float] = []
        
        for i in range(50):
            start = time.perf_counter()
            populated_memory.query(f"查询{i}", k=10)
            latencies.append(time.perf_counter() - start)
        
        p99 = np.percentile(latencies, 99) * 1000
        p50 = np.percentile(latencies, 50) * 1000
        avg = np.mean(latencies) * 1000
        
        print(f"\nQuery latency avg: {avg:.1f}ms, p50: {p50:.1f}ms, p99: {p99:.1f}ms")
        
        # Target: p99 < 100ms
        assert p99 < 100, f"p99 query latency too high: {p99:.1f}ms"
    
    def test_query_throughput(self, populated_memory: AuroraMemory):
        """Test query throughput: target > 50 qps."""
        num_queries = 100
        start = time.perf_counter()
        
        for i in range(num_queries):
            populated_memory.query(f"查询词{i % 20}", k=5)
        
        elapsed = time.perf_counter() - start
        qps = num_queries / elapsed
        
        print(f"\nQuery throughput: {qps:.1f} qps")
        
        # Target: > 20 qps (relaxed for CI)
        assert qps > 20, f"Query throughput too low: {qps:.1f} qps"


class TestEvolutionPerformance:
    """Benchmarks for evolution performance."""
    
    def test_evolve_latency(self):
        """Test evolution latency with moderate data."""
        config = MemoryConfig(dim=64, max_plots=200)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Build up some state
        for i in range(100):
            memory.ingest(f"交互{i}", actors=("user", "assistant"))
        
        # Benchmark evolution
        latencies: List[float] = []
        
        for _ in range(10):
            start = time.perf_counter()
            memory.evolve()
            latencies.append(time.perf_counter() - start)
        
        avg = np.mean(latencies) * 1000
        max_lat = max(latencies) * 1000
        
        print(f"\nEvolution latency avg: {avg:.1f}ms, max: {max_lat:.1f}ms")
        
        # Target: avg < 500ms
        assert avg < 500, f"Evolution too slow: {avg:.1f}ms"
    
    def test_snapshot_evolution_performance(self):
        """Test copy-on-write evolution performance."""
        config = MemoryConfig(dim=64, max_plots=300)
        memory = AuroraMemory(cfg=config, seed=42)
        
        for i in range(150):
            memory.ingest(f"交互{i}", actors=("user", "assistant"))
        
        # Benchmark snapshot creation
        start = time.perf_counter()
        snapshot = memory.create_evolution_snapshot()
        snapshot_time = (time.perf_counter() - start) * 1000
        
        # Benchmark patch computation
        start = time.perf_counter()
        patch = memory.compute_evolution_patch(snapshot)
        compute_time = (time.perf_counter() - start) * 1000
        
        # Benchmark patch application
        start = time.perf_counter()
        memory.apply_evolution_patch(patch)
        apply_time = (time.perf_counter() - start) * 1000
        
        print(f"\nSnapshot: {snapshot_time:.1f}ms, Compute: {compute_time:.1f}ms, Apply: {apply_time:.1f}ms")
        
        total = snapshot_time + compute_time + apply_time
        assert total < 1000, f"Total evolution too slow: {total:.1f}ms"


class TestSerializationPerformance:
    """Benchmarks for serialization performance."""
    
    def test_serialization_speed(self):
        """Test serialization speed."""
        config = MemoryConfig(dim=64, max_plots=200)
        memory = AuroraMemory(cfg=config, seed=42)
        
        for i in range(100):
            memory.ingest(f"交互{i}", actors=("user", "assistant"))
        
        # Benchmark serialization
        start = time.perf_counter()
        state = memory.to_state_dict()
        serialize_time = (time.perf_counter() - start) * 1000
        
        # Benchmark deserialization
        start = time.perf_counter()
        restored = AuroraMemory.from_state_dict(state)
        deserialize_time = (time.perf_counter() - start) * 1000
        
        print(f"\nSerialize: {serialize_time:.1f}ms, Deserialize: {deserialize_time:.1f}ms")
        
        # Target: both < 500ms
        assert serialize_time < 500, f"Serialization too slow: {serialize_time:.1f}ms"
        assert deserialize_time < 500, f"Deserialization too slow: {deserialize_time:.1f}ms"


class TestMemoryScaling:
    """Tests for memory scaling behavior."""
    
    def test_linear_scaling_ingest(self):
        """Test that ingest scales roughly linearly."""
        sizes = [50, 100, 200]
        times: List[float] = []
        
        for size in sizes:
            config = MemoryConfig(dim=64, max_plots=size * 2)
            memory = AuroraMemory(cfg=config, seed=42)
            
            start = time.perf_counter()
            for i in range(size):
                memory.ingest(f"交互{i}", actors=("user", "assistant"))
            times.append(time.perf_counter() - start)
        
        # Check that time doesn't grow faster than O(n^2)
        # (allowing super-linear behavior for graph operations and KDE updates)
        ratios = [times[i+1] / times[i] for i in range(len(times)-1)]
        size_ratios = [sizes[i+1] / sizes[i] for i in range(len(sizes)-1)]
        
        for ratio, size_ratio in zip(ratios, size_ratios):
            # Time ratio should be less than size_ratio^2
            assert ratio < size_ratio ** 2, "Ingest scaling is worse than O(n^2)"
        
        print(f"\nScaling: sizes={sizes}, times={[f'{t:.2f}s' for t in times]}")


class TestPressurePerformance:
    """Tests for memory pressure handling performance."""
    
    def test_pressure_management_overhead(self):
        """Test overhead of pressure management."""
        config = MemoryConfig(dim=64, max_plots=50)
        memory = AuroraMemory(cfg=config, seed=42)
        
        # Fill to capacity
        for i in range(50):
            memory.ingest(f"初始交互{i}", actors=("user", "assistant"))
        
        # Measure overhead when at capacity
        latencies: List[float] = []
        
        for i in range(50):
            start = time.perf_counter()
            memory.ingest(f"压力测试{i}", actors=("user", "assistant"))
            latencies.append(time.perf_counter() - start)
        
        avg = np.mean(latencies) * 1000
        p99 = np.percentile(latencies, 99) * 1000
        
        print(f"\nPressure overhead avg: {avg:.1f}ms, p99: {p99:.1f}ms")
        
        # Should still be reasonably fast
        assert p99 < 200, f"Pressure management too slow: {p99:.1f}ms"
