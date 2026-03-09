"""
AURORA Test Fixtures
====================

Centralized pytest fixtures for all AURORA tests.

Provides:
- metric: LowRankMetric instance for similarity computation
- sample_plots: Pre-generated plot fixtures
- sample_themes: Pre-generated theme fixtures
- temp_data_dir: Temporary directory for test data
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator, List

import numpy as np
import pytest

from aurora.core.models.config import MemoryConfig
from aurora.core.models.plot import Plot, RelationalContext, IdentityImpact
from aurora.core.models.story import StoryArc
from aurora.core.models.theme import Theme
from aurora.core.components.metric import LowRankMetric
from aurora.core.components.density import OnlineKDE
from aurora.utils.time_utils import now_ts
from aurora.utils.id_utils import det_id


# =============================================================================
# Core Components
# =============================================================================

@pytest.fixture
def metric() -> LowRankMetric:
    """Create a LowRankMetric instance with deterministic seed."""
    return LowRankMetric(dim=64, rank=16, seed=42)


@pytest.fixture
def kde() -> OnlineKDE:
    """Create an OnlineKDE instance with deterministic seed."""
    return OnlineKDE(dim=64, reservoir=100, seed=42)


@pytest.fixture
def config() -> MemoryConfig:
    """Create a standard memory configuration for tests."""
    return MemoryConfig(
        dim=64,
        metric_rank=16,
        max_plots=100,
        story_alpha=1.0,
        theme_alpha=0.5,
    )


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_embedding(metric: LowRankMetric) -> np.ndarray:
    """Create a sample embedding vector."""
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(64).astype(np.float32)
    return emb / np.linalg.norm(emb)


@pytest.fixture
def sample_plot(sample_embedding: np.ndarray) -> Plot:
    """Create a sample Plot instance."""
    return Plot(
        id=det_id("plot", "test_1"),
        ts=now_ts(),
        text="用户：这是一个测试交互。助理：好的，我理解了。",
        actors=("user", "assistant"),
        embedding=sample_embedding,
        relational=RelationalContext(
            with_whom="user",
            my_role_in_relation="助手",
            relationship_quality_delta=0.1,
            what_this_says_about_us="这是一次友好的互动",
        ),
        identity_impact=IdentityImpact(
            when_formed=now_ts(),
            initial_meaning="这是我作为助手的一次体现",
            current_meaning="这是我作为助手的一次体现",
            identity_dimensions_affected=["作为助手的我"],
            evolution_history=[],
        ),
    )


@pytest.fixture
def sample_plots(metric: LowRankMetric) -> List[Plot]:
    """Create a list of sample Plot instances."""
    rng = np.random.default_rng(42)
    plots = []
    
    texts = [
        "用户：帮我解释一下递归。助理：递归是函数调用自身的过程。",
        "用户：什么是面向对象？助理：面向对象是一种编程范式。",
        "用户：怎么处理异常？助理：可以使用try-except语句。",
        "用户：感谢你的帮助！助理：不客气，随时可以问我。",
        "用户：这段代码有问题。助理：让我来看看哪里出错了。",
    ]
    
    for i, text in enumerate(texts):
        emb = rng.standard_normal(64).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        
        plot = Plot(
            id=det_id("plot", f"sample_{i}"),
            ts=now_ts() + i * 60,
            text=text,
            actors=("user", "assistant"),
            embedding=emb,
        )
        plots.append(plot)
    
    return plots


@pytest.fixture
def sample_story(sample_embedding: np.ndarray) -> StoryArc:
    """Create a sample StoryArc instance."""
    return StoryArc(
        id=det_id("story", "test_rel_user"),
        created_ts=now_ts(),
        updated_ts=now_ts(),
        relationship_with="user",
        relationship_type="user",
        centroid=sample_embedding,
    )


@pytest.fixture
def sample_theme(sample_embedding: np.ndarray) -> Theme:
    """Create a sample Theme instance."""
    return Theme(
        id=det_id("theme", "helper"),
        created_ts=now_ts(),
        updated_ts=now_ts(),
        prototype=sample_embedding,
        identity_dimension="作为帮助者的我",
        theme_type="identity",
    )


# =============================================================================
# Temporary Directory
# =============================================================================

@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path(temp_data_dir: Path) -> str:
    """Create a path for a temporary SQLite database."""
    return str(temp_data_dir / "test.db")


# =============================================================================
# Cleanup Helpers
# =============================================================================

@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test for reproducibility."""
    np.random.seed(42)
    yield
