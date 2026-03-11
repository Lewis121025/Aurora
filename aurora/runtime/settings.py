"""Runtime settings for Aurora V6."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATA_DIR = ".aurora"


class AuroraSettings(BaseSettings):
    """Aurora V5 runtime settings."""

    model_config = SettingsConfigDict(
        env_prefix="AURORA_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    data_dir: str = DEFAULT_DATA_DIR
    runtime_db_filename: str = "runtime.sqlite3"
    snapshot_dirname: str = "snapshots"
    snapshot_every_projected_events: int = 20
    memory_seed: int = 42

    worker_count: int = 2
    job_poll_interval_ms: int = 100
    job_retry_limit: int = 3
    job_lease_seconds: float = 10.0
    evolve_every_seconds: float = 3600.0
    fade_every_seconds: float = 900.0
    fade_cold_age_s: float = 86400.0
    fade_mass_threshold: float = 0.08
    fade_group_min_size: int = 3
    overlay_search_limit: int = 6

    dim: int = 1024
    metric_rank: int = 64
    max_plots: int = 5000
    kde_reservoir: int = 4096
    max_recent_semantic_texts: int = 12
    axis_merge_every_events: int = 50
    persona_axis_budget: int = 24
    dreams_per_evolve: int = 2
    profile_text: str = ""
    persona_axes_json: Optional[str] = None
    graph_temporal_neighbors: int = 2
    graph_semantic_neighbors: int = 3
    graph_contradiction_neighbors: int = 10
    graph_similarity_threshold: float = 0.2
    graph_contradiction_threshold: float = 0.16
    community_refresh_every_plots: int = 50
    dream_walk_steps: int = 6
    dream_walk_samples: int = 24
    dream_persist_threshold: float = 0.18

    meaning_provider: Literal["heuristic", "llm"] = "llm"
    narrative_provider: Literal["heuristic", "llm"] = "llm"

    llm_provider: Optional[Literal["bailian", "ark"]] = "bailian"

    ark_api_key: Optional[str] = None
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    ark_llm_model: str = "doubao-seed-1-8-251228"

    bailian_llm_api_key: Optional[str] = None
    bailian_llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    bailian_llm_model: str = "qwen3-vl-flash"

    llm_timeout: float = 60.0
    llm_max_retries: int = 3

    content_embedding_provider: Literal["bailian", "local", "hash"] = "local"
    text_embedding_provider: Optional[Literal["bailian", "ark", "local", "hash"]] = None

    bailian_embedding_api_key: Optional[str] = None
    bailian_text_embedding_model: str = "text-embedding-v4"
    bailian_text_embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    bailian_content_embedding_model: str = "qwen3-vl-embedding"
    bailian_content_embedding_base_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    )

    embedding_cache_enabled: bool = True
    embedding_cache_size: int = 10000

    mcp_port: int = 8765
