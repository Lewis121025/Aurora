from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AuroraSettings(BaseSettings):
    """Runtime configuration for Aurora v4."""

    model_config = SettingsConfigDict(
        env_prefix="AURORA_",
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    data_dir: str = "./data"
    event_log_filename: str = "events.sqlite3"
    snapshot_dirname: str = "snapshots"
    snapshot_every_events: int = 200
    memory_seed: int = 42

    pii_redaction_enabled: bool = True
    evolve_every_seconds: int = 3600

    dim: int = 1024
    metric_rank: int = 64
    max_plots: int = 5000
    kde_reservoir: int = 4096
    story_alpha: float = 1.0
    theme_alpha: float = 0.6
    subconscious_reservoir: int = 1024
    mode_refractory_steps: int = 4
    mode_new_threshold: float = 0.52
    encode_min_events_before_gating: int = 6
    max_recent_texts: int = 12
    axis_merge_every_events: int = 50
    persona_axis_budget: int = 24
    dreams_per_evolve: int = 2
    profile_text: str = ""
    persona_axes_json: Optional[str] = None

    meaning_provider: Literal["heuristic", "llm"] = "llm"
    narrative_provider: Literal["heuristic", "llm"] = "llm"

    gate_forgetting_factor: float = 0.99
    metric_window_size: int = 10000
    metric_decay_factor: float = 0.5

    llm_provider: Literal["bailian", "ark", "mock"] = "mock"
    ark_api_key: Optional[str] = None
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    ark_llm_model: str = "doubao-seed-1-8-251228"

    bailian_llm_api_key: Optional[str] = None
    bailian_llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    bailian_llm_model: str = "qwen3.5-plus"

    llm_timeout: float = 60.0
    llm_max_retries: int = 3

    embedding_provider: Literal["bailian", "ark", "local", "hash"] = "local"
    axis_embedding_provider: Optional[Literal["bailian", "ark", "local", "hash"]] = None
    bailian_embedding_api_key: Optional[str] = None
    bailian_embedding_model: str = "text-embedding-v4"
    bailian_embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_cache_enabled: bool = True
    embedding_cache_size: int = 10000

    mcp_port: int = 8765
