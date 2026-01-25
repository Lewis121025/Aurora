from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class AuroraSettings(BaseSettings):
    """Runtime configuration.

    Design notes:
    - Hard thresholds are avoided in algorithmic decisions.
    - Here we still need *operational* constraints: storage paths, budgets, intervals, etc.
    """

    model_config = SettingsConfigDict(env_prefix="AURORA_", extra="ignore")

    # multi-tenant
    data_dir: str = "./data"
    tenant_max_loaded: int = 32  # LRU cap for hot tenants

    # event sourcing
    event_log_filename: str = "events.sqlite3"
    snapshot_dirname: str = "snapshots"
    snapshot_every_events: int = 200  # startup vs write cost trade-off

    # privacy
    pii_redaction_enabled: bool = True

    # offline evolution
    evolve_every_seconds: int = 3600  # optional scheduler interval

    # algorithm core
    dim: int = 384
    metric_rank: int = 64
    max_plots: int = 5000
    kde_reservoir: int = 4096
    story_alpha: float = 1.0
    theme_alpha: float = 0.5
