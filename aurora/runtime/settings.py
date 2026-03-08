from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AuroraSettings(BaseSettings):
    """Runtime configuration.

    Design notes:
    - Hard thresholds are avoided in algorithmic decisions.
    - Here we still need *operational* constraints: storage paths, budgets, intervals, etc.
    """

    model_config = SettingsConfigDict(env_prefix="AURORA_", extra="ignore")

    # single-user runtime
    data_dir: str = "./data"

    # event sourcing
    event_log_filename: str = "events.sqlite3"
    snapshot_dirname: str = "snapshots"
    snapshot_every_events: int = 200  # startup vs write cost trade-off
    memory_seed: int = 42

    # privacy
    pii_redaction_enabled: bool = True

    # offline evolution
    evolve_every_seconds: int = 3600  # optional scheduler interval

    # algorithm core
    dim: int = 1024  # Embedding dimension
    metric_rank: int = 64
    max_plots: int = 5000
    kde_reservoir: int = 4096
    story_alpha: float = 1.0
    theme_alpha: float = 0.5
    
    # --- 算法稳定性参数 ---
    
    # Thompson Gate 遗忘因子 (0.99 = 每次更新保留 99% 的历史精度)
    gate_forgetting_factor: float = 0.99
    
    # LowRank Metric 滑动窗口大小 (Adagrad 重置周期)
    metric_window_size: int = 10000
    
    # LowRank Metric 衰减因子 (窗口重置时的衰减)
    metric_decay_factor: float = 0.5
    
    # --- 火山方舟 (Volcengine Ark) LLM 配置 ---
    
    # LLM 提供者: "ark" (火山方舟) 或 "mock" (本地测试)
    llm_provider: Literal["ark", "mock"] = "mock"
    
    # 火山方舟 API Key
    ark_api_key: Optional[str] = None
    
    # 火山方舟 API 基础 URL
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    
    # LLM 模型
    ark_llm_model: str = "doubao-seed-1-8-251228"
    
    # LLM 请求超时 (秒)
    llm_timeout: float = 60.0
    
    # LLM 最大重试次数
    llm_max_retries: int = 3
    
    # --- 阿里云百炼 (Alibaba Bailian) 嵌入配置 ---
    
    # 嵌入提供者: "bailian" | "ark" | "local" | "hash"
    embedding_provider: Literal["bailian", "ark", "local", "hash"] = "local"
    
    # 阿里云百炼 API Key
    bailian_api_key: Optional[str] = None
    
    # 阿里云百炼嵌入模型
    bailian_embedding_model: str = "text-embedding-v4"
    
    # 阿里云百炼 API 基础 URL
    bailian_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    # --- 嵌入通用配置 ---
    
    # 嵌入缓存开关
    embedding_cache_enabled: bool = True
    
    # 嵌入缓存大小
    embedding_cache_size: int = 10000
    
    # --- MCP 服务器配置 ---
    
    # MCP 服务器端口 (HTTP 模式)
    mcp_port: int = 8765
