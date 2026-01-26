from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AuroraSettings(BaseSettings):
    """Runtime configuration.

    Design notes:
    - Hard thresholds are avoided in algorithmic decisions.
    - Here we still need *operational* constraints: storage paths, budgets, intervals, etc.
    
    Storage backends:
    - vector_store_backend: "memory" | "pgvector" | "milvus"
    - state_store_backend: "memory" | "sqlite" | "redis_postgres"
    - queue_backend: "memory" | "kafka" | "pulsar"
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
    
    # --- 向量存储配置 ---
    
    # 向量存储后端: "memory" (开发), "pgvector" (生产), "milvus" (备选)
    vector_store_backend: Literal["memory", "pgvector", "milvus"] = "memory"
    
    # PostgreSQL 连接字符串 (pgvector)
    postgres_dsn: Optional[str] = None
    
    # PostgreSQL 连接池大小
    postgres_pool_size: int = 10
    
    # 向量表名
    vector_table_name: str = "aurora_vectors"
    
    # Milvus 配置 (备选)
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "aurora_vectors"
    
    # --- 状态存储配置 ---
    
    # 状态存储后端: "memory" (开发), "sqlite" (单机), "redis_postgres" (分布式)
    state_store_backend: Literal["memory", "sqlite", "redis_postgres"] = "memory"
    
    # Redis URL (用于热数据)
    redis_url: str = "redis://localhost:6379/0"
    
    # Redis 键前缀
    redis_prefix: str = "aurora:state:"
    
    # Redis TTL (秒)
    redis_ttl: int = 86400  # 24 hours
    
    # 状态表名 (PostgreSQL 冷数据)
    state_table_name: str = "aurora_states"
    
    # --- 消息队列配置 ---
    
    # 队列后端: "memory" (开发), "kafka" (生产), "pulsar" (备选)
    queue_backend: Literal["memory", "kafka", "pulsar"] = "memory"
    
    # Kafka 配置
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_client_id: str = "aurora"
    kafka_ingest_topic: str = "aurora.ingest"
    
    # Pulsar 配置 (备选)
    pulsar_url: str = "pulsar://localhost:6650"
    
    # --- CQRS 模式配置 ---
    
    # 是否启用异步模式 (True = 写入队列后立即返回)
    cqrs_async_mode: bool = False
    
    # Worker 数量 (异步模式下)
    worker_count: int = 4
    
    # Worker 批处理大小
    worker_batch_size: int = 10
    
    # --- 可观测性配置 ---
    
    # 是否启用 Prometheus 指标
    metrics_enabled: bool = True
    
    # Prometheus 指标端口
    metrics_port: int = 9090
    
    # --- 火山方舟 (Volcengine Ark) LLM 配置 ---
    
    # LLM 提供者: "ark" (火山方舟) 或 "mock" (本地测试)
    llm_provider: Literal["ark", "mock"] = "ark"
    
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
    
    # 嵌入提供者: "bailian" (阿里云百炼), "ark" (火山方舟), "mock" (本地 Hash)
    embedding_provider: Literal["bailian", "ark", "mock"] = "bailian"
    
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
    
    # MCP 默认用户 ID
    mcp_default_user_id: str = "default"
    
    # MCP 服务器端口 (HTTP 模式)
    mcp_port: int = 8765

