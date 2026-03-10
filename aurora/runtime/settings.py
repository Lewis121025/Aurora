"""
aurora/runtime/settings.py
全局配置模块：利用 pydantic-settings 定义了 Aurora V4 系统的所有可配置项。
支持通过环境变量（前缀 AURORA_）或 .env 文件进行注入。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# 默认数据存储目录
DEFAULT_DATA_DIR = ".aurora"


class AuroraSettings(BaseSettings):
    """
    Aurora V4 运行时配置类。
    涵盖了存储、心理引擎、LLM 提供者、Embedding 以及性能参数。
    """

    model_config = SettingsConfigDict(
        env_prefix="AURORA_",  # 环境变量前缀
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- 基础路径与持久化 ---
    data_dir: str = DEFAULT_DATA_DIR
    event_log_filename: str = "events.sqlite3"
    snapshot_dirname: str = "snapshots"
    snapshot_every_events: int = 1  # 每摄入多少次事件执行一次快照
    memory_seed: int = 42  # 随机种子，保证动力学模拟的可复现性

    evolve_every_seconds: int = 3600  # 后台演化周期

    # --- 灵魂引擎核心参数 ---
    dim: int = 1024  # 向量维度（需与 Embedding 模型匹配）
    metric_rank: int = 64  # 度量学习矩阵的秩
    max_plots: int = 5000  # 长期记忆最大存储条数
    kde_reservoir: int = 4096  # 惊讶度计算的蓄水池大小
    story_alpha: float = 1.0  # CRP 故事聚类参数
    theme_alpha: float = 0.6  # CRP 主题聚类参数
    subconscious_reservoir: int = 1024  # 潜意识碎片池大小
    mode_refractory_steps: int = 4  # 模式切换的冷却步数
    mode_new_threshold: float = 0.52  # 发现新模式的失调阈值
    encode_min_events_before_gating: int = 6  # 开启门控前的最少事件数
    max_recent_texts: int = 12  # 最近交互上下文保留条数
    axis_merge_every_events: int = 50  # 轴合并检查频率
    persona_axis_budget: int = 24  # 允许存在的人设轴最大数量
    dreams_per_evolve: int = 2  # 每次演化产生的梦境数量
    profile_text: str = ""  # 初始人设自然语言描述
    persona_axes_json: Optional[str] = None  # 预设的性格轴配置（JSON 格式）

    # --- 语义提供者配置 ---
    meaning_provider: Literal["heuristic", "llm"] = "llm"
    narrative_provider: Literal["heuristic", "llm"] = "llm"

    # --- 门控与度量学习超参 ---
    gate_forgetting_factor: float = 0.99
    metric_window_size: int = 10000
    metric_decay_factor: float = 0.5

    # --- 大语言模型 (LLM) 配置 ---
    llm_provider: Optional[Literal["bailian", "ark"]] = None

    # 字节跳动火山引擎 (Ark)
    ark_api_key: Optional[str] = None
    ark_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    ark_llm_model: str = "doubao-seed-1-8-251228"

    # 阿里云百炼 (Bailian)
    bailian_llm_api_key: Optional[str] = None
    bailian_llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    bailian_llm_model: str = "qwen3.5-plus"

    llm_timeout: float = 60.0
    llm_max_retries: int = 3

    # --- Embedding 模型配置 ---
    embedding_provider: Literal["bailian", "ark", "local", "hash"] = "local"
    # 可选：为人设轴使用不同的 Embedding
    axis_embedding_provider: Optional[Literal["bailian", "ark", "local", "hash"]] = None

    bailian_embedding_api_key: Optional[str] = None
    bailian_embedding_model: str = "text-embedding-v4"
    bailian_embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    embedding_cache_enabled: bool = True  # 开启向量缓存以节省 API 费用
    embedding_cache_size: int = 10000

    # --- 外部接口配置 ---
    mcp_port: int = 8765  # Model Context Protocol 端口
