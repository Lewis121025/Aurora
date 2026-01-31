"""
AURORA Memory System
====================

A narrative-first memory system for AI agents.

Philosophy:
    - Memory is not about the past, but about identity
    - The self is a necessary fiction
    - Relationships define who we are
    - Some contradictions should be preserved

Quick Start:
    from aurora import AuroraMemory, MemoryConfig
    
    mem = AuroraMemory(cfg=MemoryConfig(), seed=42)
    
    # Ingest interactions
    plot = mem.ingest("User asked about memory systems")
    
    # Query with relationship context
    trace = mem.query("How does memory work?", asker_id="user_123")
    
    # Evolve (consolidate, reflect, grow)
    mem.evolve()

Modules:
    algorithms - Core memory algorithms (AuroraMemory, TensionManager)
    services   - Production services (IngestionService, QueryService)
    storage    - Storage backends (StateStore, VectorStore)
    embeddings - Embedding providers (HashEmbedding, BailianEmbedding)
    llm        - LLM providers (MockLLM, ArkLLM)
    api        - REST API (FastAPI)
    mcp        - MCP Server (Model Context Protocol)
    privacy    - PII redaction
"""

__version__ = "0.1.0"

# Core algorithm - the heart of AURORA
from aurora.algorithms import AuroraMemory, TensionManager

# Data models
from aurora.algorithms.models import (
    Plot,
    StoryArc,
    Theme,
    MemoryConfig,
    RetrievalTrace,
)

# High-level integration
from aurora.hub import AuroraHub
from aurora.config import AuroraSettings

__all__ = [
    # Version
    "__version__",
    # Core
    "AuroraMemory",
    "TensionManager",
    # Models
    "Plot",
    "StoryArc", 
    "Theme",
    "MemoryConfig",
    "RetrievalTrace",
    # Hub
    "AuroraHub",
    "AuroraSettings",
]
