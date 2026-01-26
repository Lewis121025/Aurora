"""
Aurora MCP (Model Context Protocol) Server
==========================================

Provides MCP-compliant interface for AI assistants (Claude Desktop, Cursor, etc.)
to interact with Aurora memory system.

Tools:
- save_memory: Save a memory interaction
- search_memory: Search memories by semantic similarity
- get_narrative: Get self-narrative summary
- check_coherence: Check memory coherence

Resources:
- aurora://memory/stats: Memory statistics
- aurora://memory/narrative: Current self-narrative
"""

from aurora.mcp.server import AuroraMCPServer, create_mcp_server

__all__ = [
    "AuroraMCPServer",
    "create_mcp_server",
]
