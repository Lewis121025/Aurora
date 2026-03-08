"""
Aurora MCP（模型上下文协议）服务器
==========================================

为 AI 助手（Claude Desktop、Cursor 等）提供 MCP 兼容接口
与 Aurora 内存系统交互。

工具:
- save_memory: 保存内存交互
- search_memory: 按语义相似度搜索内存
- get_narrative: 获取自叙述摘要
- check_coherence: 检查内存一致性

资源:
- aurora://memory/stats: 内存统计
- aurora://memory/narrative: 当前自叙述
"""

from aurora.interfaces.mcp.server import AuroraMCPServer, create_mcp_server

__all__ = [
    "AuroraMCPServer",
    "create_mcp_server",
]
