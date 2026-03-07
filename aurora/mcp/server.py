"""
Aurora MCP 服务器实现
================================

实现用于 AI 助手集成的模型上下文协议 (MCP)。

服务器将 Aurora 内存功能公开为 MCP 工具，可以由
Claude Desktop、Cursor 和其他 MCP 兼容客户端调用。

使用方法:
    from aurora.mcp import create_mcp_server

    server = create_mcp_server(aurora_hub)
    await server.serve()
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aurora.hub import AuroraHub
    from aurora.service import AuroraTenant

logger = logging.getLogger(__name__)


# =============================================================================
# 工具定义
# =============================================================================

@dataclass
class ToolDefinition:
    """MCP 工具的定义。"""
    name: str
    description: str
    parameters: Dict[str, Any]


AURORA_TOOLS = [
    ToolDefinition(
        name="save_memory",
        description="将内存交互保存到 Aurora。使用此功能记住重要信息、用户偏好、做出的决定或任何可能稍后有用的上下文。",
        parameters={
            "type": "object",
            "properties": {
                "user_message": {
                    "type": "string",
                    "description": "用户的消息或输入",
                },
                "agent_message": {
                    "type": "string",
                    "description": "代理的响应或采取的行动",
                },
                "actors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "参与者列表（例如 ['user', 'agent']）",
                    "default": ["user", "agent"],
                },
                "context": {
                    "type": "string",
                    "description": "关于交互的其他上下文",
                },
            },
            "required": ["user_message", "agent_message"],
        },
    ),
    ToolDefinition(
        name="search_memory",
        description="按语义相似度搜索 Aurora 内存。返回相关的过去交互、故事和主题。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "自然语言搜索查询",
                },
                "k": {
                    "type": "integer",
                    "description": "要返回的结果数（默认: 8）",
                    "default": 8,
                    "minimum": 1,
                    "maximum": 50,
                },
                "kinds": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["plot", "story", "theme"]},
                    "description": "要搜索的内存类型（默认: 全部）",
                },
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="get_narrative",
        description="获取当前自叙述摘要。提供身份、能力和关系信息。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    ToolDefinition(
        name="check_coherence",
        description="检查内存一致性并获取解决冲突的建议。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    ToolDefinition(
        name="provide_feedback",
        description="提供关于内存检索的反馈以改进未来的搜索。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "原始搜索查询",
                },
                "chosen_id": {
                    "type": "string",
                    "description": "选择/有用的内存的 ID",
                },
                "success": {
                    "type": "boolean",
                    "description": "内存是否有帮助",
                },
            },
            "required": ["query", "chosen_id", "success"],
        },
    ),
]


# =============================================================================
# 资源定义
# =============================================================================

@dataclass
class ResourceDefinition:
    """MCP 资源的定义。"""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


AURORA_RESOURCES = [
    ResourceDefinition(
        uri="aurora://memory/stats",
        name="Memory Statistics",
        description="当前内存统计（plot/story/theme 计数、一致性分数）",
    ),
    ResourceDefinition(
        uri="aurora://memory/narrative",
        name="Self Narrative",
        description="当前自叙述摘要",
    ),
]


# =============================================================================
# MCP Server Implementation
# =============================================================================

class AuroraMCPServer:
    """Aurora 内存系统的 MCP 服务器。

    实现模型上下文协议以将 Aurora 功能公开
    给 Claude Desktop 和 Cursor 等 AI 助手。
    """

    def __init__(
        self,
        hub: Optional["AuroraHub"] = None,
        tenant: Optional["AuroraTenant"] = None,
        default_user_id: str = "default",
    ):
        """初始化 MCP 服务器。

        Args:
            hub: 用于多租户访问的 AuroraHub
            tenant: 用于单租户访问的 AuroraTenant
            default_user_id: 未指定时的默认用户 ID
        """
        self.hub = hub
        self.tenant = tenant
        self.default_user_id = default_user_id

        # 构建工具和资源注册表
        self.tools = {t.name: t for t in AURORA_TOOLS}
        self.resources = {r.uri: r for r in AURORA_RESOURCES}

    def _get_tenant(self, user_id: Optional[str] = None) -> "AuroraTenant":
        """获取指定或默认用户的租户。"""
        user_id = user_id or self.default_user_id

        if self.tenant is not None:
            return self.tenant
        elif self.hub is not None:
            return self.hub.tenant(user_id)
        else:
            raise RuntimeError("No hub or tenant configured")

    # -------------------------------------------------------------------------
    # MCP 协议方法
    # -------------------------------------------------------------------------

    async def list_tools(self) -> List[Dict[str, Any]]:
        """列出可用工具（MCP tools/list）。"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters,
            }
            for tool in self.tools.values()
        ]

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """调用工具（MCP tools/call）。"""
        if name not in self.tools:
            return {"error": f"Unknown tool: {name}"}

        try:
            if name == "save_memory":
                return await self._handle_save_memory(arguments, user_id)
            elif name == "search_memory":
                return await self._handle_search_memory(arguments, user_id)
            elif name == "get_narrative":
                return await self._handle_get_narrative(user_id)
            elif name == "check_coherence":
                return await self._handle_check_coherence(user_id)
            elif name == "provide_feedback":
                return await self._handle_provide_feedback(arguments, user_id)
            else:
                return {"error": f"Tool not implemented: {name}"}
        except Exception as e:
            logger.exception(f"Tool call failed: {name}")
            return {"error": str(e)}

    async def list_resources(self) -> List[Dict[str, Any]]:
        """列出可用资源（MCP resources/list）。"""
        return [
            {
                "uri": resource.uri,
                "name": resource.name,
                "description": resource.description,
                "mimeType": resource.mime_type,
            }
            for resource in self.resources.values()
        ]

    async def read_resource(
        self,
        uri: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """读取资源（MCP resources/read）。"""
        if uri not in self.resources:
            return {"error": f"Unknown resource: {uri}"}

        try:
            if uri == "aurora://memory/stats":
                return await self._read_stats(user_id)
            elif uri == "aurora://memory/narrative":
                return await self._read_narrative(user_id)
            else:
                return {"error": f"Resource not implemented: {uri}"}
        except Exception as e:
            logger.exception(f"Resource read failed: {uri}")
            return {"error": str(e)}
    
    # -------------------------------------------------------------------------
    # 工具处理程序
    # -------------------------------------------------------------------------

    async def _handle_save_memory(
        self,
        args: Dict[str, Any],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """处理 save_memory 工具调用。"""
        tenant = self._get_tenant(user_id)

        # 生成事件 ID
        event_id = str(uuid.uuid4())
        session_id = args.get("session_id", "mcp_session")

        # 在线程池中运行以避免阻塞
        def _sync_ingest():
            return tenant.ingest_interaction(
                event_id=event_id,
                session_id=session_id,
                user_message=args["user_message"],
                agent_message=args["agent_message"],
                actors=args.get("actors"),
                context=args.get("context"),
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync_ingest)

        return {
            "success": True,
            "event_id": event_id,
            "plot_id": result.plot_id,
            "story_id": result.story_id,
            "encoded": result.encoded,
            "tension": result.tension,
        }

    async def _handle_search_memory(
        self,
        args: Dict[str, Any],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """处理 search_memory 工具调用。"""
        tenant = self._get_tenant(user_id)

        query = args["query"]
        k = args.get("k", 8)

        # 在线程池中运行
        def _sync_query():
            return tenant.query(text=query, k=k)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync_query)

        # 如果指定，按类型过滤
        kinds_filter = set(args.get("kinds", [])) if args.get("kinds") else None

        hits = []
        for hit in result.hits:
            if kinds_filter and hit.kind not in kinds_filter:
                continue
            hits.append({
                "id": hit.id,
                "kind": hit.kind,
                "score": hit.score,
                "snippet": hit.snippet,
            })

        return {
            "query": query,
            "hits": hits,
            "attractor_path_len": result.attractor_path_len,
        }

    async def _handle_get_narrative(
        self,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """处理 get_narrative 工具调用。"""
        tenant = self._get_tenant(user_id)

        def _sync_narrative():
            return tenant.get_self_narrative()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_narrative)

    async def _handle_check_coherence(
        self,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """处理 check_coherence 工具调用。"""
        tenant = self._get_tenant(user_id)

        def _sync_coherence():
            return tenant.check_coherence()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync_coherence)

        return {
            "overall_score": result.overall_score,
            "conflict_count": result.conflict_count,
            "unfinished_story_count": result.unfinished_story_count,
            "recommendations": result.recommendations,
        }

    async def _handle_provide_feedback(
        self,
        args: Dict[str, Any],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """处理 provide_feedback 工具调用。"""
        tenant = self._get_tenant(user_id)

        def _sync_feedback():
            tenant.feedback(
                query_text=args["query"],
                chosen_id=args["chosen_id"],
                success=args["success"],
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _sync_feedback)

        return {"success": True}

    # -------------------------------------------------------------------------
    # 资源处理程序
    # -------------------------------------------------------------------------

    async def _read_stats(self, user_id: Optional[str]) -> Dict[str, Any]:
        """读取内存统计资源。"""
        tenant = self._get_tenant(user_id)

        mem = tenant.mem
        coherence = tenant.check_coherence()

        return {
            "contents": [
                {
                    "uri": "aurora://memory/stats",
                    "mimeType": "application/json",
                    "text": json.dumps({
                        "plot_count": len(mem.plots),
                        "story_count": len(mem.stories),
                        "theme_count": len(mem.themes),
                        "coherence_score": coherence.overall_score,
                        "gate_pass_rate": mem.gate.pass_rate(),
                    }),
                }
            ]
        }

    async def _read_narrative(self, user_id: Optional[str]) -> Dict[str, Any]:
        """读取自叙述资源。"""
        narrative = await self._handle_get_narrative(user_id)

        return {
            "contents": [
                {
                    "uri": "aurora://memory/narrative",
                    "mimeType": "application/json",
                    "text": json.dumps(narrative),
                }
            ]
        }

    # -------------------------------------------------------------------------
    # 服务器生命周期
    # -------------------------------------------------------------------------

    async def serve_stdio(self) -> None:
        """通过 stdio 提供 MCP 协议。

        这是作为 MCP 服务器运行的主入口点。
        """
        import sys

        logger.info("Starting Aurora MCP server on stdio")

        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)

        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.Protocol, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, asyncio.get_event_loop())

        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                try:
                    request = json.loads(line.decode("utf-8"))
                    response = await self._handle_jsonrpc(request)
                    writer.write((json.dumps(response) + "\n").encode("utf-8"))
                    await writer.drain()
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.exception("MCP server error")

    async def _handle_jsonrpc(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理 JSON-RPC 请求。"""
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id")

        result = None
        error = None

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                    },
                    "serverInfo": {
                        "name": "aurora-memory",
                        "version": "0.2.0",
                    },
                }
            elif method == "tools/list":
                result = {"tools": await self.list_tools()}
            elif method == "tools/call":
                result = await self.call_tool(
                    params.get("name", ""),
                    params.get("arguments", {}),
                )
            elif method == "resources/list":
                result = {"resources": await self.list_resources()}
            elif method == "resources/read":
                result = await self.read_resource(params.get("uri", ""))
            else:
                error = {"code": -32601, "message": f"Method not found: {method}"}
        except Exception as e:
            error = {"code": -32603, "message": str(e)}

        response = {"jsonrpc": "2.0", "id": request_id}
        if error:
            response["error"] = error
        else:
            response["result"] = result

        return response


def create_mcp_server(
    hub: Optional["AuroraHub"] = None,
    tenant: Optional["AuroraTenant"] = None,
    default_user_id: str = "default",
) -> AuroraMCPServer:
    """创建 MCP 服务器的工厂函数。

    Args:
        hub: 用于多租户访问的 AuroraHub
        tenant: 用于单租户访问的 AuroraTenant
        default_user_id: 默认用户 ID

    Returns:
        AuroraMCPServer 实例
    """
    return AuroraMCPServer(
        hub=hub,
        tenant=tenant,
        default_user_id=default_user_id,
    )
