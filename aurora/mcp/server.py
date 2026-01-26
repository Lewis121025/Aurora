"""
Aurora MCP Server Implementation
================================

Implements the Model Context Protocol (MCP) for AI assistant integration.

The server exposes Aurora memory functionality as MCP tools that can be
called by Claude Desktop, Cursor, and other MCP-compatible clients.

Usage:
    from aurora.mcp import create_mcp_server
    
    server = create_mcp_server(aurora_hub)
    await server.serve()
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aurora.hub import AuroraHub
    from aurora.service import AuroraTenant

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Definitions
# =============================================================================

@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    parameters: Dict[str, Any]


AURORA_TOOLS = [
    ToolDefinition(
        name="save_memory",
        description="Save a memory interaction to Aurora. Use this to remember important information, user preferences, decisions made, or any context that might be useful later.",
        parameters={
            "type": "object",
            "properties": {
                "user_message": {
                    "type": "string",
                    "description": "The user's message or input",
                },
                "agent_message": {
                    "type": "string",
                    "description": "The agent's response or action taken",
                },
                "actors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of participants (e.g., ['user', 'agent'])",
                    "default": ["user", "agent"],
                },
                "context": {
                    "type": "string",
                    "description": "Additional context about the interaction",
                },
            },
            "required": ["user_message", "agent_message"],
        },
    ),
    ToolDefinition(
        name="search_memory",
        description="Search Aurora memories by semantic similarity. Returns relevant past interactions, stories, and themes.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 8)",
                    "default": 8,
                    "minimum": 1,
                    "maximum": 50,
                },
                "kinds": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["plot", "story", "theme"]},
                    "description": "Types of memories to search (default: all)",
                },
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="get_narrative",
        description="Get the current self-narrative summary. Provides identity, capabilities, and relationship information.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    ToolDefinition(
        name="check_coherence",
        description="Check memory coherence and get recommendations for resolving conflicts.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    ToolDefinition(
        name="provide_feedback",
        description="Provide feedback on a memory retrieval to improve future searches.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The original search query",
                },
                "chosen_id": {
                    "type": "string",
                    "description": "ID of the chosen/useful memory",
                },
                "success": {
                    "type": "boolean",
                    "description": "Whether the memory was helpful",
                },
            },
            "required": ["query", "chosen_id", "success"],
        },
    ),
]


# =============================================================================
# Resource Definitions
# =============================================================================

@dataclass
class ResourceDefinition:
    """Definition of an MCP resource."""
    uri: str
    name: str
    description: str
    mime_type: str = "application/json"


AURORA_RESOURCES = [
    ResourceDefinition(
        uri="aurora://memory/stats",
        name="Memory Statistics",
        description="Current memory statistics (plot/story/theme counts, coherence score)",
    ),
    ResourceDefinition(
        uri="aurora://memory/narrative",
        name="Self Narrative",
        description="Current self-narrative summary",
    ),
]


# =============================================================================
# MCP Server Implementation
# =============================================================================

class AuroraMCPServer:
    """MCP Server for Aurora memory system.
    
    Implements the Model Context Protocol to expose Aurora functionality
    to AI assistants like Claude Desktop and Cursor.
    """
    
    def __init__(
        self,
        hub: Optional["AuroraHub"] = None,
        tenant: Optional["AuroraTenant"] = None,
        default_user_id: str = "default",
    ):
        """Initialize the MCP server.
        
        Args:
            hub: AuroraHub for multi-tenant access
            tenant: AuroraTenant for single-tenant access
            default_user_id: Default user ID when not specified
        """
        self.hub = hub
        self.tenant = tenant
        self.default_user_id = default_user_id
        
        # Build tool and resource registries
        self.tools = {t.name: t for t in AURORA_TOOLS}
        self.resources = {r.uri: r for r in AURORA_RESOURCES}
    
    def _get_tenant(self, user_id: Optional[str] = None) -> "AuroraTenant":
        """Get tenant for the specified or default user."""
        user_id = user_id or self.default_user_id
        
        if self.tenant is not None:
            return self.tenant
        elif self.hub is not None:
            return self.hub.get(user_id)
        else:
            raise RuntimeError("No hub or tenant configured")
    
    # -------------------------------------------------------------------------
    # MCP Protocol Methods
    # -------------------------------------------------------------------------
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools (MCP tools/list)."""
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
        """Call a tool (MCP tools/call)."""
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
        """List available resources (MCP resources/list)."""
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
        """Read a resource (MCP resources/read)."""
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
    # Tool Handlers
    # -------------------------------------------------------------------------
    
    async def _handle_save_memory(
        self,
        args: Dict[str, Any],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Handle save_memory tool call."""
        tenant = self._get_tenant(user_id)
        
        # Generate event ID
        event_id = str(uuid.uuid4())
        session_id = args.get("session_id", "mcp_session")
        
        # Run in thread pool to not block
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
        """Handle search_memory tool call."""
        tenant = self._get_tenant(user_id)
        
        query = args["query"]
        k = args.get("k", 8)
        
        # Run in thread pool
        def _sync_query():
            return tenant.query(text=query, k=k)
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync_query)
        
        # Filter by kinds if specified
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
        """Handle get_narrative tool call."""
        tenant = self._get_tenant(user_id)
        
        def _sync_narrative():
            return tenant.get_self_narrative()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_narrative)
    
    async def _handle_check_coherence(
        self,
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Handle check_coherence tool call."""
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
        """Handle provide_feedback tool call."""
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
    # Resource Handlers
    # -------------------------------------------------------------------------
    
    async def _read_stats(self, user_id: Optional[str]) -> Dict[str, Any]:
        """Read memory stats resource."""
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
        """Read self-narrative resource."""
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
    # Server Lifecycle
    # -------------------------------------------------------------------------
    
    async def serve_stdio(self) -> None:
        """Serve MCP protocol over stdio.
        
        This is the main entry point for running as an MCP server.
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
        """Handle a JSON-RPC request."""
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
    """Factory function to create MCP server.
    
    Args:
        hub: AuroraHub for multi-tenant access
        tenant: AuroraTenant for single-tenant access
        default_user_id: Default user ID
        
    Returns:
        AuroraMCPServer instance
    """
    return AuroraMCPServer(
        hub=hub,
        tenant=tenant,
        default_user_id=default_user_id,
    )
