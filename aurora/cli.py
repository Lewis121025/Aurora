from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

from aurora.config import AuroraSettings
from aurora.hub import AuroraHub
from aurora.utils.logging import setup_logging
import logging


def _get_hub(data_dir: Optional[str] = None) -> AuroraHub:
    """Get or create hub instance"""
    settings = AuroraSettings(data_dir=data_dir or os.environ.get("AURORA_DATA_DIR", "./data"))
    return AuroraHub(settings=settings)


def _cmd_demo() -> None:
    """Run a minimal demo"""
    setup_logging("INFO")
    logger = logging.getLogger("aurora-demo")

    settings = AuroraSettings(data_dir=os.environ.get("AURORA_DATA_DIR", "./demo_data"))
    hub = AuroraHub(settings=settings)

    user_id = "u1"
    t = hub.tenant(user_id)

    events = [
        ("e1", "s1", "我想做一个记忆系统。", "好的，我们先定义需求与评估指标。"),
        ("e2", "s1", "我不想写阈值和硬编码权重。", "我们可以用贝叶斯策略与非参数聚类来替代。"),
        ("e3", "s1", "检索时希望能返回因果链。", "可以用图扩散与因果边的后验来重建证据链。"),
        ("e4", "s1", "系统需要保持记忆的一致性。", "我们会使用一致性守护模块来检测和解决矛盾。"),
        ("e5", "s1", "AI需要理解自己的能力边界。", "通过自我叙事模块，AI会学习自己的能力和局限。"),
    ]

    print("\n" + "="*60)
    print("AURORA Memory System Demo")
    print("="*60 + "\n")

    print("1. Ingesting interactions...")
    for eid, sid, um, am in events:
        r = t.ingest_interaction(event_id=eid, session_id=sid, user_message=um, agent_message=am, logger=logger)
        print(f"   - {eid}: plot={r.plot_id}, encoded={r.encoded}, tension={r.tension:.2f}")

    print("\n2. Running evolution...")
    t.evolve(logger=logger)
    print(f"   - Stories: {len(t.mem.stories)}, Themes: {len(t.mem.themes)}")

    print("\n3. Querying memory...")
    q = "如何避免硬编码阈值并实现叙事检索？"
    res = t.query(text=q, k=6)
    print(f"   Query: {q}")
    print(f"   Attractor path length: {res.attractor_path_len}")
    for i, hit in enumerate(res.hits[:3]):
        print(f"   {i+1}. [{hit.kind}] {hit.id}: score={hit.score:.3f}")
        if hit.snippet:
            print(f"      {hit.snippet[:80]}...")

    if res.hits:
        chosen = res.hits[0].id
        t.feedback(query_text=q, chosen_id=chosen, success=True)
        print(f"\n4. Recorded feedback: chosen={chosen}, success=True")

    print("\n5. Checking coherence...")
    coherence = t.check_coherence()
    print(f"   Overall score: {coherence.overall_score:.2f}")
    print(f"   Conflicts: {coherence.conflict_count}")
    if coherence.recommendations:
        print(f"   Recommendations:")
        for rec in coherence.recommendations[:2]:
            print(f"      - {rec}")

    print("\n6. Self-narrative...")
    narrative = t.get_self_narrative()
    print(f"   Identity: {narrative['identity_statement']}")
    print(f"   Coherence: {narrative['coherence_score']:.2f}")
    if narrative['capabilities']:
        print("   Capabilities:")
        for name, cap in list(narrative['capabilities'].items())[:3]:
            print(f"      - {name}: {cap['probability']:.2f}")

    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60 + "\n")


def _cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a single interaction"""
    setup_logging("INFO")
    hub = _get_hub(args.data_dir)
    t = hub.tenant(args.user_id)
    
    event_id = args.event_id or f"evt_{int(time.time() * 1000)}"
    session_id = args.session_id or "cli_session"
    
    result = t.ingest_interaction(
        event_id=event_id,
        session_id=session_id,
        user_message=args.user_message,
        agent_message=args.agent_message,
    )
    
    print(json.dumps({
        "event_id": result.event_id,
        "plot_id": result.plot_id,
        "story_id": result.story_id,
        "encoded": result.encoded,
        "tension": result.tension,
        "surprise": result.surprise,
    }, ensure_ascii=False, indent=2))


def _cmd_query(args: argparse.Namespace) -> None:
    """Query memory"""
    setup_logging("WARNING")
    hub = _get_hub(args.data_dir)
    t = hub.tenant(args.user_id)
    
    result = t.query(text=args.query, k=args.k)
    
    output = {
        "query": result.query,
        "attractor_path_len": result.attractor_path_len,
        "hits": [
            {
                "id": h.id,
                "kind": h.kind,
                "score": h.score,
                "snippet": h.snippet,
            }
            for h in result.hits
        ]
    }
    
    print(json.dumps(output, ensure_ascii=False, indent=2))


def _cmd_evolve(args: argparse.Namespace) -> None:
    """Trigger evolution"""
    setup_logging("INFO")
    logger = logging.getLogger("aurora-evolve")
    hub = _get_hub(args.data_dir)
    t = hub.tenant(args.user_id)
    
    t.evolve(logger=logger)
    
    print(json.dumps({
        "user_id": args.user_id,
        "plots": len(t.mem.plots),
        "stories": len(t.mem.stories),
        "themes": len(t.mem.themes),
    }, ensure_ascii=False, indent=2))


def _cmd_coherence(args: argparse.Namespace) -> None:
    """Check coherence"""
    setup_logging("WARNING")
    hub = _get_hub(args.data_dir)
    t = hub.tenant(args.user_id)
    
    result = t.check_coherence()
    
    print(json.dumps({
        "user_id": args.user_id,
        "overall_score": result.overall_score,
        "conflict_count": result.conflict_count,
        "unfinished_story_count": result.unfinished_story_count,
        "recommendations": result.recommendations,
    }, ensure_ascii=False, indent=2))


def _cmd_narrative(args: argparse.Namespace) -> None:
    """Get self-narrative"""
    setup_logging("WARNING")
    hub = _get_hub(args.data_dir)
    t = hub.tenant(args.user_id)
    
    narrative = t.get_self_narrative()
    
    if args.full:
        print(narrative["full_narrative"])
    else:
        print(json.dumps({
            "identity_statement": narrative["identity_statement"],
            "capability_narrative": narrative["capability_narrative"],
            "coherence_score": narrative["coherence_score"],
            "capabilities": narrative["capabilities"],
            "unresolved_tensions": narrative["unresolved_tensions"],
        }, ensure_ascii=False, indent=2))


def _cmd_stats(args: argparse.Namespace) -> None:
    """Get memory statistics"""
    setup_logging("WARNING")
    hub = _get_hub(args.data_dir)
    t = hub.tenant(args.user_id)
    
    coherence = t.check_coherence()
    narrative = t.get_self_narrative()
    
    print(json.dumps({
        "user_id": args.user_id,
        "plot_count": len(t.mem.plots),
        "story_count": len(t.mem.stories),
        "theme_count": len(t.mem.themes),
        "coherence_score": coherence.overall_score,
        "self_narrative_coherence": narrative["coherence_score"],
        "capability_count": len(narrative["capabilities"]),
        "relationship_count": len(narrative["relationships"]),
    }, ensure_ascii=False, indent=2))


def _cmd_causal(args: argparse.Namespace) -> None:
    """Get causal chain"""
    setup_logging("WARNING")
    hub = _get_hub(args.data_dir)
    t = hub.tenant(args.user_id)
    
    chain = t.get_causal_chain(args.node_id, args.direction)
    
    print(json.dumps({
        "node_id": args.node_id,
        "direction": args.direction,
        "chain": chain,
    }, ensure_ascii=False, indent=2))


def _cmd_serve(args: argparse.Namespace) -> None:
    """Start API server"""
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn not installed. Install with: pip install uvicorn")
        return
    
    os.environ["AURORA_DATA_DIR"] = args.data_dir or "./data"
    
    print(f"Starting AURORA API server on {args.host}:{args.port}")
    uvicorn.run(
        "aurora.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aurora",
        description="AURORA Memory System CLI"
    )
    parser.add_argument("--data-dir", type=str, help="Data directory")
    
    sub = parser.add_subparsers(dest="cmd")
    
    # Demo command
    sub.add_parser("demo", help="Run a minimal demo")
    
    # Ingest command
    ingest_p = sub.add_parser("ingest", help="Ingest an interaction")
    ingest_p.add_argument("--user-id", "-u", required=True, help="User ID")
    ingest_p.add_argument("--user-message", "-m", required=True, help="User message")
    ingest_p.add_argument("--agent-message", "-a", required=True, help="Agent message")
    ingest_p.add_argument("--event-id", "-e", help="Event ID (auto-generated if not provided)")
    ingest_p.add_argument("--session-id", "-s", help="Session ID")
    
    # Query command
    query_p = sub.add_parser("query", help="Query memory")
    query_p.add_argument("--user-id", "-u", required=True, help="User ID")
    query_p.add_argument("--query", "-q", required=True, help="Query text")
    query_p.add_argument("--k", "-k", type=int, default=8, help="Number of results")
    
    # Evolve command
    evolve_p = sub.add_parser("evolve", help="Trigger memory evolution")
    evolve_p.add_argument("--user-id", "-u", required=True, help="User ID")
    
    # Coherence command
    coherence_p = sub.add_parser("coherence", help="Check memory coherence")
    coherence_p.add_argument("--user-id", "-u", required=True, help="User ID")
    
    # Narrative command
    narrative_p = sub.add_parser("narrative", help="Get self-narrative")
    narrative_p.add_argument("--user-id", "-u", required=True, help="User ID")
    narrative_p.add_argument("--full", "-f", action="store_true", help="Output full narrative")
    
    # Stats command
    stats_p = sub.add_parser("stats", help="Get memory statistics")
    stats_p.add_argument("--user-id", "-u", required=True, help="User ID")
    
    # Causal command
    causal_p = sub.add_parser("causal", help="Get causal chain")
    causal_p.add_argument("--user-id", "-u", required=True, help="User ID")
    causal_p.add_argument("--node-id", "-n", required=True, help="Node ID")
    causal_p.add_argument("--direction", "-d", choices=["ancestors", "descendants"], default="ancestors")
    
    # Serve command
    serve_p = sub.add_parser("serve", help="Start API server")
    serve_p.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_p.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    serve_p.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Store data_dir in args for commands that need it
    if hasattr(args, 'data_dir') is False or args.data_dir is None:
        args.data_dir = parser.parse_args().data_dir
    
    if args.cmd == "demo":
        _cmd_demo()
    elif args.cmd == "ingest":
        _cmd_ingest(args)
    elif args.cmd == "query":
        _cmd_query(args)
    elif args.cmd == "evolve":
        _cmd_evolve(args)
    elif args.cmd == "coherence":
        _cmd_coherence(args)
    elif args.cmd == "narrative":
        _cmd_narrative(args)
    elif args.cmd == "stats":
        _cmd_stats(args)
    elif args.cmd == "causal":
        _cmd_causal(args)
    elif args.cmd == "serve":
        _cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
