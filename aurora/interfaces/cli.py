from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

from aurora.runtime.settings import AuroraSettings
from aurora.runtime.runtime import AuroraRuntime
from aurora.utils.logging import setup_logging
import logging


def _get_runtime(data_dir: Optional[str] = None) -> AuroraRuntime:
    """获取或创建单用户运行时实例"""
    settings = AuroraSettings(data_dir=data_dir or os.environ.get("AURORA_DATA_DIR", "./data"))
    return AuroraRuntime(settings=settings)


def _cmd_demo() -> None:
    """运行最小演示"""
    setup_logging("INFO")
    logger = logging.getLogger("aurora-demo")

    settings = AuroraSettings(data_dir=os.environ.get("AURORA_DATA_DIR", "./demo_data"))
    runtime = AuroraRuntime(settings=settings)

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

    print("1. 摄入交互...")
    for eid, sid, um, am in events:
        r = runtime.ingest_interaction(event_id=eid, session_id=sid, user_message=um, agent_message=am, logger=logger)
        print(f"   - {eid}: plot={r.plot_id}, encoded={r.encoded}, tension={r.tension:.2f}")

    print("\n2. 运行演化...")
    runtime.evolve(logger=logger)
    print(f"   - Stories: {len(runtime.mem.stories)}, Themes: {len(runtime.mem.themes)}")

    print("\n3. 查询记忆...")
    q = "如何避免硬编码阈值并实现叙事检索？"
    res = runtime.query(text=q, k=6)
    print(f"   Query: {q}")
    print(f"   Attractor path length: {res.attractor_path_len}")
    for i, hit in enumerate(res.hits[:3]):
        print(f"   {i+1}. [{hit.kind}] {hit.id}: score={hit.score:.3f}")
        if hit.snippet:
            print(f"      {hit.snippet[:80]}...")

    if res.hits:
        chosen = res.hits[0].id
        runtime.feedback(query_text=q, chosen_id=chosen, success=True)
        print(f"\n4. 记录反馈: chosen={chosen}, success=True")

    print("\n5. 检查一致性...")
    coherence = runtime.check_coherence()
    print(f"   Overall score: {coherence.overall_score:.2f}")
    print(f"   Conflicts: {coherence.conflict_count}")
    if coherence.recommendations:
        print(f"   建议:")
        for rec in coherence.recommendations[:2]:
            print(f"      - {rec}")

    print("\n6. 自我叙事...")
    narrative = runtime.get_self_narrative()
    print(f"   Identity: {narrative['identity_statement']}")
    print(f"   Coherence: {narrative['coherence_score']:.2f}")
    if narrative['capabilities']:
        print("   能力:")
        for name, cap in list(narrative['capabilities'].items())[:3]:
            print(f"      - {name}: {cap['probability']:.2f}")

    print("\n" + "="*60)
    print("演示完成!")
    print("="*60 + "\n")


def _cmd_ingest(args: argparse.Namespace) -> None:
    """摄入单个交互"""
    setup_logging("INFO")
    runtime = _get_runtime(args.data_dir)

    event_id = args.event_id or f"evt_{int(time.time() * 1000)}"
    session_id = args.session_id or "cli_session"
    
    result = runtime.ingest_interaction(
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
    """查询记忆"""
    setup_logging("WARNING")
    runtime = _get_runtime(args.data_dir)
    
    result = runtime.query(text=args.query, k=args.k)
    
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
    """触发演化"""
    setup_logging("INFO")
    logger = logging.getLogger("aurora-evolve")
    runtime = _get_runtime(args.data_dir)
    
    runtime.evolve(logger=logger)
    
    print(json.dumps({
        "plots": len(runtime.mem.plots),
        "stories": len(runtime.mem.stories),
        "themes": len(runtime.mem.themes),
    }, ensure_ascii=False, indent=2))


def _cmd_coherence(args: argparse.Namespace) -> None:
    """检查一致性"""
    setup_logging("WARNING")
    runtime = _get_runtime(args.data_dir)
    
    result = runtime.check_coherence()
    
    print(json.dumps({
        "overall_score": result.overall_score,
        "conflict_count": result.conflict_count,
        "unfinished_story_count": result.unfinished_story_count,
        "recommendations": result.recommendations,
    }, ensure_ascii=False, indent=2))


def _cmd_narrative(args: argparse.Namespace) -> None:
    """获取自我叙事"""
    setup_logging("WARNING")
    runtime = _get_runtime(args.data_dir)
    
    narrative = runtime.get_self_narrative()
    
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
    """获取记忆统计"""
    setup_logging("WARNING")
    runtime = _get_runtime(args.data_dir)
    
    coherence = runtime.check_coherence()
    narrative = runtime.get_self_narrative()
    
    print(json.dumps({
        "plot_count": len(runtime.mem.plots),
        "story_count": len(runtime.mem.stories),
        "theme_count": len(runtime.mem.themes),
        "coherence_score": coherence.overall_score,
        "self_narrative_coherence": narrative["coherence_score"],
        "capability_count": len(narrative["capabilities"]),
        "relationship_count": len(narrative["relationships"]),
    }, ensure_ascii=False, indent=2))


def _cmd_causal(args: argparse.Namespace) -> None:
    """获取因果链"""
    setup_logging("WARNING")
    runtime = _get_runtime(args.data_dir)
    
    chain = runtime.get_causal_chain(args.node_id, args.direction)
    
    print(json.dumps({
        "node_id": args.node_id,
        "direction": args.direction,
        "chain": chain,
    }, ensure_ascii=False, indent=2))


def _cmd_serve(args: argparse.Namespace) -> None:
    """启动API服务器"""
    try:
        import uvicorn
    except ImportError:
        print("错误: uvicorn未安装。使用以下命令安装: pip install -e '.[api]'")
        return

    os.environ["AURORA_DATA_DIR"] = args.data_dir or "./data"

    print(f"在 {args.host}:{args.port} 启动AURORA API服务器")
    uvicorn.run(
        "aurora.interfaces.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aurora",
        description="AURORA 记忆系统 CLI"
    )
    parser.add_argument("--data-dir", type=str, help="Data directory")

    sub = parser.add_subparsers(dest="cmd")

    # 演示命令
    sub.add_parser("demo", help="运行最小演示")

    # 摄入命令
    ingest_p = sub.add_parser("ingest", help="摄入一个交互")
    ingest_p.add_argument("--user-message", "-m", required=True, help="用户消息")
    ingest_p.add_argument("--agent-message", "-a", required=True, help="代理消息")
    ingest_p.add_argument("--event-id", "-e", help="事件ID (如果未提供则自动生成)")
    ingest_p.add_argument("--session-id", "-s", help="会话ID")

    # 查询命令
    query_p = sub.add_parser("query", help="查询记忆")
    query_p.add_argument("--query", "-q", required=True, help="查询文本")
    query_p.add_argument("--k", "-k", type=int, default=8, help="返回结果数量")

    # 演化命令
    evolve_p = sub.add_parser("evolve", help="触发记忆演化")

    # 一致性命令
    coherence_p = sub.add_parser("coherence", help="检查记忆一致性")

    # 叙事命令
    narrative_p = sub.add_parser("narrative", help="获取自我叙事")
    narrative_p.add_argument("--full", "-f", action="store_true", help="输出完整叙事")

    # 统计命令
    stats_p = sub.add_parser("stats", help="获取记忆统计")

    # 因果命令
    causal_p = sub.add_parser("causal", help="获取因果链")
    causal_p.add_argument("--node-id", "-n", required=True, help="节点ID")
    causal_p.add_argument("--direction", "-d", choices=["ancestors", "descendants"], default="ancestors")

    # 服务命令
    serve_p = sub.add_parser("serve", help="启动API服务器")
    serve_p.add_argument("--host", default="0.0.0.0", help="绑定的主机")
    serve_p.add_argument("--port", "-p", type=int, default=8000, help="绑定的端口")
    serve_p.add_argument("--reload", action="store_true", help="启用自动重载")

    args = parser.parse_args()

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
