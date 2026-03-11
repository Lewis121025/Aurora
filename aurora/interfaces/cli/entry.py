from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

from aurora.interfaces.terminal.observer import run_observer
from aurora.runtime.runtime import AuroraRuntime
from aurora.runtime.settings import AuroraSettings, DEFAULT_DATA_DIR


def _get_runtime(data_dir: Optional[str] = None) -> AuroraRuntime:
    settings = AuroraSettings(data_dir=data_dir or os.environ.get("AURORA_DATA_DIR", DEFAULT_DATA_DIR))
    return AuroraRuntime(settings=settings)


def _close_runtime(runtime: object) -> None:
    close = getattr(runtime, "close", None)
    if callable(close):
        close()


def _cmd_ingest(args: argparse.Namespace) -> None:
    runtime = _get_runtime(args.data_dir)
    try:
        receipt = runtime.accept_interaction(
            event_id=args.event_id or "evt_cli_ingest",
            session_id=args.session_id or "cli",
            user_message=args.user_message,
            agent_message=args.agent_message,
            actors=args.actors,
            context=args.context,
            ts=args.ts,
        )
        print(json.dumps(receipt.__dict__, ensure_ascii=False, indent=2))
    finally:
        _close_runtime(runtime)


def _cmd_query(args: argparse.Namespace) -> None:
    runtime = _get_runtime(args.data_dir)
    try:
        result = runtime.query(text=args.query, k=args.k, session_id=args.session_id)
        print(
            json.dumps(
                {
                    "query": result.query,
                    "attractor_path_len": result.attractor_path_len,
                    "overlay_hit_count": result.overlay_hit_count,
                    "hits": [hit.__dict__ for hit in result.hits],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        _close_runtime(runtime)


def _cmd_respond(args: argparse.Namespace) -> None:
    runtime = _get_runtime(args.data_dir)
    try:
        result = runtime.respond(
            session_id=args.session_id or "cli",
            user_message=args.user_message,
            context=args.context,
            actors=args.actors,
            k=args.k,
            ts=args.ts,
        )
        print(
            json.dumps(
                {
                    "reply": result.reply,
                    "event_id": result.event_id,
                    "mode": result.memory_context.mode,
                    "intuition": result.memory_context.intuition,
                    "salient_axes": result.memory_context.narrative_summary.salient_axes,
                    "retrieval_hits": result.memory_context.retrieval_hits,
                    "overlay_hits": result.memory_context.overlay_hits,
                    "persistence": result.persistence.__dict__,
                    "timings": result.timings.__dict__,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        _close_runtime(runtime)


def _cmd_identity(args: argparse.Namespace) -> None:
    runtime = _get_runtime(args.data_dir)
    try:
        report = runtime.get_identity()
        if args.full:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return
        print(
            json.dumps(
                {
                    "current_mode": report["identity"]["current_mode"],
                    "pressure": report["narrative_summary"]["pressure"],
                    "summary": report["narrative_summary"]["text"],
                    "salient_axes": report["narrative_summary"]["salient_axes"],
                    "narrative_tail": report["identity"]["narrative_tail"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        _close_runtime(runtime)


def _cmd_stats(args: argparse.Namespace) -> None:
    runtime = _get_runtime(args.data_dir)
    try:
        print(json.dumps(runtime.get_stats(), ensure_ascii=False, indent=2))
    finally:
        _close_runtime(runtime)


def _cmd_event(args: argparse.Namespace) -> None:
    runtime = _get_runtime(args.data_dir)
    try:
        print(json.dumps(runtime.get_event_status(args.event_id), ensure_ascii=False, indent=2))
    finally:
        _close_runtime(runtime)


def _cmd_job(args: argparse.Namespace) -> None:
    runtime = _get_runtime(args.data_dir)
    try:
        print(json.dumps(runtime.get_job_status(args.job_id), ensure_ascii=False, indent=2))
    finally:
        _close_runtime(runtime)


def _cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("错误: uvicorn未安装。使用以下命令安装: pip install -e '.[api]'")
        return
    os.environ["AURORA_DATA_DIR"] = args.data_dir or DEFAULT_DATA_DIR
    uvicorn.run("aurora.interfaces.api.app:app", host=args.host, port=args.port, reload=args.reload)


def _cmd_observe(args: argparse.Namespace) -> None:
    run_observer(
        data_dir=args.data_dir,
        session_id=args.session_id,
        max_hits=args.max_hits,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="aurora",
        description="Aurora Soul CLI。无子命令时直接进入全屏终端对话模式。",
    )
    parser.add_argument("--data-dir", type=str, help="Data directory")
    sub = parser.add_subparsers(dest="cmd")

    ingest_p = sub.add_parser("ingest", help="接收一条交互并入队投影")
    ingest_p.add_argument("user_message")
    ingest_p.add_argument("agent_message")
    ingest_p.add_argument("--event-id")
    ingest_p.add_argument("--session-id")
    ingest_p.add_argument("--context")
    ingest_p.add_argument("--actors", nargs="*")
    ingest_p.add_argument("--ts", type=float)

    query_p = sub.add_parser("query", help="查询 Aurora")
    query_p.add_argument("query")
    query_p.add_argument("-k", type=int, default=8)
    query_p.add_argument("--session-id")

    respond_p = sub.add_parser("respond", help="生成一轮回复并异步持久化")
    respond_p.add_argument("user_message")
    respond_p.add_argument("--session-id")
    respond_p.add_argument("--context")
    respond_p.add_argument("--actors", nargs="*")
    respond_p.add_argument("-k", type=int, default=6)
    respond_p.add_argument("--ts", type=float)

    identity_p = sub.add_parser("identity", help="查看当前身份快照")
    identity_p.add_argument("--full", "-f", action="store_true")

    sub.add_parser("stats", help="查看内存与队列统计")

    event_p = sub.add_parser("event", help="查看事件投影状态")
    event_p.add_argument("event_id")

    job_p = sub.add_parser("job", help="查看任务状态")
    job_p.add_argument("job_id")

    serve_p = sub.add_parser("serve", help="启动 API 服务器")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")

    observe_p = sub.add_parser("observe", help="启动全屏终端对话模式")
    observe_p.add_argument("--session-id", default="terminal_observer")
    observe_p.add_argument("--max-hits", type=int, default=6)

    args = parser.parse_args(argv)

    if args.cmd is None:
        run_observer(data_dir=args.data_dir, session_id="terminal_observer", max_hits=6)
        return
    if args.cmd == "ingest":
        _cmd_ingest(args)
    elif args.cmd == "query":
        _cmd_query(args)
    elif args.cmd == "respond":
        _cmd_respond(args)
    elif args.cmd == "identity":
        _cmd_identity(args)
    elif args.cmd == "stats":
        _cmd_stats(args)
    elif args.cmd == "event":
        _cmd_event(args)
    elif args.cmd == "job":
        _cmd_job(args)
    elif args.cmd == "serve":
        _cmd_serve(args)
    elif args.cmd == "observe":
        _cmd_observe(args)


if __name__ == "__main__":
    main()
