from __future__ import annotations

import argparse
import os
import sys
import time
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.status import Status

from aurora.memory import AuroraMemory

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aurora")
    subparsers = parser.add_subparsers(dest="command")

    add = subparsers.add_parser("add", help="Add a memory.")
    add.add_argument("text", nargs="?", help="Memory text to add.")
    add.add_argument("--interactive", "-i", action="store_true", help="Interactive mode.")

    search = subparsers.add_parser("search", help="Search memories.")
    search.add_argument("query", help="Search query.")
    search.add_argument("--limit", "-n", type=int, default=10, help="Number of results.")

    list_mem = subparsers.add_parser("list", help="List all memories.")
    list_mem.add_argument("--limit", "-n", type=int, default=50, help="Number of memories to show.")

    chat = subparsers.add_parser("chat", help="Start an interactive chat session.")
    chat.add_argument("user_text", nargs="?", help="Optional initial message.")

    subparsers.add_parser("health", help="Check system health.")

    return parser


def print_header():
    header_text = Text()
    header_text.append("A U R O R A\n", style="bold cyan")
    header_text.append("v2.0 [Memory Core]\n", style="dim italic white")
    header_text.append("Status: Online\n", style="green")

    panel = Panel(header_text, border_style="cyan", padding=(1, 2))
    console.print(panel)


def cmd_add(memory: AuroraMemory, text: str | None = None, interactive: bool = False) -> None:
    if interactive or not text:
        console.print("[dim]Enter memory text (Ctrl+C to cancel):[/dim]")
        text = Prompt.ask("[bold cyan]Memory[/bold cyan]")

    if not text:
        console.print("[red]No memory text provided.[/red]")
        return

    with Status("[dim]Storing in memory graph...[/dim]", spinner="dots", console=console):
        result = memory.add(text)

    console.print(f"[green]Memory added:[/green] {result['memory_id']}")


def cmd_search(memory: AuroraMemory, query: str, limit: int = 10) -> None:
    with Status("[dim]Searching memory graph...[/dim]", spinner="dots", console=console):
        results = memory.search(query, limit=limit)

    if not results:
        console.print("[dim]No memories found.[/dim]")
        return

    for i, r in enumerate(results, 1):
        panel = Panel(
            r["text"],
            title=f"[{i}] {r['memory_id'][:8]}... (score: {r['score']:.3f})",
            border_style="cyan",
        )
        console.print(panel)


def cmd_list(memory: AuroraMemory, limit: int = 50) -> None:
    memories = memory.get_all()

    if not memories:
        console.print("[dim]No memories stored.[/dim]")
        return

    for m in memories[:limit]:
        source_tag = f"[{m['source'][:3]}]" if m.get("source") else ""
        console.print(f"{source_tag} {m['memory_id'][:8]}... {m['text'][:60]}...")


def interactive_chat(memory: AuroraMemory, initial_text: str | None = None) -> None:
    console.clear()
    print_header()

    console.print(
        "[dim]Type [bold white]/search <query>[/bold white] to recall memories, [bold white]/quit[/bold white] to exit.[/dim]\n"
    )

    context = []

    if initial_text:
        _handle_turn(memory, initial_text, context)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            if not user_input.strip():
                continue

            if user_input.strip() == "/quit":
                console.print("\n[dim italic red]Exiting...[/dim italic red]")
                time.sleep(0.5)
                break

            if user_input.strip().startswith("/search "):
                query = user_input.strip()[8:]
                results = memory.search(query, limit=5)
                if results:
                    console.print("\n[bold]Recalled memories:[/bold]")
                    for r in results:
                        console.print(f"  • {r['text'][:80]}...")
                else:
                    console.print("[dim]No relevant memories found.[/dim]")
                continue

            if user_input.strip() == "/health":
                h = memory.health()
                console.print(
                    Panel(str(h), title="[System Health]", border_style="yellow", width=60)
                )
                continue

            _handle_turn(memory, user_input, context)

        except KeyboardInterrupt:
            console.print("\n[dim italic red]Connection closed.[/dim italic red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error:[/bold red] {e}")


def _handle_turn(memory: AuroraMemory, text: str, context: list[dict]) -> None:
    search_results = memory.search(text, limit=3)

    if search_results:
        console.print(f"[dim]Recalled {len(search_results)} memory(ies)[/dim]")

    with Status("[dim]Processing...[/dim]", spinner="dots", console=console):
        memory.add(text)

    console.print(
        "\n[bold magenta]Aurora[/bold magenta] [dim]>[/dim] [italic]Memory stored.[/italic]\n"
    )


def main() -> None:
    load_dotenv()

    parser = _build_parser()
    args = parser.parse_args()

    command = args.command or "chat"

    if (
        command in ("add", "search", "list", "chat", "health")
        and "AURORA_PROVIDER_API_KEY" not in os.environ
        and "AURORA_BAILIAN_LLM_API_KEY" not in os.environ
    ):
        console.print(
            "[dim]Note: No LLM API key found, but memory functions work without it.[/dim]"
        )

    try:
        with Status("[dim]Loading memory graph...[/dim]", spinner="bouncingBar", console=console):
            memory = AuroraMemory()
    except Exception as exc:
        console.print(f"[bold red]Failed to initialize Aurora Memory:[/bold red] {exc}")
        sys.exit(1)

    if command == "add":
        cmd_add(memory, getattr(args, "text", None), getattr(args, "interactive", False))
        return

    if command == "search":
        cmd_search(memory, args.query, args.limit)
        return

    if command == "list":
        cmd_list(memory, args.limit)
        return

    if command == "chat":
        user_text = getattr(args, "user_text", None)
        interactive_chat(memory, initial_text=user_text)
        return

    if command == "health":
        health = memory.health()
        console.print(health)
        return

    raise SystemExit(2)
