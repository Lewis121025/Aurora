from __future__ import annotations

import argparse
import sys
import os
import time
import readline
from dotenv import load_dotenv

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.status import Status
from rich.live import Live
from rich.align import Align
from rich.layout import Layout

from aurora.host_runtime.runtime import AuroraRuntime

console = Console()

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aurora")
    subparsers = parser.add_subparsers(dest="command")

    chat = subparsers.add_parser("chat", help="Start an interactive chat session.")
    chat.add_argument("user_text", nargs="?", help="Optional initial message.")
    chat.add_argument("--language", default="auto")

    subparsers.add_parser("health", help="Check system health.")
    subparsers.add_parser("integrity", help="Check system integrity.")
    return parser


def print_header():
    header_text = Text()
    header_text.append("A U R O R A\n", style="bold cyan")
    header_text.append("v10.0 [Holistic Mind Architecture]\n", style="dim italic white")
    header_text.append("Substrate: Online | Synapses: Active\n", style="green")
    
    panel = Panel(
        Align.center(header_text),
        border_style="cyan",
        padding=(1, 2)
    )
    console.print(panel)


def interactive_chat(runtime: AuroraRuntime, initial_text: str | None = None, language: str = "auto") -> None:
    console.clear()
    print_header()
    
    console.print("[dim]Type [bold white]/health[/bold white] for vitals, [bold white]/quit[/bold white] to sever connection.[/dim]\n")

    if initial_text:
        _handle_turn(runtime, initial_text, language)

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            if not user_input.strip():
                continue

            if user_input.strip() == "/quit":
                console.print("\n[dim italic red]Severing neural link...[/dim italic red]")
                time.sleep(0.5)
                break

            if user_input.strip() == "/health":
                h = runtime.health()
                status_color = "green" if h.substrate_alive else "red"
                health_text = (
                    f"Substrate Alive: [{status_color}]{h.substrate_alive}[/{status_color}]\n"
                    f"Anchor Nodes: [bold yellow]{h.anchor_count}[/bold yellow]\n"
                    f"Next Wake: [dim]{h.next_wake_at}[/dim]"
                )
                console.print(Panel(health_text, title="[System Diagnostics]", border_style="yellow", width=60))
                continue

            _handle_turn(runtime, user_input, language)

        except KeyboardInterrupt:
            console.print("\n[dim italic red]Connection forcibly closed.[/dim italic red]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Critical Error:[/bold red] {e}")


def _handle_turn(runtime: AuroraRuntime, text: str, language: str) -> None:
    with Status("[dim cyan]Processing thermodynamic resonance...[/dim cyan]", spinner="dots", console=console):
        outcome = runtime.handle_input(text, language=language)
    
    if outcome.outcome == "silence":
        console.print("\n[italic dim white]... Aurora remains silent ...[/italic dim white]\n")
    else:
        # Calculate a slight delay to simulate typing based on length
        reply = outcome.output_text or ""
        console.print(f"\n[bold magenta]Aurora[/bold magenta] [dim]>[/dim] {reply}\n")


def main() -> None:
    load_dotenv()
    
    if "AURORA_PROVIDER_API_KEY" not in os.environ and "AURORA_BAILIAN_LLM_API_KEY" not in os.environ:
        console.print("[bold red]CRITICAL:[/bold red] No API Key found in .env file or environment.")
        console.print("Create a [bold].env[/bold] file with your credentials to boot the substrate.")
        sys.exit(1)

    parser = _build_parser()
    args = parser.parse_args()
    
    command = args.command or "chat"

    try:
        with Status("[dim]Booting latent matrices and warming up encoders...[/dim]", spinner="bouncingBar", console=console):
            runtime = AuroraRuntime()
    except Exception as exc:
        console.print(f"[bold red]Failed to boot Aurora Substrate:[/bold red] {exc}")
        sys.exit(1)

    if command == "chat":
        user_text = getattr(args, "user_text", None)
        lang = getattr(args, "language", "auto")
        interactive_chat(runtime, initial_text=user_text, language=lang)
        return

    if command == "health":
        health = runtime.health()
        console.print(health)
        return

    if command == "integrity":
        report = runtime.integrity()
        console.print(report)
        return

    raise SystemExit(2)
