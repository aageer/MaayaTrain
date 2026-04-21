"""MaayaTrain CLI — Typer-based command-line interface.

Usage::

    maayatrain init                          # Create default config
    maayatrain start --model gpt2-small      # Start as coordinator
    maayatrain join auto                     # Join via mDNS
    maayatrain join 192.168.1.10:7471        # Join by address
    maayatrain status                        # Show cluster status
    maayatrain relay export --checkpoint ./checkpoints/step-5000
    maayatrain relay import ./relay-checkpoint
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from . import __version__
from .hardware import detect_device

console = Console()

app = typer.Typer(
    name="maayatrain",
    help="Cross-platform distributed ML training using DiLoCo.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
relay_app = typer.Typer(help="Checkpoint relay commands for async handoffs.")
app.add_typer(relay_app, name="relay")


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


def _banner() -> None:
    console.print(
        Panel.fit(
            f"[bold cyan]⚡ MaayaTrain[/bold cyan] v{__version__}\n"
            "[dim]Cross-platform distributed ML training with DiLoCo[/dim]",
            border_style="blue",
        )
    )


# -----------------------------------------------------------------------
# Commands
# -----------------------------------------------------------------------


@app.command()
def init(
    path: Path = typer.Option(".", help="Directory to create config in"),
) -> None:
    """Create a default ``maayatrain.toml`` configuration file."""
    _banner()
    src = Path(__file__).parent.parent / "maayatrain.toml"
    dst = path / "maayatrain.toml"

    if dst.exists():
        console.print(f"[yellow]Config already exists: {dst}[/yellow]")
        return

    if src.exists():
        shutil.copy(src, dst)
    else:
        # Write defaults inline
        from .settings import MaayaTrainSettings
        import json

        settings = MaayaTrainSettings()
        dst.write_text(
            "# MaayaTrain configuration\n"
            "# See documentation for all options.\n\n"
            "[model]\n"
            f'name = "{settings.model.name}"\n\n'
            "[dataset]\n"
            f'path = "{settings.dataset.path}"\n'
            f"seq_length = {settings.dataset.seq_length}\n\n"
            "[training]\n"
            f"batch_size = {settings.training.batch_size}\n"
            f"max_steps = {settings.training.max_steps}\n"
            f'checkpoint_dir = "{settings.training.checkpoint_dir}"\n'
            f"checkpoint_every = {settings.training.checkpoint_every}\n"
            f"log_every = {settings.training.log_every}\n"
            f"seed = {settings.training.seed}\n\n"
            "[diloco]\n"
            f"inner_steps = {settings.diloco.inner_steps}\n"
            f"inner_lr = {settings.diloco.inner_lr}\n"
            f'inner_optimizer = "{settings.diloco.inner_optimizer}"\n'
            f"inner_weight_decay = {settings.diloco.inner_weight_decay}\n"
            f"outer_lr = {settings.diloco.outer_lr}\n"
            f"outer_momentum = {settings.diloco.outer_momentum}\n"
            f"nesterov = {'true' if settings.diloco.nesterov else 'false'}\n"
            f"gradient_compression = {'true' if settings.diloco.gradient_compression else 'false'}\n"
            f"compress_fp16 = {'true' if settings.diloco.compress_fp16 else 'false'}\n\n"
            "[network]\n"
            f"port = {settings.network.port}\n"
            f"heartbeat_interval = {settings.network.heartbeat_interval}\n\n"
            "[dashboard]\n"
            f"port = {settings.dashboard.port}\n"
            f"enabled = {'true' if settings.dashboard.enabled else 'false'}\n",
            encoding="utf-8",
        )

    console.print(f"[green]✓ Created {dst}[/green]")


@app.command()
def start(
    model: str = typer.Option("gpt2-small", "--model", "-m", help="Model architecture name"),
    dataset: str = typer.Option("./data/wikitext.txt", "--dataset", "-d", help="Training data path"),
    dashboard: bool = typer.Option(False, "--dashboard", help="Start the monitoring dashboard"),
    port: int = typer.Option(7471, "--port", "-p", help="TCP port for peer connections"),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from checkpoint directory"),
    max_steps: int = typer.Option(100_000, "--max-steps", help="Maximum training steps"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Start training as the coordinator node."""
    _setup_logging(verbose)
    _banner()

    # Detect hardware
    hw = detect_device()
    console.print(f"[cyan]Device:[/cyan] {hw.summary()}")

    # Load settings
    from .settings import load_settings

    settings = load_settings()
    settings.model.name = model
    settings.dataset.path = dataset
    settings.network.port = port
    settings.training.max_steps = max_steps
    settings.dashboard.enabled = dashboard

    # Check dataset
    if not Path(dataset).exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        console.print("[dim]Create a text file or download a dataset first.[/dim]")
        raise typer.Exit(1)

    # Build dataset
    from .training.loop import SimpleTextDataset

    ds = SimpleTextDataset(dataset, seq_length=settings.dataset.seq_length, device=str(hw.device))

    # Build model
    from .architectures.catalog import create_model

    mdl = create_model(model, vocab_size=ds.vocab_size, seq_length=settings.dataset.seq_length)
    mdl.to(hw.device)
    param_m = sum(p.numel() for p in mdl.parameters()) / 1e6
    console.print(f"[cyan]Model:[/cyan] {model} ({param_m:.1f}M parameters)")

    # Attempt torch.compile for hardware-optimized execution
    from .architectures.gpt2 import try_compile

    mdl = try_compile(mdl)

    # Resume from checkpoint
    if resume:
        from .training.snapshots import load_snapshot

        meta = load_snapshot(resume, mdl, device=str(hw.device))
        console.print(f"[green]Resumed from step {meta.global_step} (loss {meta.loss:.4f})[/green]")

    # Start orchestrator
    from .training.orchestrator import Orchestrator

    orch = Orchestrator(mdl, settings, hw, ds)

    # Optional dashboard
    if dashboard:
        from .monitor.server import create_dashboard_app, push_metrics
        import uvicorn
        import threading

        dash_app = create_dashboard_app()

        def _on_metrics(m):
            push_metrics(dash_app, m.step, m.loss, m.tokens_per_sec, m.lr)

        orch.on_metrics = _on_metrics

        # Run dashboard in background thread
        def _run_dash():
            uvicorn.run(dash_app, host="0.0.0.0", port=settings.dashboard.port, log_level="warning")

        dash_thread = threading.Thread(target=_run_dash, daemon=True)
        dash_thread.start()
        console.print(f"[green]Dashboard:[/green] http://localhost:{settings.dashboard.port}")

    # mDNS advertisement
    from .discovery.zeroconf_service import ZeroconfAdvertiser

    advertiser = ZeroconfAdvertiser(
        port=port,
        model=model,
        device=hw.device_name,
        memory_gb=hw.memory_gb,
        hostname=hw.hostname,
    )
    try:
        advertiser.start()
    except Exception as e:
        console.print(f"[yellow]mDNS failed (LAN discovery disabled): {e}[/yellow]")

    console.print(f"[green]Listening on port {port} — waiting for workers…[/green]")
    console.print("[dim]Press Ctrl+C to stop training.[/dim]\n")

    # Run the training loop
    try:
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — saving checkpoint…[/yellow]")
    finally:
        try:
            advertiser.stop()
        except Exception:
            pass
    console.print("[green]✓ Training complete.[/green]")


@app.command()
def join(
    target: str = typer.Argument(
        "auto",
        help="Address to join ('auto' for mDNS discovery, or 'host:port')",
    ),
    dataset: str = typer.Option("./data/wikitext.txt", "--dataset", "-d", help="Local training data"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Join a training session as a worker node."""
    _setup_logging(verbose)
    _banner()

    hw = detect_device()
    console.print(f"[cyan]Device:[/cyan] {hw.summary()}")

    from .settings import load_settings

    settings = load_settings()
    settings.dataset.path = dataset

    # Resolve target
    if target == "auto":
        console.print("[dim]Searching for coordinator via mDNS…[/dim]")
        from .discovery.zeroconf_service import ZeroconfBrowser

        browser = ZeroconfBrowser()
        browser.start()
        peer = browser.wait_for_coordinator(timeout=30)
        browser.stop()

        if not peer:
            console.print("[red]No coordinator found on LAN. Try specifying an address.[/red]")
            raise typer.Exit(1)

        host, port = peer.host, peer.port
        console.print(f"[green]Found coordinator: {peer.model} at {host}:{port}[/green]")
    else:
        parts = target.rsplit(":", 1)
        host = parts[0]
        port = int(parts[1]) if len(parts) > 1 else 7471

    # Check dataset
    if not Path(dataset).exists():
        console.print(f"[red]Dataset not found: {dataset}[/red]")
        raise typer.Exit(1)

    from .training.loop import SimpleTextDataset

    ds = SimpleTextDataset(dataset, seq_length=settings.dataset.seq_length, device=str(hw.device))

    from .architectures.catalog import create_model

    mdl = create_model(settings.model.name, vocab_size=ds.vocab_size, seq_length=settings.dataset.seq_length)
    mdl.to(hw.device)

    from .training.participant import Participant

    worker = Participant(mdl, settings, hw, ds)

    console.print(f"[green]Connecting to {host}:{port}…[/green]\n")

    try:
        asyncio.run(worker.connect_and_train(host, port))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — saving local checkpoint…[/yellow]")
    console.print("[green]✓ Training session ended.[/green]")


@app.command()
def status() -> None:
    """Show hardware info and training status."""
    _banner()
    hw = detect_device()

    table = Table(title="System Info", show_header=False, border_style="blue")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("OS", hw.os_name)
    table.add_row("Hostname", hw.hostname)
    table.add_row("Device", hw.device_name)
    table.add_row("Backend", hw.backend.upper())
    table.add_row("Memory", f"{hw.memory_gb:.1f} GB")
    table.add_row("Compute", f"~{hw.compute_tflops:.1f} TFLOPS (est.)")
    table.add_row("PyTorch", f"{__import__('torch').__version__}")
    console.print(table)

    # List available models
    from .architectures.catalog import list_models

    console.print(f"\n[cyan]Available models:[/cyan] {', '.join(list_models())}")


@relay_app.command("export")
def relay_export(
    checkpoint: str = typer.Option(..., help="Checkpoint directory to export"),
    output: str = typer.Option("./relay-checkpoint", help="Output directory"),
    description: str = typer.Option("", help="Description for the relay"),
) -> None:
    """Export a checkpoint for relay handoff."""
    _banner()
    from .training.snapshots import export_relay

    result = export_relay(checkpoint, output, description)
    console.print(f"[green]✓ Relay checkpoint exported: {result}[/green]")


@relay_app.command("import")
def relay_import(
    path: str = typer.Argument(..., help="Relay checkpoint directory"),
) -> None:
    """Import a relay checkpoint to resume training."""
    _banner()
    console.print(f"[green]✓ Relay checkpoint ready at: {path}[/green]")
    console.print("[dim]Use 'maayatrain start --resume {path}' to continue training.[/dim]")


@app.command()
def version() -> None:
    """Show version info."""
    console.print(f"MaayaTrain v{__version__}")


def main() -> None:
    """Entry point for the ``maayatrain`` CLI."""
    app()


if __name__ == "__main__":
    main()
