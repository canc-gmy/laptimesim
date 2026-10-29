"""Command Line Interface for laptimesim using Typer and Rich."""

from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from laptimesim import __version__
from laptimesim.core.config import FullSimulationConfig
from laptimesim.core.logging import setup_logging
from laptimesim.core.registry import model_registry
import laptimesim.track  # Ensure track importers are registered

app = typer.Typer(help="laptimesim: High-fidelity Lap Time Simulator CLI")
console = Console()


@app.command()
def version():
    """Print the version of laptimesim."""
    console.print(f"[bold green]laptimesim[/bold green] version [bold cyan]{__version__}[/bold cyan]")


@app.command()
def list_models():
    """List all registered vehicle subsystem models and track importers."""
    table = Table(title="Registered Subsystem Models")
    table.add_column("Category", style="cyan", no_wrap=True)
    table.add_column("Available Models", style="magenta")

    all_models = model_registry.list_all()
    if not all_models:
        console.print("[yellow]No models registered yet.[/yellow]")
        return

    for category, models in all_models.items():
        table.add_row(category, ", ".join(models) if models else "(none)")

    console.print(table)


@app.command()
def validate_config(
    config_file: Path = typer.Argument(..., help="Path to simulation TOML or JSON config file")
):
    """Validate a simulation configuration file against the Pydantic schema."""
    try:
        if config_file.suffix.lower() == ".json":
            cfg = FullSimulationConfig.from_json(config_file)
        else:
            cfg = FullSimulationConfig.from_toml(config_file)
        console.print(f"[bold green]Configuration valid![/bold green] Simulation: [cyan]{cfg.simulation.name}[/cyan]")
    except Exception as e:
        console.print(f"[bold red]Configuration error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def run(
    config_file: Path = typer.Argument(..., help="Path to simulation TOML config file")
):
    """Run a lap time simulation with the provided configuration file."""
    try:
        cfg = FullSimulationConfig.from_toml(config_file)
        from laptimesim.core.simulator import LaptimeSimulator

        sim = LaptimeSimulator(cfg)
        console.print(f"[bold green]Starting Simulation:[/bold green] {cfg.simulation.name}")
        output = sim.run()

        console.print(f"[bold cyan]Simulation Completed![/bold cyan]")
        console.print(f"Predicted Lap Time: [bold yellow]{output.result.lap_time:.3f} s[/bold yellow]")
        console.print(f"Solver Status: {output.result.message}")
        if output.exported_data_file:
            console.print(f"Exported Telemetry: {output.exported_data_file}")
        if output.gg_diagram_file:
            console.print(f"Saved GG Diagram: {output.gg_diagram_file}")
        if output.racing_line_plot_file:
            console.print(f"Saved Racing Line Plot: {output.racing_line_plot_file}")

    except Exception as e:
        console.print(f"[bold red]Simulation error:[/bold red] {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
