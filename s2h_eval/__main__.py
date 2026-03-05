from __future__ import annotations

from pathlib import Path

import typer

from .config import load_yaml, make_run_id
from .export_audit import export_audit
from .manifest import write_manifest
from .multiturn import run_multiturn
from .run_pipeline import run_inference
from .score_pipeline import run_scoring
from .stats_pipeline import run_stats

app = typer.Typer(add_completion=False)


@app.command()
def run(config: str = typer.Option("configs/main.yaml")) -> None:
    cfg = load_yaml(config)
    run_id = make_run_id(cfg)
    run_dir = run_inference(cfg, run_id)
    write_manifest(cfg, run_dir, run_id)
    typer.echo(f"run complete: {run_dir}")


@app.command()
def score(run: str, config: str = typer.Option("configs/main.yaml")) -> None:
    cfg = load_yaml(config)
    run_dir = Path(run)
    out = run_scoring(cfg, run_dir)
    typer.echo(f"scored -> {out}")


@app.command()
def stats(run: str, config: str = typer.Option("configs/main.yaml")) -> None:
    cfg = load_yaml(config)
    run_dir = Path(run)
    out = run_stats(cfg, run_dir)
    typer.echo(f"stats -> {out}")


@app.command()
def multiturn(config: str = typer.Option("configs/main.yaml"), run_id: str | None = None) -> None:
    cfg = load_yaml(config)
    rid = run_id or make_run_id(cfg)
    out = run_multiturn(cfg, rid)
    typer.echo(f"multiturn -> {out}")


@app.command("export-audit")
def export_audit_cmd(run: str, n: int = 50) -> None:
    out = export_audit(Path(run), n_per_model=n)
    typer.echo(f"audit export -> {out}")


if __name__ == "__main__":
    app()
