"""
Model scoring commands: compute scores from a feature table guided by a YAML config.

Commands:
- `score`: load a feature table, compute score columns from the `score` section of the YAML config,
  and write the output feature table with scores appended.

Programmatic helpers:
- `compute_scores_from_df(df, score_registry, group_by='competencia')`
- `compute_scores_from_yaml(df, yaml_cfg)`
"""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="Compute model scores from feature tables.")
console = Console()


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-6)


def compute_scores_from_df(df: pd.DataFrame, score_registry: Dict[str, Any], group_by: str = "competencia") -> pd.DataFrame:
    """Compute scores defined in score_registry and append to input DataFrame copy.

    Parameters
    ----------
    df: pd.DataFrame
        Feature table containing columns referenced by score definitions.
    score_registry: dict
        Mapping of score_name -> config (type, args, adjustment)
    group_by: str
        Column name to use for cross-sectional normalization (default: 'competencia').

    Returns
    -------
    pd.DataFrame
        A copy of df with new score columns added.
    """
    out = df.copy()

    for score_name, cfg in score_registry.items():
        score_type = cfg.get("type")
        args = cfg.get("args", {}) or {}
        adjustment = cfg.get("adjustment", []) or []

        if score_type == "zscore":
            feat = args.get("feature")
            if feat not in out.columns:
                out[score_name] = np.nan
                continue

            # compute group-wise zscore
            out[score_name] = out.groupby(group_by)[feat].transform(_zscore)

            # adjustments
            if "invert" in adjustment:
                out[score_name] = -1 * out[score_name]

            # optionally coalesce NaNs to 0 if specified
            if "coalesce" in adjustment:
                out[score_name] = out[score_name].fillna(0)

        else:
            raise ValueError(f"Unsupported score type: {score_type}")

    return out


def compute_scores_from_yaml(df: pd.DataFrame, yaml_cfg: Dict[str, Any], group_by: str = "competencia") -> pd.DataFrame:
    """Apply score definitions from YAML (top-level `score` section) to DataFrame."""
    score_registry = yaml_cfg.get("score") or {}
    return compute_scores_from_df(df, score_registry, group_by=group_by)


@app.command()
def score(
    input_path: Path = typer.Option(Path("data/features.parquet"), "--input", "-i", help="Feature table input path (parquet or csv)."),
    config_path: Path = typer.Option(Path("feature.yaml"), "--config", "-c", help="YAML file with `score` section."),
    output_path: Path = typer.Option(Path("data/features_scored.parquet"), "--output", "-o", help="Output path for scored feature table."),
):
    """Load feature table, compute scores from YAML config, and write the output table."""
    import yaml

    # Load features
    if not input_path.exists():
        console.print(f"[red]Input feature table not found: {input_path}[/red]")
        raise typer.Exit(1)

    try:
        if input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        console.print(f"[red]Failed to read input file: {e}[/red]")
        raise typer.Exit(1)

    # Load yaml
    try:
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Failed to load config YAML: {e}[/red]")
        raise typer.Exit(1)

    try:
        out = compute_scores_from_yaml(df, yaml_cfg)
    except Exception as e:
        console.print(f"[red]Error computing scores: {e}[/red]")
        raise typer.Exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.to_parquet(output_path, index=False)
        console.print(f"[green]Scored features written to:[/green] {output_path}")
    except ImportError:
        csv_path = output_path.with_suffix(".csv")
        out.to_csv(csv_path, index=False)
        console.print(f"[yellow]Pyarrow missing; wrote CSV instead →[/yellow] {csv_path}")

    console.print(f"[green]Done.[/green] Rows: {len(out)}")
