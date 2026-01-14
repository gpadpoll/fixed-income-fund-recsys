"""
Ranking Policy commands: compute final score for each customer profile segment guided by a YAML config.


Commands:
- `rank`: load a score table, compute final score for different segments from the `policy` section of the YAML config,
 and write the output feature table with rankings appended.


Programmatic helpers:
- `compute_rankings_from_df(df, segment_registry)`
- `compute_rankings_from_yaml(df, yaml_cfg)`
"""
from pathlib import Path
from typing import Any, Dict, Optional


import numpy as np
import pandas as pd
import typer
from rich.console import Console


app = typer.Typer(no_args_is_help=True, help="Compute rankings from score tables.")
console = Console()




def compute_profile_scores_from_df(df: pd.DataFrame, profiles: Dict[str, Dict[str, float]]) -> pd.DataFrame:
   """
   Compute weighted scores and rankings for each customer profile.


   Parameters
   ----------
   df : pd.DataFrame
       DataFrame containing fund metrics.
   profiles : dict
       Mapping of profile -> scoring weights.


   Returns
   -------
   pd.DataFrame
       Updated DataFrame with score and ranking columns per profile.
   """
   result = df.copy()


   for profile_name, weights in profiles.items():
       score_col = f"score_{profile_name}"
       rank_col = f"rank_{profile_name}"


       # Compute weighted score
       result[score_col] = sum(
           result[metric] * weight
           for metric, weight in weights.items()
           if metric in result.columns
       )


       # Compute ranking (descending: best score → rank 1).
       # If scores are NaN, set rank to 0 to avoid conversion errors.
       result[rank_col] = (
           result[score_col]
           .rank(method="dense", ascending=False)
           .fillna(0)
           .astype(int)
       )


   return result




def compute_profile_scores_from_yaml(df: pd.DataFrame, yaml_cfg: Dict[str, Any]) -> pd.DataFrame:
   """Apply profile definitions from YAML (top-level `profile` or `profiles` section) to DataFrame.

   This helper accepts either:
   - a full YAML config dict containing `profile` or `profiles` keys, or
   - a profiles mapping already extracted from the YAML (mapping profile->weights).
   """
   # If the caller passed the full YAML (with `profile`/`profiles`) extract it; otherwise assume
   # `yaml_cfg` is already the profiles mapping.
   if isinstance(yaml_cfg, dict) and (
       ("profile" in yaml_cfg) or ("profiles" in yaml_cfg)
   ):
       profile_registry = yaml_cfg.get("profile") or yaml_cfg.get("profiles") or {}
   else:
       # yaml_cfg is expected to be the mapping {profile_name: weights}
       profile_registry = yaml_cfg or {}

   return compute_profile_scores_from_df(df, profile_registry)




@app.command("profile-score")
def profile_score(
   input_path: Path = typer.Option(Path("data/features_scored.parquet"),
                                   "--input", "-i", help="Feature table with score components."),
   config_path: Path = typer.Option(Path("profiles.yaml"),
                                    "--config", "-c", help="YAML file defining customer profile weights."),
   output_path: Path = typer.Option(Path("data/features_profile_scored.parquet"),
                                    "--output", "-o", help="Output file with profile scores and rankings."),
):
   """
   Compute weighted profile scores and rankings according to YAML profile definitions.


   The YAML must define:
   profiles:
     conservative:
       size_score: 0.25
       diversification_score: 0.20
       ...
     balanced:
       size_score: 0.20
       ...
   """
   import yaml


   # Load feature table
   if not input_path.exists():
       console.print(f"[red]Input file not found: {input_path}[/red]")
       raise typer.Exit(1)


   try:
       df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
   except Exception as e:
       console.print(f"[red]Failed to read input file: {e}[/red]")
       raise typer.Exit(1)


   # Load profiles YAML
   try:
       with open(config_path, "r") as f:
           cfg = yaml.safe_load(f)
           # Support either top-level 'profiles' or 'profile' keys to match examples/notebooks
           profiles = cfg.get("profiles") or cfg.get("profile") or {}
   except Exception as e:
       console.print(f"[red]Failed to load YAML config: {e}[/red]")
       raise typer.Exit(1)


   if not profiles:
       console.print("[red]YAML missing `profile`/`profiles` section.[/red]")
       raise typer.Exit(1)


   # Compute scores + rankings
   try:
       out = compute_profile_scores_from_yaml(df, profiles)
   except Exception as e:
       console.print(f"[red]Error computing profile scores: {e}[/red]")
       raise typer.Exit(1)


   # Save output
   output_path.parent.mkdir(parents=True, exist_ok=True)
   try:
       out.to_parquet(output_path, index=False)
       console.print(f"[green]Profile scores written to:[/green] {output_path}")
   except ImportError:
       csv_path = output_path.with_suffix(".csv")
       out.to_csv(csv_path, index=False)
       console.print(f"[yellow]Pyarrow missing, wrote CSV instead →[/yellow] {csv_path}")


   console.print(f"[green]Done.[/green] Rows: {len(out)}")

