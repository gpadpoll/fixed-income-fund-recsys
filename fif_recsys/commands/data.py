"""
Commands to download and store datasets used by the project.

Provides a `download` command that accepts one or more URLs and saves the
retrieved content into the specified output directory (default: `etc/`).
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

import io
import zipfile
import requests
import pandas as pd
import typer
import yaml
from datetime import date
from rich.console import Console


app = typer.Typer(no_args_is_help=True, help="Manage datasets: download and store files.")

console = Console()


def download_zip(url: str) -> zipfile.ZipFile:
    console.print(f"[cyan]Downloading[/cyan] {url}")

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    return zipfile.ZipFile(io.BytesIO(r.content))


def load_zip(z: zipfile.ZipFile, period: str) -> pd.DataFrame:
    """Load CSV data inside the ZIP."""
    for member in z.namelist():
        if member.lower().endswith(".csv"):
            console.print(f"  Parsing [green]{member}[/green]")

            with z.open(member) as f:
                df = pd.read_csv(
                    f,
                    sep=";",
                    encoding="latin1",
                    dtype=str,
                    engine="python",
                    on_bad_lines="warn"
                )

            df["period"] = period
            return df

    raise ValueError("No CSV file found inside ZIP.")


def fetch_manifest(manifest: dict, output_dir: Path, reference_date: Optional[str] = None) -> Dict[str, "pd.DataFrame"]:
    """Programmatically fetch and store datasets from a manifest dict.

    This helper mirrors the CLI fetch behavior but accepts a manifest dict directly.
    It adds a `reference_date` column (ISO date string) and writes partitioned
    parquet files into `output_dir / dataset / period={period} / data.parquet`.

    Returns:
        Dict[str, pd.DataFrame]: Mapping of dataset name to the combined DataFrame that was written.
    """
    import pandas as pd

    if reference_date is None:
        reference_date = date.today().isoformat()

    output_dir.mkdir(parents=True, exist_ok=True)

    fetched: Dict[str, pd.DataFrame] = {}

    for ds_name, cfg in manifest.items():
        all_dfs = []

        for period in cfg.get("periods", []):
            url = cfg["base_url"] + cfg["filename_template"].format(period=period)

            try:
                z = download_zip(url)
                df = load_zip(z, period)
                all_dfs.append(df)
            except Exception as e:
                console.print(f"[yellow]Skipped {period}[/yellow]: {e}")

        if not all_dfs:
            console.print(f"[red]No data retrieved for {ds_name}.[/red]")
            continue

        df_final = pd.concat(all_dfs, ignore_index=True)
        # add reference_date
        df_final["reference_date"] = reference_date

        # store final df in return dict
        fetched[ds_name] = df_final

        for period in sorted(df_final["period"].unique()):
            df_part = df_final[df_final["period"] == period].copy()
            part_dir = output_dir / ds_name / f"period={period}"
            part_dir.mkdir(parents=True, exist_ok=True)
            part_path = part_dir / "data.parquet"
            try:
                df_part.to_parquet(part_path, index=False)
            except ImportError:
                csv_path = part_dir / "data.csv"
                df_part.to_csv(csv_path, index=False)
                console.print(f"[yellow]Pyarrow not available; wrote CSV instead:[/yellow] {csv_path}")

        console.print(f"[green]Saved[/green] {ds_name} → {output_dir / ds_name}")

    return fetched


@app.command()
def fetch(
    manifest_path: Path = typer.Option(Path("manifest.yaml"), "--manifest", "-m", help="Path to manifest YAML"),
    output_dir: Path = typer.Option(Path("data"), "--output-dir", "-d"),
):
    """
    Download and parse datasets based on the manifest file.
    Stores parsed DataFrames under OUTPUT_DIR.
    """

    # Load YAML manifest
    if not manifest_path.exists():
        console.print(f"[red]Manifest not found:[/red] {manifest_path}")
        raise typer.Exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    # pass a reference_date (today) to the programmatic helper
    fetch_manifest(manifest, output_dir, reference_date=date.today().isoformat())

