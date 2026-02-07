"""
Feature engineering commands: build a feature store from partitioned dataset files.

Commands:
- `build`: load dataset partitions, compute features, and write a feature store parquet.

Programmatic helpers are provided: `compute_features_from_df` and `compute_features`.
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True, help="Build feature store from datasets.")
console = Console()


import pandas as pd
import numpy as np
from typing import Dict, List, Any


def build_feature_engine(feature_engine: Dict, group_keys: List[str], registry: Any):
    """
    Returns a callable function that computes features based on:
      - FEATURE_ENGINE definitions
      - YAML feature registry
    """

    def compute_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes all features defined in the YAML registry using methods in FEATURE_ENGINE.

        Returns:
            DataFrame with all group_keys + computed feature columns.
        """

        results = []

        # process each feature in registry
        for feat_name, feat_cfg in registry.items():
            method = feat_cfg["method"]
            args = feat_cfg.get("args", [])
            adjustments = feat_cfg.get("adjustment", [])

            if method not in feature_engine:
                raise ValueError(f"Method '{method}' not found in FEATURE_ENGINE")

            method_type = feature_engine[method]["type"]

            if method_type == "aggregation":
                col = args[0]
                func = feature_engine[method]["function"]["pandas"]

                df_agg = (
                    df.groupby(group_keys)
                      .apply(lambda g: func(g[col]))
                      .reset_index(name=feat_name)
                )

            elif method_type == "row_operation":
                col = args[0]
                values = args[1]
                func = feature_engine[method]["function"]["pandas"]

                temp = df.copy()
                temp[feat_name] = func(temp[col], values)

                df_agg = (
                    temp.groupby(group_keys)[feat_name]
                        .sum()
                        .reset_index()
                )

            elif method_type == "custom":
                func = feature_engine[method]["function"]["pandas"]

                # allow passing args in YAML
                df_agg = func(df, group_keys, *args)
            else:
                raise ValueError(f"Unknown method type: {method_type}")

            for adj in adjustments:
                if adj not in feature_engine:
                    raise ValueError(f"Adjustment '{adj}' not found")

                adj_func = feature_engine[adj]["function"]["pandas"]
                df_agg[feat_name] = adj_func(df_agg[feat_name])

            results.append(df_agg)

        final = results[0]
        for partial in results[1:]:
            final = final.merge(partial, on=group_keys, how="left")

        return final

    return compute_features


# custom feature functions
def _credito_share_feature_fn(df, group_keys: List[str], credito_list: List[str]) -> pd.DataFrame:

    NUM_COLS = ["VL_MERC_POS_FINAL"]

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_credito"] = df.get("TP_APLIC", pd.Series()).isin(credito_list).astype(int)
    num_credito = (df.assign(num=df["VL_MERC_POS_FINAL"] * df["is_credito"]).groupby(group_keys)["num"].sum())
    denom = df.groupby(group_keys)["VL_MERC_POS_FINAL"].sum()
    credito_share = (num_credito / denom).reset_index(name="credito_share")
    return credito_share

def _related_party_share_feature_fn(df, group_keys: List[str]) -> pd.DataFrame:

    NUM_COLS = ["VL_MERC_POS_FINAL"]

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["is_related"] = (df.get("EMISSOR_LIGADO") == "S").astype(int)
    num_related = (df.assign(num=df["VL_MERC_POS_FINAL"] * df["is_related"]).groupby(group_keys)["num"].sum())
    denom = df.groupby(group_keys)["VL_MERC_POS_FINAL"].sum()
    related_share = (num_related / denom).reset_index(name="related_party_share")
    return related_share


def _hhi_feature_fn(df, group_keys: List[str]) -> pd.DataFrame:

    NUM_COLS = ["VL_MERC_POS_FINAL"]

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    issuer_hhi = (
        df.groupby(group_keys + ["CPF_CNPJ_EMISSOR"]) 
          .agg(pos=("VL_MERC_POS_FINAL", "sum"))
          .reset_index()
    )

    issuer_hhi["weight"] = issuer_hhi.groupby(group_keys)["pos"].transform(lambda x: x / x.sum())

    issuer_hhi_score = (
        issuer_hhi.groupby(group_keys)
          .apply(lambda x: np.sum(x["weight"] ** 2))
          .rename("issuer_hhi")
          .reset_index()
    )
    return issuer_hhi_score


FEATURE_ENGINE = {
    "max": {
        "type": "aggregation",
        "description": "Returns the maximum value",
        "function": {
            "pandas": lambda s: pd.to_numeric(s, errors="coerce").max(),
            "sql": "MAX({col})",
            "spark": "max({col})",
        }
    },
    "sum": {
        "type": "aggregation",
        "description": "Returns the sum of values",
        "function": {
            "pandas": lambda s: pd.to_numeric(s, errors="coerce").sum(),
            "sql": "SUM({col})",
            "spark": "sum({col})",
        }
    },
    "nunique": {
        "type": "aggregation",
        "description": "Counts distinct values",
        "function": {
            "pandas": lambda s: s.nunique(),
            "sql": "COUNT(DISTINCT {col})",
            "spark": "countDistinct({col})",
        }
    },
    "isin": {
        "type": "row_operation",
        "description": "Checks membership of each row value in a provided list",
        "function": {
            "pandas": lambda s, values: s.isin(values).astype(int),
            "sql": "CASE WHEN {col} IN ({values}) THEN 1 ELSE 0 END",
            "spark": "when(col({col}).isin({values}), 1).otherwise(0)",
        }
    },
    "credito_share_feature_fn": {
        "type": "custom",
        "description": "Weighted share of credit-linked assets in the portfolio.",
        "function": {
            "pandas": _credito_share_feature_fn,
        }
    },
    "related_party_share_feature_fn": {
        "type": "custom",
        "description": "Weighted share of related-party issuers.",
        "function": {
            "pandas": _related_party_share_feature_fn,
        }
    },
    "hhi_feature_fn": {
        "type": "custom",
        "description": "Computes Herfindahl-Hirschman Index over weighted issuer exposures",
        "function": {
            "pandas": _hhi_feature_fn,
        }
    },
    "clip": {
        "type": "adjustment",
        "description": "Clip percentage value between 0 and 1.",
        "function": {
            "pandas": lambda s: s.clip(0, 1),
        }
    },
    "log": {
        "type": "adjustment",
        "description": "Log value.",
        "function": {
            "pandas": lambda s: np.log1p(pd.to_numeric(s, errors="coerce")),
        }
    },
    "coalesce": {
        "type": "adjustment",
        "description": "Coalesce zero.",
        "function": {
            "pandas": lambda s: s.fillna(0),
        }
    },
}


def compute_all_features(
    datasets: Dict[str, pd.DataFrame],
    yaml_cfg: Dict[str, Any],
    feature_engine: Dict[str, Any],
) -> pd.DataFrame:
    """
    Computes features defined in YAML across multiple datasets and merges the results.

    Parameters
    ----------
    datasets : Dict[str, DataFrame]
        Dictionary of dataset_name -> pandas DataFrame.
    yaml_cfg : Dict
        Parsed YAML under "feature".
    feature_engine : Dict
        Registry of methods including custom, aggregation, and adjustment functions.

    Returns
    -------
    pd.DataFrame
        Final merged feature dataframe.
    """

    feature_section = yaml_cfg["feature"]
    group_keys: List[str] = feature_section["group_keys"]
    registry: Dict[str, Any] = feature_section["feature_registry"]

    results = []

    for dataset_name in registry:

        dataset_feature_cfg = registry[dataset_name]
        # Skip datasets that have no feature definitions (empty mapping)
        if not dataset_feature_cfg:
            console.print(f"[yellow]Skipping dataset '{dataset_name}': no features defined in registry.[/yellow]")
            continue

        df = datasets[dataset_name]

        df_res = compute_features_from_df(
            df=df,
            group_keys=group_keys,
            registry=dataset_feature_cfg,
            feature_engine=feature_engine,
        )

        results.append(df_res)

    # Merge everything on the group keys
    if not results:
        raise ValueError("No features computed. Check your YAML registry.")

    final_df = results[0]

    for r in results[1:]:
        final_df = final_df.merge(r, on=group_keys, how="outer")

    return final_df


def compute_features_from_df(df: pd.DataFrame, group_keys: List[str], registry: Any, feature_engine) -> pd.DataFrame:

    df = df.copy()

    # Build the engine
    compute_features = build_feature_engine(feature_engine, group_keys, registry)

    # Compute all features
    return compute_features(df)


def _load_partitioned_dataset(input_dir: Path, dataset: str) -> pd.DataFrame:
    """Load all partition `data.parquet` files for a dataset under input_dir/dataset.

    Accepts the layout: input_dir / dataset / period=<period> / data.parquet
    """
    base = input_dir / dataset
    if not base.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base}")

    # Accept parquet or csv partition files
    parquet_files: List[Path] = list(base.rglob("data.parquet"))
    csv_files: List[Path] = list(base.rglob("data.csv"))
    files = parquet_files + csv_files

    if not files:
        raise FileNotFoundError(f"No partition files found under {base}")

    dfs = []
    for f in files:
        if f.suffix == ".parquet":
            dfs.append(pd.read_parquet(f))
        else:
            dfs.append(pd.read_csv(f))

    return pd.concat(dfs, ignore_index=True)


def compute_features_from_yaml(input_dir: Path, yaml_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load datasets based on YAML feature registry, compute all features, and return merged DataFrame.
    """

    feature_section = yaml_cfg["feature"]
    registry = feature_section["feature_registry"]

    # Load all datasets declared in YAML
    datasets = {}

    for dataset_name in registry.keys():
        df = _load_partitioned_dataset(input_dir, dataset_name)
        datasets[dataset_name] = df

    # Compute all features using unified engine
    final_features = compute_all_features(
        datasets=datasets,
        yaml_cfg=yaml_cfg,
        feature_engine=FEATURE_ENGINE,
    )

    return final_features


@app.command()
def build(
    input_dir: Path = typer.Option(Path("data"), "--input-dir", "-i", help="Directory containing datasets"),
    config_path: Path = typer.Option(Path("manifest.yaml"), "--config", "-c", help="YAML feature configuration"),
    output_path: Path = typer.Option(Path("data/features.parquet"), "--output", "-o", help="Output path to feature store"),
):
    """
    Compute all features specified in the YAML feature config and write to output_path.
    """
    import yaml

    try:
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Failed to load config YAML: {e}[/red]")
        raise typer.Exit(1)

    try:
        features = compute_features_from_yaml(input_dir, yaml_cfg)
    except Exception as e:
        console.print(f"[red]Error computing features:[/red] {e}")
        raise typer.Exit(1)

    # ---- Write Output ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        features.to_parquet(output_path, index=False)
        console.print(f"[green]Features written to:[/green] {output_path}")
    except ImportError:
        csv_path = output_path.with_suffix(".csv")
        features.to_csv(csv_path, index=False)
        console.print(f"[yellow]Pyarrow missing; wrote CSV instead →[/yellow] {csv_path}")

    console.print(f"[green]Done.[/green] Rows: {len(features)}")

