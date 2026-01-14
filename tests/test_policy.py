import pandas as pd
from pathlib import Path
import yaml

from fif_recsys.commands import policy


def _make_scored_df():
    return pd.DataFrame([
        {"CNPJ_FUNDO_CLASSE": "F1", "DENOM_SOCIAL": "Fund A", "competencia": "202501", "size_score": 1.0, "diversification_score": 0.5},
        {"CNPJ_FUNDO_CLASSE": "F2", "DENOM_SOCIAL": "Fund B", "competencia": "202501", "size_score": 2.0, "diversification_score": 0.2},
        {"CNPJ_FUNDO_CLASSE": "F3", "DENOM_SOCIAL": "Fund C", "competencia": "202502", "size_score": 1.5, "diversification_score": 0.8},
    ])


def test_compute_profile_scores_from_df():
    df = _make_scored_df()

    profiles = {
        "conservative": {"size_score": 0.25, "diversification_score": 0.75},
        "balanced": {"size_score": 0.5, "diversification_score": 0.5},
    }

    out = policy.compute_profile_scores_from_df(df, profiles)

    # Ensure score and rank columns are added
    assert "score_conservative" in out.columns
    assert "rank_conservative" in out.columns
    assert "score_balanced" in out.columns
    assert "rank_balanced" in out.columns

    # For competencia 202501, F2 has larger size_score but smaller diversification; check ranking exists
    sub = out[out["competencia"] == "202501"]
    assert sub["rank_conservative"].min() >= 1


def test_compute_profile_scores_from_yaml_accepts_profile_and_profiles():
    df = _make_scored_df()

    # Using singular 'profile' key
    cfg1 = {"profile": {"conservative": {"size_score": 0.25, "diversification_score": 0.75}}}
    out1 = policy.compute_profile_scores_from_yaml(df, cfg1)
    assert "score_conservative" in out1.columns

    # Using plural 'profiles' key
    cfg2 = {"profiles": {"conservative": {"size_score": 0.25, "diversification_score": 0.75}}}
    out2 = policy.compute_profile_scores_from_yaml(df, cfg2)
    assert "score_conservative" in out2.columns


def test_profile_score_cli_writes_output(tmp_path):
    df = _make_scored_df()
    input_path = tmp_path / "features.csv"
    df.to_csv(input_path, index=False)

    yaml_cfg = {
        "profile": {
            "conservative": {
                "size_score": 0.25,
                "diversification_score": 0.75,
            }
        }
    }

    cfg_path = tmp_path / "profiles.yaml"
    cfg_path.write_text(yaml.dump(yaml_cfg))

    out_path = tmp_path / "features_profile_scored.parquet"

    # Call the CLI command function directly (bypasses Typer runner for simplicity)
    policy.profile_score(input_path=input_path, config_path=cfg_path, output_path=out_path)

    # Output file may be parquet or CSV fallback; check either
    assert out_path.exists() or out_path.with_suffix('.csv').exists()

    # Load and verify columns
    if out_path.exists():
        res = pd.read_parquet(out_path)
    else:
        res = pd.read_csv(out_path.with_suffix('.csv'))

    assert "score_conservative" in res.columns
    assert "rank_conservative" in res.columns
