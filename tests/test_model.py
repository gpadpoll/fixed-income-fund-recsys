import pandas as pd
import numpy as np
from pathlib import Path
import yaml

from fif_recsys.commands import model


def _make_features_df():
    return pd.DataFrame([
        {"CNPJ_FUNDO_CLASSE": "F1", "DENOM_SOCIAL": "Fund A", "competencia": "202501", "log_aum": 10.0, "n_ativos": 5, "n_emissores": 2, "credito_share": 0.8, "related_party_share": 0.0, "issuer_hhi": 0.9},
        {"CNPJ_FUNDO_CLASSE": "F2", "DENOM_SOCIAL": "Fund B", "competencia": "202501", "log_aum": 8.0, "n_ativos": 3, "n_emissores": 1, "credito_share": 0.1, "related_party_share": 0.2, "issuer_hhi": 0.4},
        {"CNPJ_FUNDO_CLASSE": "F3", "DENOM_SOCIAL": "Fund C", "competencia": "202502", "log_aum": 12.0, "n_ativos": 10, "n_emissores": 5, "credito_share": 0.2, "related_party_share": 0.0, "issuer_hhi": 0.2},
    ])


def test_zscore_and_invert():
    df = _make_features_df()
    score_cfg = {
        "size_score": {"type": "zscore", "args": {"feature": "log_aum"}},
        "credit_risk_score": {"type": "zscore", "args": {"feature": "credito_share"}, "adjustment": ["invert"]},
    }

    out = model.compute_scores_from_df(df, score_cfg, group_by="competencia")

    # size_score for competencia 202501: F1 (10) vs F2 (8) -> zscores
    s = out[out["competencia"] == "202501"]["size_score"]
    assert np.isclose(s.mean(), 0.0, atol=1e-6)

    # credit_risk_score should be inverted (higher credito_share -> lower score)
    cr = out[out["competencia"] == "202501"]["credit_risk_score"]
    # F1 had high credito_share and should have lower score than F2
    vals = dict(zip(out["CNPJ_FUNDO_CLASSE"], out["credit_risk_score"]))
    assert vals["F1"] < vals["F2"]


def test_compute_scores_from_yaml_and_cli(tmp_path, monkeypatch, capsys):
    df = _make_features_df()
    input_path = tmp_path / "features.csv"
    df.to_csv(input_path, index=False)

    yaml_cfg = {
        "score": {
            "size_score": {"type": "zscore", "args": {"feature": "log_aum"}},
            "credit_risk_score": {"type": "zscore", "args": {"feature": "credito_share"}, "adjustment": ["invert"]}
        }
    }

    cfg_path = tmp_path / "score.yaml"
    cfg_path.write_text(yaml.dump(yaml_cfg))

    out_path = tmp_path / "features_scored.parquet"

    # Run the CLI command (call the command function directly)
    model.score(input_path=input_path, config_path=cfg_path, output_path=out_path)

    # Output file may be parquet or CSV fallback; check either
    assert out_path.exists() or out_path.with_suffix('.csv').exists()

    # Load the result and check columns
    if out_path.exists():
        res_df = pd.read_parquet(out_path)
    else:
        res_df = pd.read_csv(out_path.with_suffix('.csv'))

    assert "size_score" in res_df.columns
    assert "credit_risk_score" in res_df.columns
