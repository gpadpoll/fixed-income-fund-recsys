import pandas as pd
import numpy as np
from pathlib import Path

from fif_recsys.commands import feature


def _sample_df():
    return pd.DataFrame([
        {
            "CNPJ_FUNDO_CLASSE": "F1",
            "DENOM_SOCIAL": "Fund A",
            "competencia": "202501",
            "VL_PATRIM_LIQ": 1000.0,
            "VL_MERC_POS_FINAL": 1000.0,
            "CD_ATIVO": "A1",
            "CPF_CNPJ_EMISSOR": "E1",
            "TP_APLIC": "Debêntures",
            "EMISSOR_LIGADO": "N",
        },
        {
            "CNPJ_FUNDO_CLASSE": "F1",
            "DENOM_SOCIAL": "Fund A",
            "competencia": "202501",
            "VL_PATRIM_LIQ": 1000.0,
            "VL_MERC_POS_FINAL": 0.0,
            "CD_ATIVO": "A2",
            "CPF_CNPJ_EMISSOR": "E2",
            "TP_APLIC": "Tesouro",
            "EMISSOR_LIGADO": "S",
        },
        {
            "CNPJ_FUNDO_CLASSE": "F2",
            "DENOM_SOCIAL": "Fund B",
            "competencia": "202501",
            "VL_PATRIM_LIQ": 500.0,
            "VL_MERC_POS_FINAL": 500.0,
            "CD_ATIVO": "A3",
            "CPF_CNPJ_EMISSOR": "E1",
            "TP_APLIC": "CRI",
            "EMISSOR_LIGADO": "N",
        },
    ])


def test_credito_share_function():
    df = _sample_df()
    group_keys = ["CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "competencia"]
    credito_list = ["Debêntures", "CRI"]

    out = feature._credito_share_feature_fn(df, group_keys, credito_list)
    # Convert to dict mapping fund->value for assertions
    vals = dict(zip(out["CNPJ_FUNDO_CLASSE"], out["credito_share"]))

    assert np.isclose(vals["F1"], 1.0)
    assert np.isclose(vals["F2"], 1.0)


def test_related_party_share_function():
    df = _sample_df()
    group_keys = ["CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "competencia"]

    out = feature._related_party_share_feature_fn(df, group_keys)
    vals = dict(zip(out["CNPJ_FUNDO_CLASSE"], out["related_party_share"]))

    assert np.isclose(vals["F1"], 0.0)
    assert np.isclose(vals["F2"], 0.0)


def test_hhi_function():
    df = _sample_df()
    group_keys = ["CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "competencia"]

    out = feature._hhi_feature_fn(df, group_keys)
    vals = dict(zip(out["CNPJ_FUNDO_CLASSE"], out["issuer_hhi"]))

    # Given single issuer dominance in both funds, HHI should be 1.0
    assert np.isclose(vals["F1"], 1.0)
    assert np.isclose(vals["F2"], 1.0)


def test_compute_all_features_with_registry():
    df = _sample_df()
    datasets = {"myds": df}

    yaml_cfg = {
        "feature": {
            "group_keys": ["CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "competencia"],
            "feature_registry": {
                "myds": {
                    # simple aggregation
                    "patrimonio_liq": {"method": "max", "args": ["VL_PATRIM_LIQ"]},
                    "n_ativos": {"method": "nunique", "args": ["CD_ATIVO"]},
                    # custom function
                    "credito_share": {"method": "credito_share_feature_fn", "args": [["Debêntures", "CRI"]]},
                    "related_party_share": {"method": "related_party_share_feature_fn", "args": []},
                    "issuer_hhi": {"method": "hhi_feature_fn", "args": []},
                }
            }
        }
    }

    out = feature.compute_all_features(datasets, yaml_cfg, feature.FEATURE_ENGINE)

    # Basic assertions
    assert "patrimonio_liq" in out.columns
    assert "credito_share" in out.columns
    assert "related_party_share" in out.columns
    assert "issuer_hhi" in out.columns

    # Two funds
    assert set(out["CNPJ_FUNDO_CLASSE"]) == {"F1", "F2"}


def test_compute_features_from_yaml_end_to_end(tmp_path):
    # prepare partitioned dataset
    ds = tmp_path / "myds" / "period=202501"
    ds.mkdir(parents=True)
    df = _sample_df()
    (ds / "data.csv").write_text(df.to_csv(index=False))

    yaml_cfg = {
        "feature": {
            "group_keys": ["CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "competencia"],
            "feature_registry": {
                "myds": {
                    "patrimonio_liq": {"method": "max", "args": ["VL_PATRIM_LIQ"]},
                    "credito_share": {"method": "credito_share_feature_fn", "args": [["Debêntures", "CRI"]]},
                }
            }
        }
    }

    final = feature.compute_features_from_yaml(tmp_path, yaml_cfg)

    assert "patrimonio_liq" in final.columns
    assert "credito_share" in final.columns
    assert set(final["CNPJ_FUNDO_CLASSE"]) == {"F1", "F2"}
