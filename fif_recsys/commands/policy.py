# # Cross-sectional normalization
#     def zscore(s: pd.Series) -> pd.Series:
#         return (s - s.mean()) / (s.std() + 1e-6)

#     SCORE_COLS = {
#         "log_aum": "size_score",
#         "n_ativos": "diversification_score",
#         "n_emissores": "issuer_diversification_score",
#         "credito_share": "credit_risk_score",
#         "related_party_share": "governance_risk_score",
#         "issuer_hhi": "concentration_risk_score",
#     }

#     for raw, score in SCORE_COLS.items():
#         if raw in features.columns:
#             features[score] = features.groupby("competencia")[raw].transform(zscore)
#         else:
#             features[score] = 0.0

#     # Invert bad risk scores
#     for col in ["credit_risk_score", "governance_risk_score", "concentration_risk_score"]:
#         features[col] = features[col] * -1

    # # Customer profiles (weights)
    # CUSTOMER_PROFILES = {
    #     "conservative": {
    #         "size_score": 0.25,
    #         "diversification_score": 0.20,
    #         "issuer_diversification_score": 0.20,
    #         "credit_risk_score": 0.15,
    #         "governance_risk_score": 0.10,
    #         "concentration_risk_score": 0.10,
    #     },
    #     "balanced": {
    #         "size_score": 0.20,
    #         "diversification_score": 0.15,
    #         "issuer_diversification_score": 0.15,
    #         "credit_risk_score": 0.20,
    #         "governance_risk_score": 0.15,
    #         "concentration_risk_score": 0.15,
    #     },
    #     "institutional": {
    #         "size_score": 0.30,
    #         "diversification_score": 0.20,
    #         "issuer_diversification_score": 0.20,
    #         "credit_risk_score": 0.10,
    #         "governance_risk_score": 0.10,
    #         "concentration_risk_score": 0.10,
    #     }
    # }

    # for profile, weights in CUSTOMER_PROFILES.items():
    #     features[f"score_{profile}"] = sum(features[k] * w for k, w in weights.items())