#%%
import os
import requests
import zipfile
import io
import pandas as pd
from tqdm import tqdm


# ——————————————
# Configuration
# ——————————————


BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/"
MONTHS = [
   "202501", "202502", "202503",
   "202504", "202505", "202506",
   "202507", "202508", "202509",
   "202510", "202511", "202512"
]
OUTPUT_DIR = "cvm_cda_data"


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ——————————————
# Download & Load into DataFrames
# ——————————————


dataframes = {}


def download_and_extract_csv(year_month: str):
   """Download a CVM CDA ZIP for a month and load to DataFrame."""
   filename = f"cda_fi_{year_month}.zip"
   url = BASE_URL + filename
   print(f"Downloading: {url}")


   response = requests.get(url, timeout=60)
   response.raise_for_status()


   with zipfile.ZipFile(io.BytesIO(response.content)) as z:
       for member in z.namelist():
           if member.lower().endswith(".csv"):
               print(f"  Loading: {member}")


               with z.open(member) as f:
                   df = pd.read_csv(
                       f,
                       sep=";",
                       dtype=str,
                       encoding="latin1",     # ✅ correct encoding
                       engine="python",       # ✅ handles malformed quotes
                       on_bad_lines="warn",   # ✅ skip bad rows safely
                       # low_memory=False
                   )


               df["competencia"] = year_month
               dataframes[year_month] = df
               break  # assume one CSV per ZIP


# Loop through months and process
for ym in tqdm(MONTHS, desc="Processing months"):
   try:
       download_and_extract_csv(ym)
   except requests.HTTPError as e:
       print(f"  ! Skipped {ym}: {e}")


print("\nDone! Loaded months:", list(dataframes.keys()))
# %%
df_cda = pd.concat(dataframes.values())




######################################################


#%%
import os
import requests
import zipfile
import io
import pandas as pd
from tqdm import tqdm


# ——————————————
# Configuration
# ——————————————


BASE_URL = "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
# YEARS = ["2023", "2024", "2025"]   # ajuste conforme necessário
OUTPUT_DIR = "cvm_cotas_data"


os.makedirs(OUTPUT_DIR, exist_ok=True)


# ——————————————
# Download & Load into DataFrames
# ——————————————


dataframes = {}


def download_and_extract_cotas(year: str):
   """Download CVM INF_DIARIO ZIP for a year and load into DataFrame."""
   filename = f"inf_diario_fi_{year}.zip"
   url = BASE_URL + filename
   print(f"Downloading: {url}")


   response = requests.get(url, timeout=60)
   response.raise_for_status()


   with zipfile.ZipFile(io.BytesIO(response.content)) as z:
       for member in z.namelist():
           if member.lower().endswith(".csv"):
               print(f"  Loading: {member}")


               with z.open(member) as f:
                   df = pd.read_csv(
                       f,
                       sep=";",
                       dtype=str,
                       encoding="latin1",     # padrão CVM
                       engine="python",
                       on_bad_lines="warn"
                   )


               df["ano"] = year
               dataframes[year] = df
               break  # um CSV por ZIP


# Loop through years and process
for y in tqdm(MONTHS, desc="Processing months"):
   try:
       download_and_extract_cotas(y)
   except requests.HTTPError as e:
       print(f"  ! Skipped {y}: {e}")


print("\nDone! Loaded months:", list(dataframes.keys()))


# ——————————————
# Consolidated df_cotas
# ——————————————


df_cotas = pd.concat(dataframes.values(), ignore_index=True)


#%%
df_cda.to_csv('df_cda.csv')
df_cotas.to_csv('df_cotas.csv')




######################################################


#%%
from pandas.tseries.offsets import MonthEnd


df_cotas['score_date'] = (pd.to_datetime(df_cotas['DT_COMPTC']) - pd.Timedelta(days=1) - pd.offsets.MonthEnd(0)).dt.strftime('%Y-%m-%d')
#%%
df_cda['score_date'] = (pd.to_datetime(df_cda['DT_COMPTC']) - pd.Timedelta(days=1) - pd.offsets.MonthEnd(0)).dt.strftime('%Y-%m-%d')


#%%
df_cotas[:50]


# %%
df_cda.dtypes
df_cotas.dtypes


# %%
import pandas as pd
import numpy as np


# ===============================
# 1. Load data (assumed ready)
# ===============================
# df_raw = pd.read_parquet("cvm_cda_concat.parquet")


df = df_cda.copy()


# ===============================
# 2. Basic normalization
# ===============================
NUM_COLS = [
   "VL_PATRIM_LIQ",
   "QT_POS_FINAL",
   "VL_MERC_POS_FINAL",
   "VL_CUSTO_POS_FINAL"
]


DATE_COLS = ["score_date", "DT_COMPTC", "DT_VENC"]


for c in NUM_COLS:
   df[c] = pd.to_numeric(df[c], errors="coerce")


for c in DATE_COLS:
   df[c] = pd.to_datetime(df[c], errors="coerce")


df["competencia"] = df["competencia"].astype(str)


# ===============================
# 3. Core fund-month aggregation
# ===============================
group_keys = ["CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "competencia"]


fund_month = (
   df.groupby(group_keys)
     .agg(
         patrimonio_liq=("VL_PATRIM_LIQ", "max"),
         total_posicao=("VL_MERC_POS_FINAL", "sum"),
         n_ativos=("CD_ATIVO", "nunique"),
         n_emissores=("CPF_CNPJ_EMISSOR", "nunique")
     )
     .reset_index()
)


# ===============================
# 4. Portfolio composition features
# ===============================


# ---- Credit vs Fund-of-Funds proxy
df["is_credito"] = df["TP_APLIC"].isin([
   "Debêntures",
   "Cédula de Crédito",
   "CRI",
   "CRA",
   "Notas Promissórias"
]).astype(int)


credito_share = (
   df.groupby(group_keys)
     .apply(lambda x: np.nansum(x["VL_MERC_POS_FINAL"] * x["is_credito"]) /
                      np.nansum(x["VL_MERC_POS_FINAL"]))
     .rename("credito_share")
     .reset_index()
)


# ---- Related party exposure
df["is_related"] = (df["EMISSOR_LIGADO"] == "S").astype(int)


related_share = (
   df.groupby(group_keys)
     .apply(lambda x: np.nansum(x["VL_MERC_POS_FINAL"] * x["is_related"]) /
                      np.nansum(x["VL_MERC_POS_FINAL"]))
     .rename("related_party_share")
     .reset_index()
)


# ---- Issuer concentration (HHI)
issuer_hhi = (
   df.groupby(group_keys + ["CPF_CNPJ_EMISSOR"])
     .agg(pos=("VL_MERC_POS_FINAL", "sum"))
     .reset_index()
)


issuer_hhi["weight"] = issuer_hhi.groupby(group_keys)["pos"].transform(
   lambda x: x / x.sum()
)


issuer_hhi_score = (
   issuer_hhi.groupby(group_keys)
     .apply(lambda x: np.sum(x["weight"] ** 2))
     .rename("issuer_hhi")
     .reset_index()
)


# ===============================
# 5. Merge all features
# ===============================
features = (
   fund_month
   .merge(credito_share, on=group_keys, how="left")
   .merge(related_share, on=group_keys, how="left")
   .merge(issuer_hhi_score, on=group_keys, how="left")
)


# ===============================
# 6. Feature cleaning & clipping
# ===============================
for col in ["credito_share", "related_party_share", "issuer_hhi"]:
   features[col] = features[col].clip(0, 1)


features["log_aum"] = np.log1p(features["patrimonio_liq"])


# ===============================
# 7. Cross-sectional normalization
# ===============================
def zscore(s):
   return (s - s.mean()) / (s.std() + 1e-6)


SCORE_COLS = {
   "log_aum": "size_score",
   "n_ativos": "diversification_score",
   "n_emissores": "issuer_diversification_score",
   "credito_share": "credit_risk_score",
   "related_party_share": "governance_risk_score",
   "issuer_hhi": "concentration_risk_score"
}


for raw, score in SCORE_COLS.items():
   features[score] = features.groupby("competencia")[raw].transform(zscore)


# Invert "bad" risk scores
features["credit_risk_score"] *= -1
features["governance_risk_score"] *= -1
features["concentration_risk_score"] *= -1


# ===============================
# 8. Customer profile definitions
# ===============================
CUSTOMER_PROFILES = {
   "conservative": {
       "size_score": 0.25,
       "diversification_score": 0.20,
       "issuer_diversification_score": 0.20,
       "credit_risk_score": 0.15,
       "governance_risk_score": 0.10,
       "concentration_risk_score": 0.10,
   },
   "balanced": {
       "size_score": 0.20,
       "diversification_score": 0.15,
       "issuer_diversification_score": 0.15,
       "credit_risk_score": 0.20,
       "governance_risk_score": 0.15,
       "concentration_risk_score": 0.15,
   },
   "institutional": {
       "size_score": 0.30,
       "diversification_score": 0.20,
       "issuer_diversification_score": 0.20,
       "credit_risk_score": 0.10,
       "governance_risk_score": 0.10,
       "concentration_risk_score": 0.10,
   }
}


# ===============================
# 9. Score aggregation
# ===============================
for profile, weights in CUSTOMER_PROFILES.items():
   features[f"score_{profile}"] = sum(
       features[k] * w for k, w in weights.items()
   )


# ===============================
# 10. Top-5 funds per profile
# ===============================
LATEST_COMP = features["competencia"].max()


top5 = {}
for profile in CUSTOMER_PROFILES:
   top5[profile] = (
       features[features["competencia"] == LATEST_COMP]
       .sort_values(f"score_{profile}", ascending=False)
       .head(5)
       [
           [
               "CNPJ_FUNDO_CLASSE",
               "DENOM_SOCIAL",
               "patrimonio_liq",
               f"score_{profile}",
               "credito_share",
               "issuer_hhi",
               "related_party_share"
           ]
       ]
   )


# ===============================
# 11. Output
# ===============================
for profile, df_out in top5.items():
   print(f"\n=== TOP 5 – {profile.upper()} ===")
   print(df_out.to_string(index=False))
