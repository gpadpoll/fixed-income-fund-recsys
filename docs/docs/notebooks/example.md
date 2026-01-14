# Fixed Income Funds Recommendation System

This notebook demonstrates the data ingestion, feature engineering, and scoring pipeline for a fixed-income fund recommendation system.

What you'll see here:

- Define data sources and feature/score configuration using a YAML manifest.
- Fetch and prepare datasets (partitioned by period).
- Compute fund-month features and derive scores using configurable YAML registries.
- Produce ranked fund profiles based on profile weights.

**Prerequisites:** Python packages: `pandas`, `requests`, `pyyaml`. Optional: `pyarrow` or `fastparquet` for Parquet I/O.

> Note: For quick demos this notebook may use local CSV fallbacks; substitute `fetch_manifest(...)` to run the full end-to-end pipeline against remote sources.


```python
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


## Quick check

Run a quick sanity check to ensure the package is importable and that basic helpers (like `hello()`) work as expected. This is useful to confirm the development environment is set up correctly before running heavier pipeline steps.


```python
from fif_recsys import hello

# This will print to the notebook output
hello()
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Hello from fif_recsys!
</pre>



## Configuration manifest (YAML)

The configuration dictionary (`config_d`) defines how data is fetched and how features and scores are computed.

- `fetch`: datasets to download. Each dataset includes `base_url`, `periods`, and `filename_template`.
- `feature`: registry of features to compute, including aggregation method and optional adjustments.
- `score`: scoring definitions (type, feature source, and adjustments like `invert`).
- `profile`: named profile weightings used to aggregate scores into a single ranking for each investor profile.

Edit these values to match your data sources and scoring preferences.


```python
import yaml

config_d = yaml.safe_load("""
fetch:
    cda:
        base_url: "https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/"
        periods:
            - "202501"
            - "202502"
            - "202503"
            - "202504"
            - "202505"
            - "202506"
            - "202507"
            - "202508"
            - "202509"
            - "202510"
            - "202511"
            - "202512"
        filename_template: "cda_fi_{period}.zip"

    cotas:
        base_url: "https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/"
        periods:
            - "202301"
            - "202302"
            - "202303"
            - "202304"
            - "202305"
            - "202306"
            - "202307"
            - "202308"
            - "202309"
            - "202310"
            - "202311"
            - "202312"
                          
            - "202401"
            - "202402"
            - "202403"
            - "202404"
            - "202405"
            - "202406"
            - "202407"
            - "202408"
            - "202409"
            - "202410"
            - "202411"
            - "202412"
                          
            - "202501"
            - "202502"
            - "202503"
            - "202504"
            - "202505"
            - "202506"
            - "202507"
            - "202508"
            - "202509"
            - "202510"
            - "202511"
            - "202512"
        filename_template: "inf_diario_fi_{period}.zip"
feature:
    group_keys:
        - CNPJ_FUNDO_CLASSE
        - DENOM_SOCIAL
        - reference_date
    feature_registry:
        cda:
            patrimonio_liq:
                description: "Maximum reported net asset value per fund-month."
                method: max
                args:
                    - VL_PATRIM_LIQ
                            
            log_aum:
                description: "Maximum reported net asset value per fund-month."
                method: max
                args:
                    - VL_PATRIM_LIQ
                adjustment:
                    - log

            total_posicao:
                description: "Sum of final market value of all positions in the period."
                method: sum
                args:
                    - VL_MERC_POS_FINAL

            n_ativos:
                description: "Number of unique assets in the fund portfolio."
                method: nunique
                args:
                    - CD_ATIVO

            n_emissores:
                description: "Number of unique issuers in the fund portfolio."
                method: nunique
                args:
                    - CPF_CNPJ_EMISSOR

            credito_share:
                description: "Weighted share of credit-linked assets in the portfolio."
                method: credito_share_feature_fn
                args:
                    - ["Debêntures", "Cédula de Crédito", "CRI", "CRA", "Notas Promissórias"]
                adjustment:
                    - clip

            related_party_share:
                description: "Weighted share of related-party issuers."
                method: related_party_share_feature_fn
                adjustment:
                    - clip

            issuer_hhi:
                description: "Herfindahl-Hirschman index based on issuer weights."
                method: hhi_feature_fn
                adjustment:
                    - clip
                    - coalesce
        cotas:
            
score:
    size_score:
        type: zscore
        description: >
            Measures the relative size of the fund based on its assets under
            management. Larger funds typically exhibit greater operational
            stability, better liquidity access, and lower idiosyncratic risk.
            Computed using the z-score of the log-transformed AUM (log_aum).
        args:
            feature: log_aum

    diversification_score:
        type: zscore
        description: >
            Evaluates how diversified the fund's portfolio is in terms of
            the number of unique assets held. Higher values indicate broader
            asset diversification, reducing exposure to security-specific risks.
        args:
            feature: n_ativos

    issuer_diversification_score:
        type: zscore
        description: >
            Measures diversification across issuers by counting how many distinct
            counterparties the fund is exposed to. Funds with exposures distributed
            across more issuers typically have lower concentration and reduced
            issuer-specific credit risk.
        args:
            feature: n_emissores

    credit_risk_score:
        type: zscore
        description: >
            Quantifies the fund's exposure to credit-linked instruments such as
            debentures, CRIs/CRAs, and promissory notes. A higher credit share
            typically increases sensitivity to credit events. The score is inverted
            so that higher credit exposure corresponds to a lower (worse) score.
        args:
            feature: credito_share
        adjustment:
            - invert

    governance_risk_score:
        type: zscore
        description: >
            Captures exposure to related-party transactions, which may increase
            governance risk due to potential conflicts of interest and reduced
            market discipline. The score is inverted, so funds with higher
            related-party share receive a lower (worse) score.
        args:
            feature: related_party_share
        adjustment:
            - invert

    concentration_risk_score:
        type: zscore
        description: >
            Measures portfolio concentration using the Herfindahl-Hirschman Index
            (HHI) computed over issuer exposure weights. Higher HHI values indicate
            more concentrated portfolios and elevated idiosyncratic and liquidity
            risks. Score is inverted so higher concentration yields a lower score.
        args:
            feature: issuer_hhi
        adjustment:
            - invert
profile:
  conservative:
    description: >
      Designed for risk-averse investors prioritizing capital preservation and stability.
      Emphasizes fund size, diversification, and issuer spread to minimize volatility,
      while keeping exposure to credit and governance risks tightly controlled.
    size_score: 0.25
    diversification_score: 0.20
    issuer_diversification_score: 0.20
    credit_risk_score: 0.15
    governance_risk_score: 0.10
    concentration_risk_score: 0.10

  balanced:
    description: >
      Suitable for investors seeking a middle ground between safety and return.
      Balances diversification and issuer exposure with moderate tolerance for credit
      and concentration risks, aiming for a stable but growth-oriented allocation.
    size_score: 0.20
    diversification_score: 0.15
    issuer_diversification_score: 0.15
    credit_risk_score: 0.20
    governance_risk_score: 0.15
    concentration_risk_score: 0.15

  institutional:
    description: >
      Targeted at large professional allocators who value scale and diversification
      but can tolerate more concentrated or complex positions. Prioritizes fund size
      and issuer spread while placing relatively lower weight on credit and governance constraints.
    size_score: 0.30
    diversification_score: 0.20
    issuer_diversification_score: 0.20
    credit_risk_score: 0.10
    governance_risk_score: 0.10
    concentration_risk_score: 0.10

""")
```

## Fetch datasets

Use `fetch_manifest` to download and assemble datasets defined in the manifest. The function returns a `dict` mapping dataset names to `pandas.DataFrame` objects and writes partitioned files to `output_dir/<dataset>/period=<period>/data.parquet` when a Parquet engine is available (a CSV fallback is used otherwise).

Example usage (below) demonstrates both the programmatic fetch and a temporary offline fallback for quick demos.


```python
from pathlib import Path

from fif_recsys.commands.data import fetch_manifest


data_sources_d = fetch_manifest(config_d['fetch'], output_dir=Path("/tmp"))

```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202501.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202501.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1011: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2071: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2279: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 3391: field larger than field limit (131072)
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202502.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202502.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1073: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2175: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2373: field larger than field limit (131072)
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202503.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202503.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1090: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2316: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2480: field larger than field limit (131072)
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202504.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202504.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1190: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2459: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2625: field larger than field limit (131072)
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202505.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202505.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1215: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2627: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2816: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 3394: field larger than field limit (131072)
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202506.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202506.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 648: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1430: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2930: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 3148: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 3889: ';' expected after '"'
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202507.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202507.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 347: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1139: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2693: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2917: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 3737: ';' expected after '"'
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202508.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202508.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 241: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1044: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2640: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2733: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 3606: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 4049: field larger than field limit (131072)
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202509.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202509.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 3164: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 3921: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 5597: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 5687: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 6578: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 7053: field larger than field limit (131072)
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202510.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202510.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 174: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 614: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1935: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2260: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 2459: field larger than field limit (131072)
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202511.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202511.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 145: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 355: ';' expected after '"'
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1248: field larger than field limit (131072)
    
      df = pd.read_csv(
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 1411: ';' expected after '"'
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/cda_fi_202512.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">cda_fie_202512.csv</span>
</pre>



    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/data.py:41: ParserWarning: Skipping line 684: field larger than field limit (131072)
    
      df = pd.read_csv(



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202501</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202502</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202503</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202504</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202505</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202506</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202507</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202508</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202509</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202510</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202511</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cda/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202512</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008000; text-decoration-color: #008000">Saved</span> cda → <span style="color: #800080; text-decoration-color: #800080">/tmp/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">cda</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202301.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202301.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202302.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202302.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202303.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202303.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202304.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202304.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202305.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202305.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202306.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202306.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202307.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202307.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202308.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202308.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202309.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202309.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202310.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202310.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202311.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202311.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202312.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202312.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202401.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202401.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202402.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202402.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202403.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202403.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202404.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202404.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202405.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202405.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202406.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202406.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202407.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202407.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202408.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202408.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202409.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202409.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202410.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202410.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202411.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202411.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202412.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202412.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202501.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202501.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202502.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202502.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202503.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202503.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202504.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202504.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202505.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202505.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202506.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202506.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202507.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202507.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202508.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202508.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202509.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202509.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202510.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202510.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202511.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202511.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008080; text-decoration-color: #008080">Downloading</span> <span style="color: #0000ff; text-decoration-color: #0000ff; text-decoration: underline">https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_202512.zip</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  Parsing <span style="color: #008000; text-decoration-color: #008000">inf_diario_fi_202512.csv</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202301</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202302</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202303</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202304</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202305</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202306</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202307</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202308</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202309</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202310</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202311</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202312</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202401</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202402</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202403</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202404</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202405</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202406</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202407</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202408</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202409</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202410</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202411</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202412</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202501</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202502</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202503</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202504</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202505</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202506</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202507</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202508</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202509</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202510</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202511</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #808000; text-decoration-color: #808000">Pyarrow not available; wrote CSV instead:</span> <span style="color: #800080; text-decoration-color: #800080">/tmp/cotas/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">period</span>=<span style="color: #008080; text-decoration-color: #008080; font-weight: bold">202512</span>/data.csv
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #008000; text-decoration-color: #008000">Saved</span> cotas → <span style="color: #800080; text-decoration-color: #800080">/tmp/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">cotas</span>
</pre>



## Compute features

Call `compute_all_features` (or `compute_all_features(...)` via the `FEATURE_ENGINE`) to aggregate fund-month features according to your `feature_registry`. The result is a DataFrame with one row per fund-month and computed features ready for scoring.


```python
from fif_recsys.commands.feature import compute_all_features, FEATURE_ENGINE


feature_df = compute_all_features(data_sources_d, config_d, FEATURE_ENGINE)

feature_df.head()
```

    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/feature.py:26: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      def build_feature_engine(feature_engine: Dict, group_keys: List[str], registry: Any):
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/feature.py:26: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      def build_feature_engine(feature_engine: Dict, group_keys: List[str], registry: Any):
    /opt/homebrew/Caskroom/miniconda/base/envs/py313-fif/lib/python3.13/site-packages/pandas/core/arraylike.py:399: RuntimeWarning: invalid value encountered in log1p
      result = getattr(ufunc, method)(*inputs, **kwargs)
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/feature.py:26: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      def build_feature_engine(feature_engine: Dict, group_keys: List[str], registry: Any):
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/feature.py:26: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      def build_feature_engine(feature_engine: Dict, group_keys: List[str], registry: Any):
    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/feature.py:26: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      def build_feature_engine(feature_engine: Dict, group_keys: List[str], registry: Any):


    [33mSkipping dataset [0m[33m'cotas'[0m[33m: no features defined in registry.[0m


    /Users/gustavopolleti/dev/fixed-income-fund-recsys/fif_recsys/commands/feature.py:158: FutureWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNPJ_FUNDO_CLASSE</th>
      <th>DENOM_SOCIAL</th>
      <th>reference_date</th>
      <th>patrimonio_liq</th>
      <th>log_aum</th>
      <th>total_posicao</th>
      <th>n_ativos</th>
      <th>n_emissores</th>
      <th>credito_share</th>
      <th>related_party_share</th>
      <th>issuer_hhi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06.323.688/0001-27</td>
      <td>IT NOW PIBB IBRX-50 FUNDO DE ÍNDICE RESPONSABI...</td>
      <td>2026-01-14</td>
      <td>9.863479e+08</td>
      <td>20.709520</td>
      <td>5.825532e+09</td>
      <td>55</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.125814</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09.260.031/0001-56</td>
      <td>FUNDO DE INVESTIMENTO EM QUOTAS DE FUNDO DE IN...</td>
      <td>2026-01-14</td>
      <td>8.236450e+07</td>
      <td>18.226665</td>
      <td>5.039806e+08</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>0.479135</td>
      <td>0.298536</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.292.322/0001-05</td>
      <td>KONDOR KOBOLD FUNDO DE INVESTIMENTO EM COTAS D...</td>
      <td>2026-01-14</td>
      <td>5.282581e+08</td>
      <td>20.085096</td>
      <td>4.007817e+09</td>
      <td>0</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.999686</td>
      <td>0.612586</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.406.511/0001-61</td>
      <td>ISHARES IBOVESPA CLASSE DE ÍNDICE - RESPONSABI...</td>
      <td>2026-01-14</td>
      <td>1.499092e+10</td>
      <td>23.430710</td>
      <td>1.028544e+11</td>
      <td>103</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.013466</td>
      <td>0.364377</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.406.600/0001-08</td>
      <td>ISHARES BM&amp;FBOVESPA SMALL CAP CLASSE DE  ÍNDIC...</td>
      <td>2026-01-14</td>
      <td>2.112755e+09</td>
      <td>21.471258</td>
      <td>1.813606e+10</td>
      <td>131</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.035685</td>
      <td>0.856891</td>
    </tr>
  </tbody>
</table>
</div>



## Compute scores

Convert features into normalized scores using `compute_scores_from_yaml`. The `score` section in the configuration defines score types (e.g., `zscore`) and optional adjustments (e.g., `invert`). The resulting DataFrame will contain the base features and the derived score columns.


```python
from fif_recsys.commands.model import compute_scores_from_yaml

score_df = compute_scores_from_yaml(feature_df, config_d)

score_df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNPJ_FUNDO_CLASSE</th>
      <th>DENOM_SOCIAL</th>
      <th>reference_date</th>
      <th>patrimonio_liq</th>
      <th>log_aum</th>
      <th>total_posicao</th>
      <th>n_ativos</th>
      <th>n_emissores</th>
      <th>credito_share</th>
      <th>related_party_share</th>
      <th>issuer_hhi</th>
      <th>size_score</th>
      <th>diversification_score</th>
      <th>issuer_diversification_score</th>
      <th>credit_risk_score</th>
      <th>governance_risk_score</th>
      <th>concentration_risk_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06.323.688/0001-27</td>
      <td>IT NOW PIBB IBRX-50 FUNDO DE ÍNDICE RESPONSABI...</td>
      <td>2026-01-14</td>
      <td>9.863479e+08</td>
      <td>20.709520</td>
      <td>5.825532e+09</td>
      <td>55</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.125814</td>
      <td>1.000000</td>
      <td>1.270380</td>
      <td>2.366598</td>
      <td>-0.537225</td>
      <td>0.064385</td>
      <td>0.500056</td>
      <td>-1.078161</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09.260.031/0001-56</td>
      <td>FUNDO DE INVESTIMENTO EM QUOTAS DE FUNDO DE IN...</td>
      <td>2026-01-14</td>
      <td>8.236450e+07</td>
      <td>18.226665</td>
      <td>5.039806e+08</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>0.479135</td>
      <td>0.298536</td>
      <td>0.205049</td>
      <td>-0.227050</td>
      <td>-0.023439</td>
      <td>0.064385</td>
      <td>-0.376790</td>
      <td>0.839722</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.292.322/0001-05</td>
      <td>KONDOR KOBOLD FUNDO DE INVESTIMENTO EM COTAS D...</td>
      <td>2026-01-14</td>
      <td>5.282581e+08</td>
      <td>20.085096</td>
      <td>4.007817e+09</td>
      <td>0</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.999686</td>
      <td>0.612586</td>
      <td>1.002455</td>
      <td>-0.227050</td>
      <td>-0.317031</td>
      <td>0.064385</td>
      <td>-1.668657</td>
      <td>-0.018927</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.406.511/0001-61</td>
      <td>ISHARES IBOVESPA CLASSE DE ÍNDICE - RESPONSABI...</td>
      <td>2026-01-14</td>
      <td>1.499092e+10</td>
      <td>23.430710</td>
      <td>1.028544e+11</td>
      <td>103</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.013466</td>
      <td>0.364377</td>
      <td>2.437974</td>
      <td>4.630145</td>
      <td>0.049959</td>
      <td>0.064385</td>
      <td>0.778873</td>
      <td>0.659707</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.406.600/0001-08</td>
      <td>ISHARES BM&amp;FBOVESPA SMALL CAP CLASSE DE  ÍNDIC...</td>
      <td>2026-01-14</td>
      <td>2.112755e+09</td>
      <td>21.471258</td>
      <td>1.813606e+10</td>
      <td>131</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.035685</td>
      <td>0.856891</td>
      <td>1.597222</td>
      <td>5.950548</td>
      <td>0.196754</td>
      <td>0.064385</td>
      <td>0.723733</td>
      <td>-0.686886</td>
    </tr>
  </tbody>
</table>
</div>



## Compute profile rankings

Use `compute_profile_scores_from_yaml` (from `fif_recsys.commands.policy`) to aggregate weighted scores into a single profile score and ranking for each fund. Profiles are defined in the `profile` section of the configuration (e.g., `conservative`, `balanced`, `institutional`).


```python
from fif_recsys.commands.policy import compute_profile_scores_from_yaml

ranking_df = compute_profile_scores_from_yaml(score_df.fillna(0), config_d)

ranking_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNPJ_FUNDO_CLASSE</th>
      <th>DENOM_SOCIAL</th>
      <th>reference_date</th>
      <th>patrimonio_liq</th>
      <th>log_aum</th>
      <th>total_posicao</th>
      <th>n_ativos</th>
      <th>n_emissores</th>
      <th>credito_share</th>
      <th>related_party_share</th>
      <th>...</th>
      <th>issuer_diversification_score</th>
      <th>credit_risk_score</th>
      <th>governance_risk_score</th>
      <th>concentration_risk_score</th>
      <th>score_conservative</th>
      <th>rank_conservative</th>
      <th>score_balanced</th>
      <th>rank_balanced</th>
      <th>score_institutional</th>
      <th>rank_institutional</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06.323.688/0001-27</td>
      <td>IT NOW PIBB IBRX-50 FUNDO DE ÍNDICE RESPONSABI...</td>
      <td>2026-01-14</td>
      <td>9.863479e+08</td>
      <td>20.709520</td>
      <td>5.825532e+09</td>
      <td>55</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.125814</td>
      <td>...</td>
      <td>-0.537225</td>
      <td>0.064385</td>
      <td>0.500056</td>
      <td>-1.078161</td>
      <td>0.635317</td>
      <td>84</td>
      <td>0.454643</td>
      <td>125</td>
      <td>0.695616</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>09.260.031/0001-56</td>
      <td>FUNDO DE INVESTIMENTO EM QUOTAS DE FUNDO DE IN...</td>
      <td>2026-01-14</td>
      <td>8.236450e+07</td>
      <td>18.226665</td>
      <td>5.039806e+08</td>
      <td>0</td>
      <td>8</td>
      <td>0.0</td>
      <td>0.479135</td>
      <td>...</td>
      <td>-0.023439</td>
      <td>0.064385</td>
      <td>-0.376790</td>
      <td>0.839722</td>
      <td>0.057116</td>
      <td>363</td>
      <td>0.085753</td>
      <td>367</td>
      <td>0.064149</td>
      <td>367</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10.292.322/0001-05</td>
      <td>KONDOR KOBOLD FUNDO DE INVESTIMENTO EM COTAS D...</td>
      <td>2026-01-14</td>
      <td>5.282581e+08</td>
      <td>20.085096</td>
      <td>4.007817e+09</td>
      <td>0</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.999686</td>
      <td>...</td>
      <td>-0.317031</td>
      <td>0.064385</td>
      <td>-1.668657</td>
      <td>-0.018927</td>
      <td>-0.017303</td>
      <td>442</td>
      <td>-0.121382</td>
      <td>598</td>
      <td>0.029600</td>
      <td>408</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10.406.511/0001-61</td>
      <td>ISHARES IBOVESPA CLASSE DE ÍNDICE - RESPONSABI...</td>
      <td>2026-01-14</td>
      <td>1.499092e+10</td>
      <td>23.430710</td>
      <td>1.028544e+11</td>
      <td>103</td>
      <td>9</td>
      <td>0.0</td>
      <td>0.013466</td>
      <td>...</td>
      <td>0.049959</td>
      <td>0.064385</td>
      <td>0.778873</td>
      <td>0.659707</td>
      <td>1.699030</td>
      <td>7</td>
      <td>1.418274</td>
      <td>7</td>
      <td>1.817709</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10.406.600/0001-08</td>
      <td>ISHARES BM&amp;FBOVESPA SMALL CAP CLASSE DE  ÍNDIC...</td>
      <td>2026-01-14</td>
      <td>2.112755e+09</td>
      <td>21.471258</td>
      <td>1.813606e+10</td>
      <td>131</td>
      <td>11</td>
      <td>0.0</td>
      <td>0.035685</td>
      <td>...</td>
      <td>0.196754</td>
      <td>0.064385</td>
      <td>0.723733</td>
      <td>-0.686886</td>
      <td>1.642109</td>
      <td>8</td>
      <td>1.259944</td>
      <td>9</td>
      <td>1.718750</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



## Next steps & CLI

- Run the full pipeline from the command line using the Typer-based CLI:
  - `fif-recsys data fetch` to download and prepare datasets
  - `fif-recsys feature build` to compute and write feature tables
  - `fif-recsys model score` to compute scores

- Tips:
  - Install `pyarrow` for faster Parquet I/O when running on large datasets.
  - For reproducible fetches, consider passing a deterministic `reference_date` to `fetch_manifest`.

Feel free to update this notebook with real data paths and run the pipeline end-to-end.

## Inspecting pipeline outputs

If you ran the Docker pipeline and mounted an output directory (e.g., `/tmp/fif_data` on the host → `/data` in the container), the pipeline writes the final profile-scored table to `features_profile_scored.parquet` or `features_profile_scored.csv` in that directory. Use the cell below to load and preview the output; update the `output_path` if you used a different directory.


```python
ranking_df.sort_values(by='rank_conservative', ascending=True)[:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNPJ_FUNDO_CLASSE</th>
      <th>DENOM_SOCIAL</th>
      <th>reference_date</th>
      <th>patrimonio_liq</th>
      <th>log_aum</th>
      <th>total_posicao</th>
      <th>n_ativos</th>
      <th>n_emissores</th>
      <th>credito_share</th>
      <th>related_party_share</th>
      <th>...</th>
      <th>issuer_diversification_score</th>
      <th>credit_risk_score</th>
      <th>governance_risk_score</th>
      <th>concentration_risk_score</th>
      <th>score_conservative</th>
      <th>rank_conservative</th>
      <th>score_balanced</th>
      <th>rank_balanced</th>
      <th>score_institutional</th>
      <th>rank_institutional</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>219</th>
      <td>40.155.573/0001-09</td>
      <td>TREND ETF IBOVESPA CLASSE DE ÍNDICE -  RESPONS...</td>
      <td>2026-01-14</td>
      <td>1.001094e+09</td>
      <td>20.724359</td>
      <td>9.626491e+09</td>
      <td>130</td>
      <td>130</td>
      <td>0.0</td>
      <td>0.001832</td>
      <td>...</td>
      <td>8.931106</td>
      <td>0.064385</td>
      <td>0.807746</td>
      <td>1.533615</td>
      <td>3.529880</td>
      <td>1</td>
      <td>2.844605</td>
      <td>1</td>
      <td>3.590498</td>
      <td>1</td>
    </tr>
    <tr>
      <th>133</th>
      <td>32.203.211/0001-18</td>
      <td>FUNDO DE INVESTIMENTO DE ÍNDICE - CLASSE DE IN...</td>
      <td>2026-01-14</td>
      <td>1.910818e+09</td>
      <td>21.370797</td>
      <td>1.100393e+10</td>
      <td>93</td>
      <td>79</td>
      <td>0.0</td>
      <td>0.059015</td>
      <td>...</td>
      <td>5.187812</td>
      <td>0.064385</td>
      <td>0.665834</td>
      <td>1.495788</td>
      <td>2.483626</td>
      <td>2</td>
      <td>2.049902</td>
      <td>2</td>
      <td>2.558113</td>
      <td>2</td>
    </tr>
    <tr>
      <th>419</th>
      <td>48.643.130/0001-79</td>
      <td>FUNDO DE INVESTIMENTO DE ÍNDICE - CI B-INDEX M...</td>
      <td>2026-01-14</td>
      <td>7.718110e+07</td>
      <td>18.161665</td>
      <td>2.279355e+08</td>
      <td>97</td>
      <td>70</td>
      <td>0.0</td>
      <td>0.061258</td>
      <td>...</td>
      <td>4.527231</td>
      <td>0.064385</td>
      <td>0.660266</td>
      <td>1.587268</td>
      <td>2.053588</td>
      <td>3</td>
      <td>1.716604</td>
      <td>3</td>
      <td>2.059226</td>
      <td>3</td>
    </tr>
    <tr>
      <th>143</th>
      <td>34.606.480/0001-50</td>
      <td>BB ETF IBOVESPA FUNDO DE ÍNDICE RESPONSABILIDA...</td>
      <td>2026-01-14</td>
      <td>2.199202e+09</td>
      <td>21.511360</td>
      <td>1.252859e+10</td>
      <td>95</td>
      <td>45</td>
      <td>0.0</td>
      <td>0.036515</td>
      <td>...</td>
      <td>2.692283</td>
      <td>0.064385</td>
      <td>0.721672</td>
      <td>0.820588</td>
      <td>1.956525</td>
      <td>4</td>
      <td>1.608878</td>
      <td>5</td>
      <td>2.034027</td>
      <td>4</td>
    </tr>
    <tr>
      <th>725</th>
      <td>57.848.980/0001-02</td>
      <td>BB ETF ÍNDICE BOVESPA B3 BR+ FUNDO DE ÍNDICE R...</td>
      <td>2026-01-14</td>
      <td>2.608017e+07</td>
      <td>17.076686</td>
      <td>2.580044e+08</td>
      <td>106</td>
      <td>67</td>
      <td>0.0</td>
      <td>0.033134</td>
      <td>...</td>
      <td>4.307037</td>
      <td>0.064385</td>
      <td>0.730064</td>
      <td>1.245778</td>
      <td>1.950878</td>
      <td>5</td>
      <td>1.613376</td>
      <td>4</td>
      <td>1.933240</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
# Load and preview the profile-scored table
from pathlib import Path
import pandas as pd

# Update this path to the directory you mounted into the container (host path: /tmp/fif_data)
output_dir = Path("/tmp/fif_data")

pj = output_dir / "features_profile_scored.parquet"
pcsv = output_dir / "features_profile_scored.csv"

if pj.exists():
    df = pd.read_parquet(pj)
elif pcsv.exists():
    df = pd.read_csv(pcsv)
else:
    raise FileNotFoundError(f"No profile-scored output found at {pj} or {pcsv}. Make sure you mounted the output dir and ran the pipeline.")

# Quick preview
print("Path:", pj if pj.exists() else pcsv)
print("Rows:", len(df))
print("Columns:", list(df.columns))
df.head()

```

    Path: /tmp/fif_data/features_profile_scored.csv
    Rows: 5476
    Columns: ['CNPJ_FUNDO_CLASSE', 'DENOM_SOCIAL', 'competencia', 'patrimonio_liq', 'log_aum', 'total_posicao', 'n_ativos', 'n_emissores', 'credito_share', 'related_party_share', 'issuer_hhi', 'size_score', 'diversification_score', 'issuer_diversification_score', 'credit_risk_score', 'governance_risk_score', 'concentration_risk_score', 'score_conservative', 'rank_conservative', 'score_balanced', 'rank_balanced', 'score_institutional', 'rank_institutional']





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CNPJ_FUNDO_CLASSE</th>
      <th>DENOM_SOCIAL</th>
      <th>competencia</th>
      <th>patrimonio_liq</th>
      <th>log_aum</th>
      <th>total_posicao</th>
      <th>n_ativos</th>
      <th>n_emissores</th>
      <th>credito_share</th>
      <th>related_party_share</th>
      <th>...</th>
      <th>issuer_diversification_score</th>
      <th>credit_risk_score</th>
      <th>governance_risk_score</th>
      <th>concentration_risk_score</th>
      <th>score_conservative</th>
      <th>rank_conservative</th>
      <th>score_balanced</th>
      <th>rank_balanced</th>
      <th>score_institutional</th>
      <th>rank_institutional</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>06.323.688/0001-27</td>
      <td>IT NOW PIBB IBRX-50 FUNDO DE ÍNDICE RESPONSABI...</td>
      <td>202506</td>
      <td>9.630971e+08</td>
      <td>20.685665</td>
      <td>9.659350e+08</td>
      <td>47</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.127924</td>
      <td>...</td>
      <td>-0.606022</td>
      <td>0.081555</td>
      <td>0.486351</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>06.323.688/0001-27</td>
      <td>IT NOW PIBB IBRX-50 FUNDO DE ÍNDICE RESPONSABI...</td>
      <td>202507</td>
      <td>9.206483e+08</td>
      <td>20.640589</td>
      <td>9.225100e+08</td>
      <td>46</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.127067</td>
      <td>...</td>
      <td>-0.538943</td>
      <td>0.066368</td>
      <td>0.469103</td>
      <td>-0.97675</td>
      <td>0.547754</td>
      <td>533</td>
      <td>0.394923</td>
      <td>765</td>
      <td>0.60993</td>
      <td>485</td>
    </tr>
    <tr>
      <th>2</th>
      <td>06.323.688/0001-27</td>
      <td>IT NOW PIBB IBRX-50 FUNDO DE ÍNDICE RESPONSABI...</td>
      <td>202508</td>
      <td>9.333802e+08</td>
      <td>20.654323</td>
      <td>1.015659e+09</td>
      <td>50</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.118529</td>
      <td>...</td>
      <td>-0.619399</td>
      <td>0.066605</td>
      <td>0.498580</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>06.323.688/0001-27</td>
      <td>IT NOW PIBB IBRX-50 FUNDO DE ÍNDICE RESPONSABI...</td>
      <td>202509</td>
      <td>9.502398e+08</td>
      <td>20.672225</td>
      <td>9.649803e+08</td>
      <td>49</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.127355</td>
      <td>...</td>
      <td>-0.600803</td>
      <td>0.063425</td>
      <td>0.398001</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>06.323.688/0001-27</td>
      <td>IT NOW PIBB IBRX-50 FUNDO DE ÍNDICE RESPONSABI...</td>
      <td>202510</td>
      <td>9.650222e+08</td>
      <td>20.687662</td>
      <td>9.672142e+08</td>
      <td>49</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.127783</td>
      <td>...</td>
      <td>-0.603854</td>
      <td>0.073029</td>
      <td>0.511019</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python

```
