# fif_recsys

Ranking Brazilian fixed income funds

This CLI tool provides configuration management capabilities with a clean, modern interface built using Typer and Rich.

## Features

- **Configuration Management**: Set, get, list, and reset configuration values
- **Type-aware Values**: Automatic conversion of boolean, numeric, and string values
- **Rich Formatting**: Beautiful table output and colorized messages
- **Interactive Prompts**: Set values interactively when not provided
- **Environment Support**: Custom config file location via `ACLI_CONFIG_PATH`
- **Comprehensive Testing**: Full test suite with pytest
- **Professional Documentation**: Auto-generated docs with MkDocs

## Important: Poetry Version

To avoid compatibility errors (such as TypeError related to canonicalize_version), ensure you are using an up-to-date version of Poetry:

   ```bash
   pip install --upgrade poetry
   ```

If you encounter installation issues, upgrading Poetry usually resolves them.

## Quick Start

1. **Installation**: Install the package in development mode
   ```bash
   cd fif_recsys
   make install
   ```

2. **Basic Usage**: Try the configuration commands
   ```bash
   fif config list
   fif config set theme dark
   fif config get theme
   ```

## CLI Commands

### Configuration Management

- **`fif config set KEY VALUE`** - Set a configuration value
- **`fif config get KEY`** - Get a configuration value  
- **`fif config list`** - List all configuration values
- **`fif config reset`** - Reset to default values

### Examples

```bash
# View current configuration
fif config list

# Set values directly
fif config set theme dark
fif config set debug true
fif config set timeout 30

# Set values interactively
fif config set custom_key
# > Enter value for 'custom_key': my_value

# Get specific values
fif config get theme
fif config get debug

# Reset to defaults (with confirmation)
fif config reset
```

### Data, feature and model commands 🔧

This project includes end-to-end commands to fetch raw datasets, compute fund-month features, and compute normalized scores driven by YAML configuration files.

#### Data commands (fetching and ingestion)

- `fif data fetch MANIFEST_YAML -d/--output-dir PATH [--ref-date YYYY-MM-DD]`
  - Fetch multiple datasets defined in a manifest YAML file.
  - The manifest describes `base_url`, `periods`, and `filename_template` for each dataset.
  - The command will download archives (ZIP), extract CSVs, concatenate by dataset and period, and write partitioned dataset files to `output_dir/<dataset>/period=<period>/data.parquet`.
  - If a Parquet engine is not available (no `pyarrow`/`fastparquet`) the command will fall back to writing `data.csv` files instead.
  - The fetch adds a `reference_date` column (ISO date) to rows indicating when the fetch occurred.

Programmatic helper: `fif_recsys.commands.data.fetch_manifest(manifest_dict, output_dir, reference_date=None)` returns a `dict[str, pandas.DataFrame]` mapping dataset names to DataFrames for further programmatic processing.

Example manifest snippet:

```yaml
fetch:
  cda:
    base_url: "https://dados.cvm.gov.br/dados/FI/DOC/CDA/DADOS/"
    periods: ["202501", "202502"]
    filename_template: "cda_fi_{period}.zip"
```

#### Feature commands (feature engineering) 🔧

- `fif feature build -i INPUT_DIR -d DATASET -o OUTPUT_PATH`
  - Loads partitioned datasets from `INPUT_DIR` (supports Parquet partitions and CSV fallbacks).
  - Computes fund-month features according to a feature registry (defined in YAML or programmatically) using the project's `FEATURE_ENGINE`.
  - Writes the feature table to `OUTPUT_PATH` (Parquet preferred, CSV fallback if Parquet engine missing).

Programmatic helper: `compute_all_features(data_sources_d, config_d, FEATURE_ENGINE)` returns a DataFrame with computed features.

Feature registry YAML example (simplified):

```yaml
feature:
  group_keys:
    - CNPJ_FUNDO_CLASSE
    - DENOM_SOCIAL
    - competencia
  feature_registry:
    cda:
      patrimonio_liq:
        description: "Maximum reported net asset value per fund-month."
        method: max
        args:
          - VL_PATRIM_LIQ

      log_aum:
        description: "Log-transformed AUM (for size comparisons)."
        method: max
        args:
          - VL_PATRIM_LIQ
        adjustment:
          - log

      credito_share:
        description: "Weighted share of credit-linked assets in the portfolio."
        method: credito_share_feature_fn
        args:
          - ["Debêntures", "Cédula de Crédito", "CRI", "CRA", "Notas Promissórias"]
        adjustment:
          - clip
```

Notes:
- Methods can be built-in aggregations (e.g., `sum`, `max`, `nunique`) or custom feature functions (e.g., `credito_share_feature_fn`, `hhi_feature_fn`).
- Adjustments (e.g., `log`, `clip`, `coalesce`) are applied after aggregation to normalize or clean values for scoring.
- Ensure the `group_keys` reflect how you want to aggregate fund-month rows.

#### Model commands (scoring) 🎯

- `fif model score -i INPUT_PATH -c CONFIG_YAML -o OUTPUT_PATH`
  - Reads the feature table and scoring YAML config.
  - Computes normalized scores (currently `zscore`) and applies adjustments such as `invert` or `coalesce`.
  - Appends score columns (e.g., `size_score`) to the table and writes the scored table to `OUTPUT_PATH`.

Programmatic helper: `compute_scores_from_yaml(features_df, config_d)` returns a DataFrame with added score columns.

Scoring YAML example (simplified):

```yaml
score:
  size_score:
    type: zscore
    description: "Z-score of `log_aum` to capture fund size (bigger → better)."
    args:
      feature: log_aum

  credit_risk_score:
    type: zscore
    description: "Credit exposure inverted (higher credit → lower score)."
    args:
      feature: credito_share
    adjustment:
      - invert
```

Notes:
- `type` currently supports `zscore` (standardized values). Additional score types can be added as required.
- `args.feature` points to the feature column to be scored (e.g., `log_aum`, `n_ativos`).
- `adjustment` entries are applied after computing the raw score (e.g., `invert` flips the sign so higher raw means lower score).
- After scoring, you can compute profile-level aggregations with `fif policy profile-score` (see below).

#### Policy commands (profile scoring & ranking)

- `fif policy profile-score -i INPUT_PATH -c CONFIG_YAML -o OUTPUT_PATH`
  - Loads a scored feature table and a YAML file containing profile definitions (either top-level `profile:` or `profiles:`).
  - Computes weighted profile scores by summing feature score columns multiplied by weights defined in each profile.
  - Appends `score_<profile>` and `rank_<profile>` columns (dense ranking: best score → rank 1) to the table and writes the result to `OUTPUT_PATH` (Parquet preferred, CSV fallback available).

Programmatic helpers: `compute_profile_scores_from_yaml(features_df, config_d)` and `compute_profile_scores_from_df(features_df, profiles)` are available for programmatic usage in notebooks and scripts.

Example profile YAML snippet:

```yaml
profile:
  conservative:
    size_score: 0.25
    diversification_score: 0.20
  balanced:
    size_score: 0.20
    diversification_score: 0.15
```
#### Notes & tips 💡

- Dependencies: `pandas` is required; `pyarrow` or `fastparquet` are optional but recommended for efficient Parquet IO.
- Tests: The test suite uses CSV fallbacks to avoid requiring Parquet dependencies in CI.
- Reproducibility: Use `--ref-date` (or pass `reference_date` programmatically) to produce deterministic fetch outputs.

## Development

### Prerequisites

This project uses [Poetry](https://python-poetry.org/) for dependency management. Install Poetry first:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Setup

1. **Install dependencies**:
   ```bash
   make install
   ```
   This installs the package and all development dependencies using Poetry.

2. **Install pre-commit hooks**:
   ```bash
   make pre-commit
   ```

### Testing

Run the comprehensive test suite:

```bash
make test
```

Or run tests directly with Poetry:

```bash
poetry run pytest -vvv
```

Tests cover:
- All config subcommands (set, get, list, reset)
- Type conversion (boolean, numeric, string)
- Error handling (missing keys, corrupted files)
- Interactive prompts and confirmations
- Environment variable configuration
- File creation and management

### Documentation

1. **Install docs dependencies**:
   ```bash
   make docs
   ```

2. **Serve docs locally**:
   ```bash
   make serve-docs
   ```
   Or run directly with Poetry:
   ```bash
   poetry run mkdocs serve -f docs/mkdocs.yml
   ```

3. **View documentation**: Open http://localhost:8000

### Code Quality

- **Format code**: `make format` or `poetry run black .`
- **Check formatting**: `make check` or `poetry run black --check --diff .`
- **Run linting**: `poetry run flake8`
- **Type checking**: `poetry run mypy .`
- **Clean artifacts**: `make clean`

### Docker Testing

Test the CLI in a clean container environment:

1. **Build image**:
   ```bash
   make docker-image
   ```

2. **Run commands** (example: show help or run specific CLI command):
   ```bash
   docker run --rm acli --help
   docker run --rm acli fif --help
   ```

3. **Run the full pipeline using a manifest** (fetch → feature → score → profile ranking):

   Mount your `manifest.yaml` and an output directory, and run the `pipeline` entrypoint. For example:

   ```bash
   mkdir -p /tmp/fif_data
   docker run --rm \
     -v "$(pwd)/manifest.yaml:/manifest.yaml" \
     -v "/tmp/fif_data:/data" \
     acli pipeline /manifest.yaml /data
   ```

   The container will execute the following steps in order:
   - `fif data fetch --manifest /manifest.yaml --output-dir /data`
   - `fif feature build --input-dir /data --config /manifest.yaml --output /data/features.parquet`
   - `fif model score --input /data/features.parquet --config /manifest.yaml --output /data/features_scored.parquet`
   - `fif policy profile-score --input /data/features_scored.parquet --config /manifest.yaml --output /data/features_profile_scored.parquet`

   Output files will be written into the mounted `/data` directory on the host.

   Note: If the container can't write Parquet because `pyarrow` isn't installed, CSV fallbacks will be written instead (e.g., `features.csv`).

## Configuration Storage

- **Default location**: `~/.acli_config.json`
- **Custom location**: Set `ACLI_CONFIG_PATH` environment variable
- **Format**: JSON with automatic type preservation
- **Default values**: Includes theme, output_format, auto_save, and debug settings

## Distribution

### PyPI Publishing

> **NOTE**: Ensure you have a [PyPI account](https://pypi.org/account/register/) before publishing.

1. **Create distributions**:
   ```bash
   make distributions
   ```
   This builds the package using Poetry.

2. **Upload to PyPI**:
   ```bash
   poetry publish
   ```
   Or use twine:
   ```bash
   twine upload dist/*
   ```

### Package Structure

The generated package includes:
- **Clean CLI interface** with professional help text
- **Comprehensive test coverage** for all functionality
- **Type-safe configuration handling** with automatic conversions
- **Rich formatting** for beautiful output
- **Professional documentation** ready for deployment
- **Docker support** for containerized usage

### Project layout (directory tree)

A concise view of the repository layout (truncated) to help you locate commands, modules, and tests:

```text
.
├── core.py                 # high-level package helpers and CLI entrypoints
├── Dockerfile              # container image build steps for testing/deployment
├── Makefile                # convenience commands (install, test, docs, etc.)
├── pyproject.toml          # project metadata and dependencies (Poetry)
├── README.md               # this file
├── docs/                   # MkDocs site and notebook resources
│   ├── mkdocs.yml          # docs configuration
│   └── docs/
│       └── notebooks/      # example notebooks and tutorials
├── notebooks/              # interactive notebooks (examples, experiments)
│   └── example.ipynb
├── etc/                    # auxiliary scripts and sample artifacts
│   ├── artifact.py
│   └── dump.py
├── fif_recsys/             # main package code
│   ├── __init__.py
│   ├── constants.py        # global constants and settings
│   ├── main.py             # top-level Typer app and command registration
│   ├── utils.py            # reusable helpers (I/O, parsing, small utilities)
│   └── commands/           # CLI command implementations (Typer)
│       ├── __init__.py
│       ├── config.py       # configuration management commands
│       ├── data.py         # data download & ingestion (fetch/manifest)
│       ├── feature.py      # feature engineering pipeline and registry
│       ├── model.py        # scoring logic and CLI commands
│       └── policy.py       # profile scoring and ranking commands
└── tests/                  # test suite (pytest)
    ├── test_config.py
    ├── test_feature.py
    ├── test_fetch.py
    ├── test_model.py
    └── test_policy.py
```

Use this tree as a quick reference: command implementations live in `fif_recsys/commands/*`, reusable logic in `fif_recsys/` top-level modules, and tests in `tests/`.

## Architecture

Built with modern Python CLI best practices:

- **[Poetry](https://python-poetry.org/)** - Modern dependency management
- **[Typer](https://typer.tiangolo.com/)** - Type-based CLI framework
- **[Rich](https://rich.readthedocs.io/)** - Beautiful terminal output
- **[Pytest](https://pytest.org/)** - Reliable testing framework
- **[MkDocs](https://mkdocs.org/)** - Professional documentation
- **[Black](https://black.readthedocs.io/)** - Code formatting
- **[Pre-commit](https://pre-commit.com/)** - Git hooks for quality

## Help

View all available make commands:

```bash
make help
```

Get CLI help:

```bash
fif --help
fif config --help
```
