#!/usr/bin/env bash
set -euo pipefail

# Entrypoint wrapper for the container.
# Usage:
#  - Run the CLI normally: docker run --rm <image> fif --help
#  - Run the pipeline: docker run --rm <image> pipeline /manifest.yaml /data [YYYY-MM-DD]

if [ "${1:-}" = "pipeline" ]; then
  MANIFEST="${2:-/manifest.yaml}"
  OUTDIR="${3:-/data}"
  REF_DATE="${4:-$(date -I)}"

  echo "Starting pipeline"
  echo " manifest: $MANIFEST"
  echo " outdir:   $OUTDIR"
  echo " ref_date: $REF_DATE"

  if [ ! -f "$MANIFEST" ]; then
    echo "[error] Manifest file not found: $MANIFEST" >&2
    exit 2
  fi

  # 1) Fetch
  echo "--> fetching datasets"
  poetry run fif data fetch --manifest "$MANIFEST" --output-dir "$OUTDIR"

  # 2) Build features
  echo "--> computing features"
  poetry run fif feature build --input-dir "$OUTDIR" --config "$MANIFEST" --output "$OUTDIR/features.parquet"

  # Determine features input (parquet preferred, csv fallback)
  if [ -f "$OUTDIR/features.parquet" ]; then
    FEATURES_IN="$OUTDIR/features.parquet"
  elif [ -f "$OUTDIR/features.csv" ]; then
    FEATURES_IN="$OUTDIR/features.csv"
  else
    echo "[error] Features output not found (neither parquet nor csv): $OUTDIR/features.parquet|$OUTDIR/features.csv" >&2
    exit 2
  fi

  # 3) Compute scores
  echo "--> computing scores using input: $FEATURES_IN"
  poetry run fif model score --input "$FEATURES_IN" --config "$MANIFEST" --output "$OUTDIR/features_scored.parquet"

  # Determine scored features input (parquet preferred, csv fallback)
  if [ -f "$OUTDIR/features_scored.parquet" ]; then
    SCORED_IN="$OUTDIR/features_scored.parquet"
  elif [ -f "$OUTDIR/features_scored.csv" ]; then
    SCORED_IN="$OUTDIR/features_scored.csv"
  else
    echo "[error] Scored features output not found (neither parquet nor csv): $OUTDIR/features_scored.parquet|$OUTDIR/features_scored.csv" >&2
    exit 2
  fi

  # 4) Compute profile rankings
  echo "--> computing profile rankings using input: $SCORED_IN"
  poetry run fif policy profile-score --input "$SCORED_IN" --config "$MANIFEST" --output "$OUTDIR/features_profile_scored.parquet"

  # Finalize: if profile output was written as CSV, report that path
  if [ -f "$OUTDIR/features_profile_scored.parquet" ]; then
    echo "Pipeline finished successfully. Profile scores: $OUTDIR/features_profile_scored.parquet"
  elif [ -f "$OUTDIR/features_profile_scored.csv" ]; then
    echo "Pipeline finished successfully. Profile scores: $OUTDIR/features_profile_scored.csv"
  else
    echo "Pipeline finished but profile output not found." >&2
    exit 2
  fi
  echo "Pipeline finished successfully. Outputs written to: $OUTDIR"
  exit 0
fi

# default: pass-through to the CLI
exec poetry run fif "$@"
