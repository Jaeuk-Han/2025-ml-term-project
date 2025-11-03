#!/usr/bin/env bash
# root에서 bash run_make_rerank_dataset.sh로 실행

set -euo pipefail

# VAL
poetry run python src/make_rerank_dataset.py \
  --candidates  data/derived/candidates/val_candidates.parquet \
  --splits_dir  data/derived/als/splits \
  --features_dir data/derived/features \
  --target_split val \
  --out_path data/derived/rerank/val_dataset.parquet

# TEST
poetry run python src/make_rerank_dataset.py \
  --candidates  data/derived/candidates/test_candidates.parquet \
  --splits_dir  data/derived/als/splits \
  --features_dir data/derived/features \
  --target_split test \
  --out_path data/derived/rerank/test_dataset.parquet
