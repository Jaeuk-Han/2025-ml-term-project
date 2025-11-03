#!/usr/bin/env bash
# root에서 bash run_dump_candidates.sh로 실행
set -euo pipefail

TOPN="${1:-200}"

# VAL
poetry run python src/dump_candidates_from_als.py \
  --model_npz data/derived/als/als_model.npz \
  --splits_dir data/derived/als/splits \
  --target_split val \
  --topN "$TOPN" \
  --out_path data/derived/candidates/val_candidates.parquet

# TEST
poetry run python src/dump_candidates_from_als.py \
  --model_npz data/derived/als/als_model.npz \
  --splits_dir data/derived/als/splits \
  --target_split test \
  --topN "$TOPN" \
  --out_path data/derived/candidates/test_candidates.parquet
