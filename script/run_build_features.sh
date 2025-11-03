#!/usr/bin/env bash
# root에서 bash run_build_features.sh로 실행
set -euo pipefail

poetry run python src/build_user_item_features.py \
  --interactions_csv data/derived/als/splits/train.csv \
  --artist_features_csv data/derived/spotify_full/artist_features_weighted.csv \
  --out_dir data/derived/features
