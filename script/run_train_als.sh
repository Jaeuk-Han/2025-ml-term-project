#!/usr/bin/env bash
# root에서 bash run_train_als.sh로 실행

set -euo pipefail

poetry run python src/train_implicit_als.py \
  --interactions_csv data/derived/lastfm_join_strict/joined.csv \
  --out_dir data/derived/als \
  --alpha 40 --factors 64 --reg 0.01 --iterations 20 \
  --val_ratio 0.10 --test_ratio 0.10 --seed 42 \
  --log1p_plays --min_user_interactions 2 --min_item_interactions 2 \
  --weighting bm25 --cosine_score --pop_blend 0.1