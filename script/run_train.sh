#!/usr/bin/env bash
# root에서 bash run_train.sh로 실행

poetry run python src/train_implicit_als.py \
  --interactions_csv data/derived/lastfm_join_strict/joined.csv \
  --out_dir data/derived/als_bm25_cos_pop01 \
  --alpha 40 --factors 128 --reg 0.005 --iterations 40 \
  --val_ratio 0.10 --test_ratio 0.10 --seed 42 \
  --log1p_plays --min_user_interactions 2 --min_item_interactions 2 \
  --weighting bm25 --cosine_score --pop_blend 0.1