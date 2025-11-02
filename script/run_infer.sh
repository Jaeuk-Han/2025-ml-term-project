#!/usr/bin/env bash
# root에서 bash run_infer.sh로 실행

poetry run python src/predict_topk.py \
  --model data/derived/als_bm25_cos_pop01/als_model.npz \
  --user_mapping data/derived/als_bm25_cos_pop01/user_mapping.csv \
  --artist_mapping data/derived/als_bm25_cos_pop01/artist_mapping.csv \
  --k 10 \
  --user_id 0 \
  --cosine_score \
  --pop_blend 0.1 \
  --pop_from_csv data/derived/als_bm25_cos_pop01/splits/train.csv \
  --interactions_csv data/derived/als_bm25_cos_pop01/splits/train.csv \
  --out_json data/derived/als_bm25_cos_pop01/preds/user0_top10.json \
  --out_csv  data/derived/als_bm25_cos_pop01/preds/user0_top10.csv


# 이름기준
# poetry run python src/predict_topk.py \
#   --model data/derived/als_bm25_cos_pop01/als_model.npz \
#   --user_mapping data/derived/als_bm25_cos_pop01/user_mapping.csv \
#   --artist_mapping data/derived/als_bm25_cos_pop01/artist_mapping.csv \
#   --k 10 \
#   --username "<USERNAME_HERE>" \
#   --cosine_score \
#   --pop_blend 0.1 \
#   --pop_from_csv data/derived/als_bm25_cos_pop01/splits/train.csv \
#   --interactions_csv data/derived/als_bm25_cos_pop01/splits/train.csv \
#   --out_json data/derived/als_bm25_cos_pop01/preds/<USERNAME_HERE>_top10.json \
#   --out_csv  data/derived/als_bm25_cos_pop01/preds/<USERNAME_HERE>_top10.csv
