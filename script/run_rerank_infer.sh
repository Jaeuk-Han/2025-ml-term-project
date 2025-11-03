#!/usr/bin/env bash
# root에서 bash run_rerank_infer.sh로 실행

set -euo pipefail

ALPHAS=("0.3" "0.4" "0.5" "0.6" "0.7")
TOPK=10

for a in "${ALPHAS[@]}"; do
  out="data/derived/rerank/test_reranked_topk_a${a/./}.csv"
  poetry run python src/rerank_infer.py \
    --candidates data/derived/candidates/test_candidates.parquet \
    --features_dir data/derived/features \
    --model_pkl   data/derived/rerank/lgbm/lgbm_reranker.pkl \
    --alpha "$a" --topk "$TOPK" \
    --out_path "$out"
  echo "[ok] $out"
done
