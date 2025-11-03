#!/usr/bin/env bash
# root에서 bash run_train_reranker.sh로 실행

set -euo pipefail

poetry run python src/train_reranker_lgbm.py \
  --train data/derived/rerank/val_dataset.parquet \
  --features_meta data/derived/rerank/val_dataset.features.json \
  --eval  data/derived/rerank/test_dataset.parquet \
  --out_dir data/derived/rerank/lgbm \
  --k 10
