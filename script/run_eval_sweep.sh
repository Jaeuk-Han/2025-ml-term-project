#!/usr/bin/env bash
# root에서 bash run_eval_sweep.sh로 실행

set -euo pipefail

K=10
OUT_JSONL="data/derived/rerank/alpha_sweep_k${K}.jsonl"
: > "$OUT_JSONL"

# 1) ALS 베이스라인(참고용)
poetry run python src/eval_topk_csv.py \
  --input data/derived/candidates/test_candidates.parquet \
  --splits_dir data/derived/als/splits --split test \
  --score_col als_score --k "$K"

# 2) α 스윕 평가(final_score 기준)
for a in 0.3 0.4 0.5 0.6 0.7; do
  poetry run python src/eval_topk_csv.py \
    --input data/derived/rerank/test_reranked_topk_a${a/./}.csv \
    --splits_dir data/derived/als/splits --split test \
    --score_col final_score --k "$K" \
  | A=$a python -c "import sys,os,json; r=json.load(sys.stdin); r['alpha']=float(os.environ['A']); print(json.dumps(r, ensure_ascii=False))" \
  >> "$OUT_JSONL"
done

# 3) 정렬 출력
python - <<'PY'
import json, pandas as pd
rows=[json.loads(l) for l in open("data/derived/rerank/alpha_sweep_k10.jsonl", encoding="utf-8")]
df=pd.DataFrame(rows)
print(df.sort_values("ndcg@k", ascending=False))
PY
