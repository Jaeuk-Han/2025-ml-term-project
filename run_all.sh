#!/usr/bin/env bash
# root에서 bash run_all.sh로 실행

set -euo pipefail

# ========= User Config (필요시만 변경) =========
TOPN="${TOPN:-200}"                 # ALS 후보 개수
ALPHAS="${ALPHAS:-0.3 0.4 0.5 0.6 0.7}"  # 블렌딩 가중치 스윕
K="${K:-10}"                        # 평가 K
SEED="${SEED:-42}"                  # ALS seed
# ==============================================

# 경로 상수
DATA_DIR="data"
DERIVED="${DATA_DIR}/derived"
SPOTIFY_FULL="${DERIVED}/spotify_full"
JOIN_DIR="${DERIVED}/lastfm_join_strict"
ALS_DIR="${DERIVED}/als"
SPLITS="${ALS_DIR}/splits"
CANDS="${DERIVED}/candidates"
FEATS="${DERIVED}/features"
RERANK="${DERIVED}/rerank"

# 공용 입력
ARTISTS_CSV="${DATA_DIR}/spotify/artists.csv"
TRACKS_CSV="${DATA_DIR}/spotify/tracks.csv"
LASTFM_CSV="${DATA_DIR}/lastfm/Last.fm_data.csv"
ARTIST_FEATS="${SPOTIFY_FULL}/artist_features_weighted.csv"

# 유틸
ts() { date "+%Y-%m-%d %H:%M:%S"; }
banner() { echo -e "\n========== [$1] $(ts) ==========\n"; }
need() { [[ -f "$1" ]] || { echo "!! missing: $1"; exit 1; }; }

main() {
  mkdir -p "$SPOTIFY_FULL" "$JOIN_DIR" "$ALS_DIR" "$SPLITS" "$CANDS" "$FEATS" "$RERANK"

  banner "1) Spotify → 아티스트 벡터"
  need "$ARTISTS_CSV"; need "$TRACKS_CSV"; need "$LASTFM_CSV"
  poetry run python src/build_artist_vectors.py \
    --artists_csv "$ARTISTS_CSV" \
    --tracks_csv  "$TRACKS_CSV" \
    --lastfm_csv  "$LASTFM_CSV" \
    --out_features_csv   "$ARTIST_FEATS" \
    --out_per_genre_csv  "${SPOTIFY_FULL}/artist_genre_aggregates.csv" \
    --out_coverage_json  "${SPOTIFY_FULL}/coverage.json"

  banner "2) Last.fm × Spotify 조인"
  poetry run python src/join_lastfm_spotify.py \
    --lastfm_csv "$LASTFM_CSV" \
    --spotify_features_csv "$ARTIST_FEATS" \
    --out_dir "$JOIN_DIR" \
    --min_plays 1 --normalize basic
  need "${JOIN_DIR}/joined.csv"

  banner "3) ALS 학습/분할/매핑"
  poetry run python src/train_implicit_als.py \
  --interactions_csv "${JOIN_DIR}/joined.csv" \
  --out_dir "$ALS_DIR" \
  --factors 64 --reg 0.05 --iterations 20 --alpha 10 \
  --val_ratio 0.10 --test_ratio 0.10 --seed "$SEED" \
  --log1p_plays \
  --min_user_interactions 2 --min_item_interactions 2 \
  --weighting bm25 \
  --cosine_score \
  --pop_blend 0.10


  need "${ALS_DIR}/als_model.npz"
  need "${SPLITS}/train.csv"; need "${SPLITS}/val.csv"; need "${SPLITS}/test.csv"

  banner "4) ALS 후보 TopN 덤프 (VAL/TEST)"
  poetry run python src/dump_candidates_from_als.py \
    --model_npz "${ALS_DIR}/als_model.npz" \
    --splits_dir "$SPLITS" \
    --target_split val \
    --topN "$TOPN" \
    --out_path "${CANDS}/val_candidates.parquet"

  poetry run python src/dump_candidates_from_als.py \
    --model_npz "${ALS_DIR}/als_model.npz" \
    --splits_dir "$SPLITS" \
    --target_split test \
    --topN "$TOPN" \
    --out_path "${CANDS}/test_candidates.parquet"

  banner "5) 리랭커용 유저/아이템 피처 생성"
  poetry run python src/build_user_item_features.py \
    --interactions_csv "${SPLITS}/train.csv" \
    --artist_features_csv "$ARTIST_FEATS" \
    --out_dir "$FEATS"
  need "${FEATS}/item_features.csv"; need "${FEATS}/user_profiles.csv"; need "${FEATS}/features_meta.json"

  banner "6) 리랭커 학습 테이블 생성 (VAL/TEST)"
  poetry run python src/make_rerank_dataset.py \
    --candidates  "${CANDS}/val_candidates.parquet" \
    --splits_dir  "$SPLITS" \
    --features_dir "$FEATS" \
    --target_split val \
    --out_path "${RERANK}/val_dataset.parquet"

  poetry run python src/make_rerank_dataset.py \
    --candidates  "${CANDS}/test_candidates.parquet" \
    --splits_dir  "$SPLITS" \
    --features_dir "$FEATS" \
    --target_split test \
    --out_path "${RERANK}/test_dataset.parquet"

  banner "7) LGBM 리랭커 학습/리포트"
  poetry run python src/train_reranker_lgbm.py \
    --train "${RERANK}/val_dataset.parquet" \
    --features_meta "${RERANK}/val_dataset.features.json" \
    --eval  "${RERANK}/test_dataset.parquet" \
    --out_dir "${RERANK}/lgbm" \
    --k "$K"
  need "${RERANK}/lgbm/lgbm_reranker.pkl"

  banner "8) 리랭크+블렌딩 Top-K 생성 (α 스윕)"
  for a in $ALPHAS; do
    out="${RERANK}/test_reranked_topk_a${a/./}.csv"
    poetry run python src/rerank_infer.py \
      --candidates "${CANDS}/test_candidates.parquet" \
      --features_dir "$FEATS" \
      --model_pkl   "${RERANK}/lgbm/lgbm_reranker.pkl" \
      --alpha "$a" --topk "$K" \
      --out_path "$out"
    echo "[ok] $out"
  done

  banner "9) α 스윕 평가(JSONL 집계/정렬)"
  sweep_jsonl="${RERANK}/alpha_sweep_k${K}.jsonl"
  : > "$sweep_jsonl"
  for a in $ALPHAS; do
    poetry run python src/eval_topk_csv.py \
      --input "${RERANK}/test_reranked_topk_a${a/./}.csv" \
      --splits_dir "$SPLITS" --split test \
      --score_col final_score --k "$K" \
    | A="$a" python -c "import sys,os,json; r=json.load(sys.stdin); r['alpha']=float(os.environ['A']); print(json.dumps(r, ensure_ascii=False))" \
    >> "$sweep_jsonl"
  done

  echo -e "\n[α sweep results] $sweep_jsonl"
  python - <<'PY'
import json, pandas as pd
p="data/derived/rerank/alpha_sweep_k10.jsonl"
try:
    rows=[json.loads(l) for l in open(p, encoding="utf-8")]
    df=pd.DataFrame(rows)
    print(df.sort_values("ndcg@k", ascending=False))
except FileNotFoundError:
    print("alpha sweep file not found:", p)
PY

  banner "[DONE]"
}

main "$@"
