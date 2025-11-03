import argparse, json, os, pickle
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

# Note: 학습된 LGBM 리랭커로 후보 점수를 예측하고 ALS 점수와 블렌딩하여 최종 Top-K 추천을 생성합니다.
# 기대 입력: candidates(ALS), features_dir(item_features.csv, user_profiles.csv, features_meta.json), 모델 pkl
# 출력: user_id별 Top-K 추천 CSV

# dir util
def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# multi reader for parquet or csv 
def read_any(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)


# numric value helper
def safe_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# meta data loader
def load_meta(features_dir: str) -> Dict:
    meta_path = os.path.join(features_dir, "features_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_vector_features(row: pd.Series, ucols: List[str], icols: List[str]) -> Tuple[float, float, float]:
    u = row[ucols].values.astype(float)
    v = row[icols].values.astype(float)
    un = np.linalg.norm(u); vn = np.linalg.norm(v)
    cos = float(np.dot(u, v) / (un * vn)) if (un > 0 and vn > 0) else 0.0
    l2 = float(np.linalg.norm(u - v))
    ad = float(np.mean(np.abs(u - v)))
    return cos, l2, ad

# metric for sanity check
def _precision_at_k(rel, k): k=min(k,len(rel)); return 0.0 if k<=0 else float(np.sum(rel[:k])/k)

def _ap_at_k(rel, k):
    k=min(k,len(rel)); hits=0; ap=0.0; pos=max(1,int(np.sum(rel)))
    for i in range(k):
        if rel[i]>0: hits+=1; ap+=hits/(i+1.0)
    return float(ap/pos)

def _ndcg_at_k(rel, k):
    k=min(k,len(rel)); gains=(2**rel[:k]-1); disc=1.0/np.log2(np.arange(2,k+2))
    dcg=float(np.sum(gains*disc)); ideal=np.sort(rel)[::-1][:k]
    idcg=float(np.sum((2**ideal-1)*disc)); return float(dcg/idcg) if idcg>0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="ALS candidates (parquet/csv) with [user_id, artist_id, als_score, als_rank]")
    ap.add_argument("--features_dir", required=True, help="dir containing item_features.csv, user_profiles.csv, features_meta.json")
    ap.add_argument("--model_pkl", required=True, help="trained reranker pickle (from train_reranker_lgbm.py)")
    ap.add_argument("--alpha", type=float, default=0.5, help="blend weight for ranker (0~1)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--out_path", required=True)

    # optional quick eval
    ap.add_argument("--splits_dir", default=None, help="for optional quick metrics")
    ap.add_argument("--split", choices=["train","val","test"], default=None)
    ap.add_argument("--k_eval", type=int, default=10)
    args = ap.parse_args()

    ensure_dir(args.out_path)

    # load
    cand = read_any(args.candidates)
    cand["user_id"]   = pd.to_numeric(cand["user_id"], errors="coerce").fillna(-1).astype(int)
    cand["artist_id"] = pd.to_numeric(cand["artist_id"], errors="coerce").fillna(-1).astype(int)
    safe_numeric(cand, ["als_score", "als_rank"])
    cand["artist_id_str"] = cand["artist_id"].astype(str)

    with open(args.model_pkl, "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]
    model_feats: List[str] = obj["features"]

    meta = load_meta(args.features_dir)
    
    item_features = pd.read_csv(os.path.join(args.features_dir, "item_features.csv"))

    user_profiles = pd.read_csv(os.path.join(args.features_dir, "user_profiles.csv"))
    user_profiles["user_id"] = pd.to_numeric(user_profiles["user_id"], errors="coerce").fillna(-1).astype(int)
    
    item_features["artist_id"] = item_features["artist_id"].astype(str)

    # join
    df = cand.merge(user_profiles, how="left", on="user_id")
    item_join = item_features.rename(columns={"artist_id":"artist_id_join"})
    df = df.merge(item_join, how="left", left_on="artist_id_str", right_on="artist_id_join")
    if "artist_id_join" in df.columns:
        df.drop(columns=["artist_id_join"], inplace=True)

    # rebuild derived features (as in make_rerank_dataset.py)
    # numeric pairs
    numeric_items = meta.get("numeric_item_feature_cols", [])
    uf_numeric = [f"uf_{c}" for c in numeric_items if f"uf_{c}" in df.columns]
    item_numeric = [c for c in numeric_items if c in df.columns]
    pair_uf, pair_it = [], []
    for c in item_numeric:
        ufc = f"uf_{c}"
        if ufc in df.columns:
            pair_it.append(c); pair_uf.append(ufc)
    safe_numeric(df, pair_uf + pair_it)

    if len(pair_uf) and len(pair_uf) == len(pair_it):
        cos_list, l2_list, adiff_list = [], [], []
        for _, row in df.iterrows():
            cs, l2, ad = build_vector_features(row, pair_uf, pair_it)
            cos_list.append(cs); l2_list.append(l2); adiff_list.append(ad)
        df["cos_user_item"] = cos_list
        df["l2_user_item"] = l2_list
        df["absdiff_mean_user_item"] = adiff_list
    else:
        df["cos_user_item"] = 0.0
        df["l2_user_item"] = 0.0
        df["absdiff_mean_user_item"] = 0.0

    # simple derived
    if "als_rank" in df.columns:
        df["als_rank_inv"] = 1.0 / (1.0 + pd.to_numeric(df["als_rank"], errors="coerce").fillna(0.0))
    if "item_pop_users" in df.columns:
        df["log_item_pop_users"] = np.log1p(pd.to_numeric(df["item_pop_users"], errors="coerce").fillna(0.0))
    if "item_pop_plays" in df.columns:
        df["log_item_pop_plays"] = np.log1p(pd.to_numeric(df["item_pop_plays"], errors="coerce").fillna(0.0))

    # interactions
    for a, b, name in [
        ("als_score", "log_item_pop_plays", "als_x_pop"),
        ("als_score", "cos_user_item",      "als_x_cos"),
    ]:
        if a in df.columns and b in df.columns:
            df[name] = pd.to_numeric(df[a], errors="coerce").fillna(0.0) * \
                       pd.to_numeric(df[b], errors="coerce").fillna(0.0)

    # model inputs
    for c in model_feats:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # predict and blend
    pred = model.predict(df[model_feats], num_iteration=getattr(model, "best_iteration", None))
    df["ranker_score"] = pred.astype(float)
    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    df["final_score"] = alpha * df["ranker_score"] + (1.0 - alpha) * df["als_score"]

    # per-user Top-K
    outs = []
    for uid, g in df.groupby("user_id"):
        top = g.sort_values("final_score", ascending=False).head(args.topk)
        outs.append(top[["user_id","artist_id","final_score","ranker_score","als_score"]])
    out = pd.concat(outs, axis=0)
    out.to_csv(args.out_path, index=False)

    print(f"[ok] reranked -> {args.out_path}")
    print(f"users: {out['user_id'].nunique()} | rows: {len(out)} | topk: {args.topk} | alpha: {alpha}")

    # optional quick eval if labels available
    if args.splits_dir and args.split:
        lab = pd.read_csv(os.path.join(args.splits_dir, f"{args.split}.csv"), usecols=["user_id","artist_id"])
        lab["user_id"] = lab["user_id"].astype(int); lab["artist_id"]=lab["artist_id"].astype(int)
        label_set = set(map(tuple, lab[["user_id","artist_id"]].to_numpy()))
        out_eval = out.copy()
        out_eval["label"] = out_eval.apply(lambda r: 1 if (int(r["user_id"]),int(r["artist_id"])) in label_set else 0, axis=1)

        stats = []
        for _, g in out_eval.groupby("user_id"):
            rel = g.sort_values("final_score", ascending=False)["label"].astype(int).values
            stats.append([_precision_at_k(rel, args.k_eval), _ap_at_k(rel, args.k_eval), _ndcg_at_k(rel, args.k_eval)])
        
        arr = np.array(stats)
        
        print({
            "k": args.k_eval,
            "users": int(len(arr)),
            "precision@k": float(arr[:,0].mean() if len(arr) else 0.0),
            "map@k": float(arr[:,1].mean() if len(arr) else 0.0),
            "ndcg@k": float(arr[:,2].mean() if len(arr) else 0.0),
        })


if __name__ == "__main__":
    main()
