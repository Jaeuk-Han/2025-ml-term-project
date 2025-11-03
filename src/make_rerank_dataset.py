import argparse, json, os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Note: 이 코드는 후보들의 아이템/유저 사이드피처를 합쳐 리랭킹 학습 세트를 만드는 코드입니다.
# target split(val/test/train)의 (user_id, artist_id) 존재 여부(1/0)가 학습을 위한 라벨이 됩니다.

# dir helper
def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)


# multi reader for parquet or csv 
def read_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)

# meta data loader
def load_meta(features_dir: str) -> Dict:
    meta_path = os.path.join(features_dir, "features_meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)

# label loader
def load_label_set(splits_dir: str, target_split: str) -> set:
    p = os.path.join(splits_dir, f"{target_split}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"split csv not found: {p}")
    df = pd.read_csv(p, usecols=["user_id", "artist_id"])
    df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce").fillna(-1).astype(int)
    df["artist_id"] = pd.to_numeric(df["artist_id"], errors="coerce").fillna(-1).astype(int)
    return set(zip(df["user_id"].tolist(), df["artist_id"].tolist()))

# numeric null filler
def safe_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# vector feature builder for labeling
def build_vector_features(row: pd.Series, ucols: List[str], icols: List[str]) -> Tuple[float, float, float]:
    u = row[ucols].values.astype(float)
    v = row[icols].values.astype(float)

    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)

    cos = float(np.dot(u, v) / (un * vn)) if (un > 0 and vn > 0) else 0.0
    l2 = float(np.linalg.norm(u - v))
    adiff = float(np.mean(np.abs(u - v)))

    return cos, l2, adiff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="parquet/csv with [user_id, artist_id, als_score, als_rank]")
    ap.add_argument("--splits_dir", required=True, help="data/derived/als/splits")
    ap.add_argument("--features_dir", required=True, help="data/derived/features (item_features.csv, user_profiles.csv, features_meta.json)")
    ap.add_argument("--target_split", required=True, choices=["train","val","test"])
    ap.add_argument("--out_path", required=True, help="output parquet/csv for reranker dataset")
    args = ap.parse_args()

    ensure_dir(args.out_path)

    # data load
    cand = read_any(args.candidates)
    cand["user_id"]  = pd.to_numeric(cand["user_id"],  errors="coerce").fillna(-1).astype(int)
    cand["artist_id"] = pd.to_numeric(cand["artist_id"], errors="coerce").fillna(-1).astype(int)
    safe_numeric(cand, ["als_score", "als_rank"])

    meta = load_meta(args.features_dir)
    item_features = pd.read_csv(os.path.join(args.features_dir, "item_features.csv"))
    user_profiles = pd.read_csv(os.path.join(args.features_dir, "user_profiles.csv"))

    # key allign
    user_profiles["user_id"] = pd.to_numeric(user_profiles["user_id"], errors="coerce").fillna(-1).astype(int)
    item_features["artist_id"] = item_features["artist_id"].astype(str)
    cand["artist_id_str"] = cand["artist_id"].astype(str)

    # load label
    label_pos = load_label_set(args.splits_dir, args.target_split)

    # join user activity (how they played)
    train_csv = os.path.join(args.splits_dir, "train.csv")
    if os.path.exists(train_csv):
        tr = pd.read_csv(train_csv, usecols=["user_id", "artist_id", "plays"])
        tr["user_id"]   = pd.to_numeric(tr["user_id"],   errors="coerce").fillna(-1).astype(int)
        tr["artist_id"] = pd.to_numeric(tr["artist_id"], errors="coerce").fillna(-1).astype(int)
        tr["plays"]     = pd.to_numeric(tr["plays"],     errors="coerce").fillna(0.0)

        agg = tr.groupby("user_id").agg(
            user_seen_items=("artist_id", "nunique"),
            user_total_plays=("plays", "sum"),
            user_mean_plays=("plays", "mean"),
        ).reset_index()
        user_profiles = user_profiles.merge(agg, on="user_id", how="left")
        safe_numeric(user_profiles, ["user_seen_items","user_total_plays","user_mean_plays"])
    else:
        for c in ["user_seen_items","user_total_plays","user_mean_plays"]:
            user_profiles[c] = 0.0

    # merging
    df = cand.merge(user_profiles, how="left", on="user_id")
    item_join = item_features.rename(columns={"artist_id": "artist_id_join"}).copy()
    df = df.merge(item_join, how="left", left_on="artist_id_str", right_on="artist_id_join")
    df.drop(columns=["artist_id_join"], inplace=True)

    # safe casting for feature
    base_keep = ["user_id", "artist_id", "als_score", "als_rank",
                 "item_pop_users", "item_pop_plays",
                 "user_seen_items","user_total_plays","user_mean_plays"]
    for c in base_keep:
        if c in df.columns and c not in ["user_id","artist_id"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # vector feature (uf_numeric ↔ item_numeric)
    numeric_items = meta.get("numeric_item_feature_cols", [])
    uf_numeric = [f"uf_{c}" for c in numeric_items if f"uf_{c}" in df.columns]
    item_numeric = [c for c in numeric_items if c in df.columns]

    pair_uf, pair_it = [], []
    for c in item_numeric:
        ufc = f"uf_{c}"
        if ufc in df.columns:
            pair_it.append(c)
            pair_uf.append(ufc)

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

    # categoric user feature
    extra_user_feats = []
    for c in ["uf_explicit"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            extra_user_feats.append(c)

    # make some feature for reranker
    if "als_rank" in df.columns:
        df["als_rank_inv"] = 1.0 / (1.0 + pd.to_numeric(df["als_rank"], errors="coerce").fillna(0.0))
    if "item_pop_users" in df.columns:
        df["log_item_pop_users"] = np.log1p(pd.to_numeric(df["item_pop_users"], errors="coerce").fillna(0.0))
    if "item_pop_plays" in df.columns:
        df["log_item_pop_plays"] = np.log1p(pd.to_numeric(df["item_pop_plays"], errors="coerce").fillna(0.0))

    # interaction feature
    for a, b, name in [
        ("als_score", "log_item_pop_plays", "als_x_pop"),
        ("als_score", "cos_user_item",      "als_x_cos"),
    ]:
        if a in df.columns and b in df.columns:
            df[name] = pd.to_numeric(df[a], errors="coerce").fillna(0.0) * \
                       pd.to_numeric(df[b], errors="coerce").fillna(0.0)

    # make label
    df["label"] = df.apply(lambda r: 1 if (int(r["user_id"]), int(r["artist_id"])) in label_pos else 0, axis=1)

    # feature list
    feature_cols = []
    feature_cols += ["als_score", "als_rank", "als_rank_inv"]
    for c in ["item_pop_users", "item_pop_plays", "log_item_pop_users", "log_item_pop_plays"]:
        if c in df.columns: feature_cols.append(c)
    feature_cols += ["cos_user_item", "l2_user_item", "absdiff_mean_user_item"]
    feature_cols += ["als_x_pop", "als_x_cos"]
    feature_cols += extra_user_feats
    for c in ["user_seen_items","user_total_plays","user_mean_plays"]:
        if c in df.columns: feature_cols.append(c)

    # save cols set
    out_cols = ["user_id", "artist_id", "label"] + feature_cols
    out_df = df[out_cols].copy()

    # save
    if args.out_path.lower().endswith(".parquet"):
        out_df.to_parquet(args.out_path, index=False)
    else:
        out_df.to_csv(args.out_path, index=False)

    # save feature meta data
    meta_out = os.path.splitext(args.out_path)[0] + ".features.json"
    ensure_dir(meta_out)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump({
            "features": feature_cols,
            "group_by": "user_id",
            "label": "label",
            "notes": {
                "pairs_used": {"user": pair_uf, "item": pair_it},
                "source_candidates": args.candidates,
                "target_split": args.target_split
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"[ok] rerank dataset -> {args.out_path}")
    print(f"  rows: {len(out_df):,} | positives: {int(out_df['label'].sum()):,} | features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
