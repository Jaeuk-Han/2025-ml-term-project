import argparse, json, os
from pathlib import Path
from typing import Tuple, Set, Dict, List

import numpy as np
import pandas as pd

# Note: 이 코드는 ALS 학습 결과(als_model.npz)와 splits를 사용해 사용자별 Top-N 후보를 생성해 저장합니다.
# 점수는 이전과 동일하게 L2 정규화 후 코사인 유사도를 사용하며 이미 들은 음악은 제외합니다.


# dir util
def ensure_dir(p: str):
    Path(p).parent.mkdir(parents=True, exist_ok=True)

# model loader
def load_model_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    npz = np.load(path, allow_pickle=True)

    U = npz["user_factors_true"]  # [n_users, f]
    V = npz["item_factors_true"]  # [n_items, f]
    
    cfg = {}
    if "config" in npz.files:
        cfg = dict(npz["config"].item())  # stored as object
    
    return U, V, cfg

# l2 norm util
def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n

# topk util (sorted)
def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k-1)[:k]
    return idx[np.argsort(-scores[idx])]

# data split checker
def load_split_users(splits_dir: str, target_split: str) -> np.ndarray:
    target_split = target_split.lower()

    if target_split not in {"train", "val", "test", "all"}:
        raise ValueError("--target_split must be one of [train, val, test, all]")

    if target_split == "all":
        users = set()
        for name in ["train", "val", "test"]:
            p = os.path.join(splits_dir, f"{name}.csv")
            if os.path.exists(p):
                df = pd.read_csv(p, usecols=["user_id"])
                users.update(df["user_id"].astype(int).unique().tolist())
        return np.array(sorted(users), dtype=int)

    p = os.path.join(splits_dir, f"{target_split}.csv")
    
    if not os.path.exists(p):
        raise FileNotFoundError(f"split csv not found: {p}")
    
    df = pd.read_csv(p, usecols=["user_id"])

    return df["user_id"].astype(int).unique()

# seen table maker for exception
def build_seen_table(train_csv: str) -> Dict[int, Set[int]]:
    if not os.path.exists(train_csv):
        return {}
    
    tr = pd.read_csv(train_csv, usecols=["user_id", "artist_id"])
    tr["user_id"] = tr["user_id"].astype(int)
    tr["artist_id"] = tr["artist_id"].astype(int)
    
    seen = {}
    
    for u, g in tr.groupby("user_id"):
        seen[int(u)] = set(int(x) for x in g["artist_id"].tolist())
    
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_npz", required=True, help="data/derived/als/als_model.npz")
    ap.add_argument("--splits_dir", required=True, help="data/derived/als/splits")
    ap.add_argument("--target_split", default="val", choices=["train","val","test","all"])
    ap.add_argument("--topN", type=int, default=200)
    ap.add_argument("--out_path", required=True, help="where to save candidates (parquet or csv)")
    args = ap.parse_args()

    # load model
    U, V, cfg = load_model_npz(args.model_npz)
    cosine_score = bool(cfg.get("cosine_score", False))

    if cosine_score:
        U_ = l2_normalize_rows(U)
        V_ = l2_normalize_rows(V)
    else:
        U_, V_ = U, V

    # load users in data
    users = load_split_users(args.splits_dir, args.target_split)
    users = users[(users >= 0) & (users < U_.shape[0])]
    if users.size == 0:
        raise ValueError("No valid users found for target_split.")

    # seen table (train)
    train_csv = os.path.join(args.splits_dir, "train.csv")
    seen = build_seen_table(train_csv)

    rows_user, rows_item, rows_score, rows_rank = [], [], [], []

    # pre calcul item score
    Vt = V_.T  # [f, n_items]
    n_items = V_.shape[0]

    for u in users:
        s = U_[u].dot(Vt)  # [n_items]
        # remove seen item
        if u in seen:
            idx_seen = np.fromiter(seen[u], dtype=int)
            idx_seen = idx_seen[(idx_seen >= 0) & (idx_seen < n_items)]
            s[idx_seen] = -np.inf

        # get top n
        top_idx = topk_indices(s, args.topN)
        top_scores = s[top_idx]
        # ranking (start with 0)
        order = np.argsort(-top_scores)
        top_idx = top_idx[order]
        top_scores = top_scores[order]
        ranks = np.arange(len(top_idx))

        rows_user.append(np.full_like(top_idx, u))
        rows_item.append(top_idx.astype(int))
        rows_score.append(top_scores.astype(float))
        rows_rank.append(ranks.astype(int))

    # concatnation
    user_arr = np.concatenate(rows_user)
    item_arr = np.concatenate(rows_item)
    score_arr = np.concatenate(rows_score)
    rank_arr = np.concatenate(rows_rank)

    out = pd.DataFrame({
        "user_id": user_arr.astype(int),
        "artist_id": item_arr.astype(int),
        "als_score": score_arr,
        "als_rank": rank_arr,
    })

    ensure_dir(args.out_path)
    if args.out_path.lower().endswith(".parquet"):
        out.to_parquet(args.out_path, index=False)
    else:
        out.to_csv(args.out_path, index=False)

    print(f"[done] candidates saved -> {args.out_path}")
    print(f"  users: {len(users):,} | rows: {len(out):,} | topN per user: ~{args.topN}")


if __name__ == "__main__":
    main()
