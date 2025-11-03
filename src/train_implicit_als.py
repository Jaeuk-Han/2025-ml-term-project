import argparse, json, os
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight, tfidf_weight

# Note: als 모델을 학습시키는 역할을 수행하는 코드입니다.

# dir check util
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# csv schema reader util for various input format
def read_interactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    # case 1: already id based
    if all(x in cols for x in ["user_id","artist_id","plays"]):
        return df.rename(columns={
            cols["user_id"]:"user_id",
            cols["artist_id"]:"artist_id",
            cols["plays"]:"plays"
        })[["user_id","artist_id","plays"]]

    # case 2: Username/Artist + plays
    if all(x in cols for x in ["username","artist","plays"]):
        g = df.rename(columns={
            cols["username"]:"Username",
            cols["artist"]:"Artist",
            cols["plays"]:"plays"
        })[["Username","Artist","plays"]]
        g = g.groupby(["Username","Artist"], as_index=False)["plays"].sum()
        return g

    # case 3: 원본 로그형 Username/Artist
    if all(x in cols for x in ["username","artist"]):
        g = df.rename(columns={
            cols["username"]:"Username",
            cols["artist"]:"Artist"
        })
        g["plays"] = 1
        g = g.groupby(["Username","Artist"], as_index=False)["plays"].sum()
        return g

    raise ValueError(f"not support colunms: {list(df.columns)}")

# id builder for traing (ALS need id) support string key(Username/Artist) for continous key
def build_id_maps(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if "user_id" in df.columns and "artist_id" in df.columns:
        ui = df.copy()[["user_id","artist_id","plays"]]
        user_map = pd.DataFrame({"user_id": sorted(ui["user_id"].unique())})
        user_map["Username"] = user_map["user_id"].astype(str)
        art_map = pd.DataFrame({"artist_id": sorted(ui["artist_id"].unique())})
        art_map["Artist"] = art_map["artist_id"].astype(str)
        return ui, user_map, art_map

    users = sorted(df["Username"].astype(str).unique())
    arts  = sorted(df["Artist"].astype(str).unique())

    u2id = {u:i for i,u in enumerate(users)}
    a2id = {a:i for i,a in enumerate(arts)}

    ui = df.copy()
    ui["user_id"]   = ui["Username"].map(u2id)
    ui["artist_id"] = ui["Artist"].map(a2id)
    ui = ui[["user_id","artist_id","plays"]]

    return ui, \
        pd.DataFrame({"user_id": list(range(len(users))), "Username": users}), \
        pd.DataFrame({"artist_id": list(range(len(arts))),  "Artist": arts})

# min interaction filter (not in our data but for versatility, skip when 0)
def filter_min_interactions(df: pd.DataFrame, user_min=0, item_min=0) -> pd.DataFrame:
    if user_min <= 0 and item_min <= 0:
        return df
    changed = True
    cur = df.copy()
    while changed:
        changed = False
        if user_min > 0:
            du = cur.groupby("user_id")["artist_id"].count()
            keep_u = set(du[du >= user_min].index)
            before = len(cur)
            cur = cur[cur["user_id"].isin(keep_u)]
            changed = changed or (len(cur) < before)
        if item_min > 0:
            di = cur.groupby("artist_id")["user_id"].count()
            keep_i = set(di[di >= item_min].index)
            before = len(cur)
            cur = cur[cur["artist_id"].isin(keep_i)]
            changed = changed or (len(cur) < before)
    return cur.reset_index(drop=True)

# data split util based on interaction (train / val / test)
def split_train_val_test(ui: pd.DataFrame, val_ratio=0.1, test_ratio=0.1, seed=42):
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise ValueError(f"val_ratio+test_ratio must be < 1.0 (got {val_ratio}+{test_ratio})")
    
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ui))
    rng.shuffle(idx)
    n = len(idx)

    n_test = int(round(n * test_ratio))
    n_val  = int(round(n * val_ratio))
    n_train = n - n_val - n_test

    tr_idx = idx[:n_train]
    va_idx = idx[n_train:n_train+n_val]
    te_idx = idx[n_train+n_val:]

    tr = ui.iloc[tr_idx].reset_index(drop=True)
    va = ui.iloc[va_idx].reset_index(drop=True)
    te = ui.iloc[te_idx].reset_index(drop=True)

    return tr, va, te

# CSR(Compressed Sparse Row vector) maker funtion for als calculation
def to_csr(df: pd.DataFrame, n_users: int, n_items: int) -> csr_matrix:
    rows = df["user_id"].astype(int).values
    cols = df["artist_id"].astype(int).values
    data = df["plays"].astype(float).values
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

# alreadt seen set for filtering
def _user_pos_items(R: csr_matrix, u: int):
    start, end = R.indptr[u], R.indptr[u+1]
    return set(R.indices[start:end])

# topk with score
def _topk_from_scores(scores: np.ndarray, K: int):
    if K >= scores.size:
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, K)[:K]
        idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

# custom recommendation scoring function 
def recommend_for_user(
    u: int,
    U: np.ndarray,
    V: np.ndarray,
    R_train: csr_matrix,
    K: int,
    *,
    cosine_score: bool = False,
    pop_prior: np.ndarray | None = None,
    pop_blend: float = 0.0
):
    if cosine_score:
        # cos-sin for pre regularized U,V
        scores = U[u] @ V.T
    else:
        scores = U[u] @ V.T

    # popularity prior 0~1
    if pop_prior is not None and pop_blend > 0:
        scores = (1.0 - pop_blend) * scores + pop_blend * pop_prior

    # filter seen item
    for i in _user_pos_items(R_train, u):
        scores[i] = -np.inf

    ids, s = _topk_from_scores(scores, K)
    return ids.astype(int), s.astype(float)

# Precision function
# Note: 추천이 얼마나 정확했는지 확인
def precision_at_k(U, V, R_train, R_test, K: int, **rec_kwargs):
    Uc = U.shape[0]
    hit = 0; total = 0
    for u in range(Uc):
        ids, _ = recommend_for_user(u, U, V, R_train, K, **rec_kwargs)
        start, end = R_test.indptr[u], R_test.indptr[u+1]
        test_set = set(R_test.indices[start:end])
        if not test_set:
            continue
        hit += sum(1 for i in ids if i in test_set)
        total += K
    return (hit/total) if total else 0.0

# Mean Average Precision at K function
# Note: 정답을 맞출 때마다 그 시점의 Precision을 더해서 평균내는 것을 통해 정답을 위쪽에 배치할수록 점수가 커짐
def map_at_k(U, V, R_train, R_test, K: int, **rec_kwargs):
    Uc = U.shape[0]
    s = 0.0; n = 0
    for u in range(Uc):
        ids, _ = recommend_for_user(u, U, V, R_train, K, **rec_kwargs)
        start, end = R_test.indptr[u], R_test.indptr[u+1]
        test_set = set(R_test.indices[start:end])
        if not test_set:
            continue
        hits = 0; ap = 0.0
        for r, iid in enumerate(ids, 1):
            if iid in test_set:
                hits += 1
                ap += hits / r
        s += ap / min(K, len(test_set))
        n += 1
    return (s/n) if n else 0.0

# Normalized Discounted Cumulative Gain function for eval
# Note: 추천 순위의 상위에 맞출수록 더 큰 값
def ndcg_at_k(U, V, R_train, R_test, K: int, **rec_kwargs):
    Uc = U.shape[0]
    s = 0.0; n = 0
    log2 = np.log2
    for u in range(Uc):
        ids, _ = recommend_for_user(u, U, V, R_train, K, **rec_kwargs)
        start, end = R_test.indptr[u], R_test.indptr[u+1]
        test_set = set(R_test.indices[start:end])
        if not test_set:
            continue
        dcg = 0.0
        for r, iid in enumerate(ids, 1):
            if iid in test_set:
                dcg += 1.0 / log2(r+1)
        ideal_hits = min(K, len(test_set))
        idcg = sum(1.0/log2(r+1) for r in range(1, ideal_hits+1))
        s += (dcg/idcg) if idcg > 0 else 0.0
        n += 1
    return (s/n) if n else 0.0

# result saving function
def export_topk_csv(U, V, R_train, K: int, out_csv: str, **rec_kwargs):
    rows = []
    for u in range(R_train.shape[0]):
        ids, scores = recommend_for_user(u, U, V, R_train, K, **rec_kwargs)
        for rank, (iid, sc) in enumerate(zip(ids, scores), 1):
            rows.append((u, rank, int(iid), float(sc)))
    pd.DataFrame(rows, columns=["user_id","rank","artist_id","score"]).to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    # ALS
    ap.add_argument("--alpha", type=float, default=40.0)
    ap.add_argument("--factors", type=int, default=64)
    ap.add_argument("--reg", type=float, default=0.01)
    ap.add_argument("--iterations", type=int, default=20)
    # Splits
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    # Preprocess
    ap.add_argument("--log1p_plays", action="store_true", help="plays에 log1p 적용")
    ap.add_argument("--min_user_interactions", type=int, default=0)
    ap.add_argument("--min_item_interactions", type=int, default=0)
    ap.add_argument("--weighting", choices=["none","bm25","tfidf"], default="none",
                    help="학습용 R_train 가중치")
    # Scoring options
    ap.add_argument("--cosine_score", action="store_true", help="코사인 점수 사용(L2 정규화)")
    ap.add_argument("--pop_blend", type=float, default=0.0, help="0~1, 점수에 POP prior 블렌딩")
    ap.add_argument("--k", type=int, default=10)
    # Misc (자리 유지)
    ap.add_argument("--neg_samples", type=int, default=100)
    args = ap.parse_args()

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    ensure_dir(args.out_dir)

    # load & aggregate
    print("[1/8] Load & aggregate…")
    ui_raw = read_interactions(args.interactions_csv)

    # id maps
    print("[2/8] Build ID maps…")
    ui, user_map, art_map = build_id_maps(ui_raw)

    # log1p scaling (some play is too high)
    if args.log1p_plays:
        ui["plays"] = np.log1p(ui["plays"])

    # min interactions filter
    if args.min_user_interactions > 0 or args.min_item_interactions > 0:
        before = len(ui)
        ui = filter_min_interactions(ui, args.min_user_interactions, args.min_item_interactions)
        print(f"  - filter_min_interactions: {before:,} -> {len(ui):,}")

    n_users = int(ui["user_id"].max()+1)
    n_items = int(ui["artist_id"].max()+1)

    user_map.to_csv(os.path.join(args.out_dir,"user_mapping.csv"), index=False)
    art_map.to_csv(os.path.join(args.out_dir,"artist_mapping.csv"), index=False)

    # split train / val / test
    print("[3/8] Train/Val/Test split…")
    tr_df, va_df, te_df = split_train_val_test(ui, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    splits_dir = os.path.join(args.out_dir, "splits")
    ensure_dir(splits_dir)

    # save csv for later use(part2)
    tr_df.to_csv(os.path.join(splits_dir, "train.csv"), index=False)
    va_df.to_csv(os.path.join(splits_dir, "val.csv"), index=False)
    te_df.to_csv(os.path.join(splits_dir, "test.csv"), index=False)
    print(f"  - counts | train: {len(tr_df):,}  val: {len(va_df):,}  test: {len(te_df):,}")

    # CSR(Compressed Sparse Row vector) for als calculation
    print("[4/8] CSR build…")
    R_tr = to_csr(tr_df, n_users, n_items)
    R_va = to_csr(va_df, n_users, n_items)
    R_te = to_csr(te_df, n_users, n_items)

    # weighting for training
    print("[5/8] Apply weighting for training…")
    R_train_for_fit = R_tr
    if args.weighting == "bm25":
        R_train_for_fit = bm25_weight(R_tr, K1=1.2, B=0.75)
    elif args.weighting == "tfidf":
        R_train_for_fit = tfidf_weight(R_tr)

    # fit ALS
    print("[6/8] Fit ALS (implicit)…")
    model = AlternatingLeastSquares(
        factors=args.factors,
        regularization=args.reg,
        iterations=args.iterations,
        random_state=args.seed
    )
    model.fit((R_train_for_fit.T).tocsr() * args.alpha)

    # factor swap(train input raw = user)
    user_factors_true = model.item_factors  # real user
    item_factors_true = model.user_factors  # real item(artist)

    # optional cosine normalization for scoring
    U_for_score = user_factors_true
    V_for_score = item_factors_true
    if args.cosine_score:
        U_for_score = U_for_score / (np.linalg.norm(U_for_score, axis=1, keepdims=True) + 1e-12)
        V_for_score = V_for_score / (np.linalg.norm(V_for_score, axis=1, keepdims=True) + 1e-12)

    # popularity prior (0~1 scale for recommend score blending)
    item_pop = np.asarray(R_tr.sum(axis=0)).ravel().astype(float)
    if item_pop.size > 0:
        item_pop = np.log1p(item_pop)
        if item_pop.max() > 0:
            item_pop = item_pop / item_pop.max()
        else:
            item_pop = np.zeros_like(item_pop)
    else:
        item_pop = None

    rec_kwargs = dict(
        cosine_score=args.cosine_score,
        pop_prior=item_pop,
        pop_blend=max(0.0, min(1.0, args.pop_blend))
    )

    # evaluate with val/test
    print("[7/8] Evaluate (val/test)…")

    def _eval_user_count(R):
        cnt = 0
        for u in range(R.shape[0]):
            if R.indptr[u] != R.indptr[u+1]:
                cnt += 1
        return cnt

    val_users = _eval_user_count(R_va)
    metrics_val = {
        "precision@k": float(precision_at_k(U_for_score, V_for_score, R_tr, R_va, args.k, **rec_kwargs)) if val_users > 0 else None,
        "map@k":       float(map_at_k      (U_for_score, V_for_score, R_tr, R_va, args.k, **rec_kwargs)) if val_users > 0 else None,
        "ndcg@k":      float(ndcg_at_k     (U_for_score, V_for_score, R_tr, R_va, args.k, **rec_kwargs)) if val_users > 0 else None,
        "_users_evaluated": val_users,
    }
    with open(os.path.join(args.out_dir,"metrics_val.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_val, f, ensure_ascii=False, indent=2)
    print(f"Val (users={val_users}):", {k:v for k,v in metrics_val.items() if not k.startswith('_')})

    test_users = _eval_user_count(R_te)
    if test_users > 0:
        metrics_test = {
            "precision@k": float(precision_at_k(U_for_score, V_for_score, R_tr, R_te, args.k, **rec_kwargs)),
            "map@k":       float(map_at_k      (U_for_score, V_for_score, R_tr, R_te, args.k, **rec_kwargs)),
            "ndcg@k":      float(ndcg_at_k     (U_for_score, V_for_score, R_tr, R_te, args.k, **rec_kwargs)),
            "_users_evaluated": test_users,
        }
        with open(os.path.join(args.out_dir,"metrics_test.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_test, f, ensure_ascii=False, indent=2)
        print(f"Test(users={test_users}):", {k:v for k,v in metrics_test.items() if not k.startswith('_')})
        summary_payload = {"K": args.k, "val": metrics_val, "test": metrics_test}
    else:
        print("Test: skipped (empty split)")
        summary_payload = {"K": args.k, "val": metrics_val, "test": None}

    with open(os.path.join(args.out_dir,"metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    # save model & export top-K (train users)
    print("[8/8] Save model & export top-K (train users)…")
    np.savez_compressed(
        os.path.join(args.out_dir, "als_model.npz"),
        user_factors_true=user_factors_true,
        item_factors_true=item_factors_true,
        raw_user_factors=model.user_factors,
        raw_item_factors=model.item_factors,
        config=dict(
            alpha=args.alpha, factors=args.factors, reg=args.reg, iterations=args.iterations,
            val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
            log1p_plays=args.log1p_plays, min_user_interactions=args.min_user_interactions,
            min_item_interactions=args.min_item_interactions, weighting=args.weighting,
            cosine_score=args.cosine_score, pop_blend=args.pop_blend, k=args.k,
        )
    )
    export_topk_csv(U_for_score, V_for_score, R_tr, K=args.k,
                    out_csv=os.path.join(args.out_dir,"topk_user_recs.csv"), **rec_kwargs)
    print("Done.")

if __name__ == "__main__":
    main()
