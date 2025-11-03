import argparse, json, os, pickle
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import lightgbm as lgb

# Note: lightgbm을 사용한 리랭커 학습용 코드입니다. 앞선 과정에서 제작한 데이터셋을 바탕으로 리랭커를 학습합니다.

# Precision@K
# Top-K 추천 중 정답이 몇 개 들어갔는가.
def _precision_at_k(rel: np.ndarray, k: int) -> float:
    k = min(k, len(rel));  return 0.0 if k <= 0 else float(np.sum(rel[:k]) / k)

# avg Precision@K
def _average_precision_at_k(rel: np.ndarray, k: int) -> float:
    k = min(k, len(rel));  
    if k <= 0: return 0.0
    hits = 0; ap = 0.0
    pos = int(np.sum(rel))
    pos = max(pos, 1)
    for i in range(k):
        if rel[i] > 0:
            hits += 1
            ap += hits / (i + 1.0)
    return float(ap / pos)

# NDCG@K (Normalized Discounted Cumulative Gain at K)
# 앞순위에 더 큰 가중을 주면서 이상적인 순서와 비교해 정규화한 점수.
def _ndcg_at_k(rel: np.ndarray, k: int) -> float:
    k = min(k, len(rel)); 
    if k <= 0: return 0.0
    gains = (2 ** rel[:k] - 1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(np.sum((2 ** ideal - 1) * discounts))
    return float(dcg / idcg) if idcg > 0 else 0.0

def evaluate_grouped(df: pd.DataFrame, user_col: str, label_col: str, score_col: str, k: int) -> Dict[str, float]:
    precs, maps, ndcgs = [], [], []
    for _, g in df.groupby(user_col, sort=False):
        g = g.sort_values(score_col, ascending=False)
        rel = g[label_col].astype(int).values
        precs.append(_precision_at_k(rel, k))
        maps.append(_average_precision_at_k(rel, k))
        ndcgs.append(_ndcg_at_k(rel, k))
    n = len(precs)
    return {
        "precision@k": float(np.mean(precs) if n else 0.0),
        "map@k": float(np.mean(maps) if n else 0.0),
        "ndcg@k": float(np.mean(ndcgs) if n else 0.0),
        "_users_evaluated": int(n),
    }

# data loader
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.lower().endswith(".parquet") else pd.read_csv(path)

def make_lgb_dataset(df: pd.DataFrame, feats: List[str], group_by: str, label_col: str) -> lgb.Dataset:
    # group order sorting
    df = df.sort_values(group_by).reset_index(drop=True)

    groups = df.groupby(group_by, sort=False).size().tolist()

    X = df[feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[label_col].astype(int).values

    return lgb.Dataset(X, label=y, group=groups, free_raw_data=False), df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--features_meta", required=True)
    ap.add_argument("--eval", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--k", type=int, default=10)
    # param for hard train for more acc
    ap.add_argument("--num_leaves", type=int, default=31)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--n_estimators", type=int, default=2000)
    ap.add_argument("--min_data_in_leaf", type=int, default=50)
    ap.add_argument("--feature_fraction", type=float, default=0.9)
    ap.add_argument("--bagging_fraction", type=float, default=0.9)
    ap.add_argument("--bagging_freq", type=int, default=1)
    ap.add_argument("--lambda_l1", type=float, default=0.0)
    ap.add_argument("--lambda_l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    meta = json.loads(Path(args.features_meta).read_text(encoding="utf-8"))
    feats: List[str] = meta["features"]
    group_by = meta.get("group_by", "user_id")
    label_col = meta.get("label", "label")

    tr_raw = load_dataset(args.train).copy()
    for c in feats:
        if c not in tr_raw.columns: tr_raw[c] = 0.0
    tr_raw[group_by] = pd.to_numeric(tr_raw[group_by], errors="coerce").fillna(-1).astype(int)
    tr_raw[label_col] = tr_raw[label_col].astype(int)

    dtrain, tr = make_lgb_dataset(tr_raw, feats, group_by, label_col)

    valid_sets = [dtrain]
    valid_names = ["train"]

    if args.eval:
        ev_raw = load_dataset(args.eval).copy()
        for c in feats:
            if c not in ev_raw.columns: ev_raw[c] = 0.0
        ev_raw[group_by] = pd.to_numeric(ev_raw[group_by], errors="coerce").fillna(-1).astype(int)
        ev_raw[label_col] = ev_raw[label_col].astype(int)
        dvalid, ev = make_lgb_dataset(ev_raw, feats, group_by, label_col)
        valid_sets.append(dvalid)
        valid_names.append("valid")

    params = dict(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[args.k],
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        verbosity=-1,
        force_row_wise=True,
        seed=args.seed,
    )

    # model train
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100)
        ],
    )

    # save
    model_path = os.path.join(args.out_dir, "lgbm_reranker.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "features": feats, "k": args.k}, f)

    # simple eval
    tr_pred = model.predict(tr[feats], num_iteration=model.best_iteration)
    tr_eval = tr.copy(); tr_eval["pred"] = tr_pred
    train_metrics = evaluate_grouped(tr_eval, group_by, label_col, "pred", args.k)

    report = {"k": args.k, "train_rows": int(len(tr)), "train_users": int(tr[group_by].nunique()), "train": train_metrics}

    if args.eval:
        ev_pred = model.predict(ev[feats], num_iteration=model.best_iteration)
        ev_eval = ev.copy(); ev_eval["pred"] = ev_pred
        eval_metrics = evaluate_grouped(ev_eval, group_by, label_col, "pred", args.k)
        report["eval"] = eval_metrics

    report_path = os.path.join(args.out_dir, "report.json")
    Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    imp = pd.DataFrame({
        "feature": feats,
        "gain": model.feature_importance(importance_type="gain"),
        "split": model.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(os.path.join(args.out_dir, "feature_importance.csv"), index=False)

    print(f"[saved] model -> {model_path}")
    print(f"[saved] report -> {report_path}")
    print("train:", report.get("train"))
    if "eval" in report:
        print("eval:", report["eval"])

if __name__ == "__main__":
    main()
