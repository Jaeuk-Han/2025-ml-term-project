import argparse, json
from pathlib import Path
from typing import Optional, Set, Tuple

import numpy as np
import pandas as pd

# Note: 이 코드는 훈련된 모델을 불러 추론을 진행합니다.

# model loader
def load_model(model_npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    npz = np.load(model_npz_path)
    U = npz["user_factors_true"]  # [n_users, f]
    V = npz["item_factors_true"]  # [n_items, f]
    return U, V

# pre-stored map load
def load_mappings(user_mapping_csv: str, artist_mapping_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users = pd.read_csv(user_mapping_csv)
    arts = pd.read_csv(artist_mapping_csv)
    if "user_id" not in users.columns:
        raise ValueError("user_mapping.csv must contain 'user_id' column")
    if "artist_id" not in arts.columns:
        raise ValueError("artist_mapping.csv must contain 'artist_id' column")
    return users, arts


def normalize_pop_from_csv(pop_csv: str, n_items: int, artist_map: pd.DataFrame) -> Optional[np.ndarray]:
    if not pop_csv:
        return None

    df = pd.read_csv(pop_csv)

    # depensive
    if "artist_id" in df.columns:
        counts = df["artist_id"].astype(int).value_counts()
    elif "Artist" in df.columns and "Artist" in artist_map.columns:
        # Artist 문자열 → artist_id 매핑
        m = df.merge(artist_map[["artist_id", "Artist"]], on="Artist", how="left")
        counts = m["artist_id"].dropna().astype(int).value_counts()
    else:
        # 다른 포맷은 무시
        return None

    pop = np.zeros(n_items, dtype=float)

    valid_ids = counts.index.values
    valid_ids = valid_ids[(valid_ids >= 0) & (valid_ids < n_items)]
    pop[valid_ids] = counts.loc[valid_ids].values.astype(float)

    # 로그 스케일 정규화
    pop = np.log1p(pop)
    mx = pop.max()
    if mx > 0:
        pop = pop / mx
    return pop


def collect_seen_items(
    interactions_csv: Optional[str],
    target_user_id: int,
    user_map: pd.DataFrame,
    artist_map: pd.DataFrame,
) -> Set[int]:
    if not interactions_csv:
        return set()

    df = pd.read_csv(interactions_csv)
    cols = {c.lower(): c for c in df.columns}

    if "user_id" in cols and "artist_id" in cols:
        ucol, icol = cols["user_id"], cols["artist_id"]
        return set(df.loc[df[ucol].astype(int) == target_user_id, icol].astype(int).tolist())

    # Username/Artist → id 매핑
    if "username" in cols and "artist" in cols and "Username" in user_map.columns and "Artist" in artist_map.columns:
        df2 = df.rename(columns={cols["username"]: "Username", cols["artist"]: "Artist"})
        m = df2.merge(user_map[["user_id", "Username"]], on="Username", how="left")
        m = m.merge(artist_map[["artist_id", "Artist"]], on="Artist", how="left")
        m = m.dropna(subset=["user_id", "artist_id"])
        return set(m.loc[m["user_id"].astype(int) == target_user_id, "artist_id"].astype(int).tolist())

    # 그 외 포맷은 제외
    return set()



def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k >= scores.size:
        return np.argsort(-scores)
    idx = np.argpartition(-scores, k)[:k]
    return idx[np.argsort(-scores[idx])]

# recommender function
def recommend_for_user(
    user_id: int,
    U: np.ndarray,
    V: np.ndarray,
    seen_items: Set[int],
    k: int,
    pop_prior: Optional[np.ndarray] = None,
    pop_blend: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    scores = U[user_id] @ V.T  # U,V가 코사인 정규화됐다면 코사인 점수와 동일
    if pop_prior is not None and pop_blend > 0:
        scores = (1.0 - pop_blend) * scores + pop_blend * pop_prior

    # 이미 본 아이템 제외
    for i in seen_items:
        if 0 <= i < scores.shape[0]:
            scores[i] = -np.inf

    idx = topk_indices(scores, k)
    return idx, scores[idx]


def main():
    ap = argparse.ArgumentParser(description="Predict Top-K recommendations for a user from ALS factors.")
    # 필수 파라미터
    ap.add_argument("--model", required=True, help="Path to als_model.npz")
    ap.add_argument("--user_mapping", required=True, help="Path to user_mapping.csv")
    ap.add_argument("--artist_mapping", required=True, help="Path to artist_mapping.csv")
    ap.add_argument("--k", type=int, default=10)

    # 대상 유저 지정
    ap.add_argument("--user_id", type=int, help="Internal numeric user_id")
    ap.add_argument("--username", help="Alternative to user_id (must exist in user_mapping.csv)")

    # 점수 옵션 (학습 평가와 정합성 유지)
    ap.add_argument("--cosine_score", action="store_true", help="L2-normalize U,V before scoring")
    ap.add_argument("--pop_blend", type=float, default=0.0, help="Blend amount for popularity prior (0~1)")
    ap.add_argument("--pop_from_csv", help="CSV to compute popularity prior (e.g., splits/train.csv)")

    # User seen item 제외
    ap.add_argument("--interactions_csv", help="CSV to build seen-items set (e.g., splits/train.csv)")

    # 출력
    ap.add_argument("--out_json", default="pred.json")
    ap.add_argument("--out_csv", default=None, help="If set, also save recommendations to CSV")

    args = ap.parse_args()

    # 모델 / 매핑 로드
    U, V = load_model(args.model)
    users, arts = load_mappings(args.user_mapping, args.artist_mapping)

    n_users, n_items = U.shape[0], V.shape[0]

    # user_id 결정
    if args.user_id is not None:
        uid = int(args.user_id)
    elif args.username is not None and "Username" in users.columns:
        row = users.loc[users["Username"] == str(args.username)]
        if row.empty:
            raise SystemExit(f"Username not found in user_mapping: {args.username}")
        uid = int(row["user_id"].iloc[0])
    else:
        raise SystemExit("Provide --user_id or --username (if user_mapping has 'Username').")

    if not (0 <= uid < n_users):
        raise SystemExit(f"user_id out of range: {uid} (n_users={n_users})")

    # 코사인 점수
    if args.cosine_score:
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    # popurality prior
    pop_prior = normalize_pop_from_csv(args.pop_from_csv, n_items, arts) if args.pop_from_csv else None
    blend = max(0.0, min(1.0, float(args.pop_blend)))

    # seen 집합
    seen = collect_seen_items(args.interactions_csv, uid, users, arts) if args.interactions_csv else set()

    # 추천
    idx, sco = recommend_for_user(uid, U, V, seen, args.k, pop_prior=pop_prior, pop_blend=blend)

    # 결과 조립
    out_df = pd.DataFrame({"artist_id": idx.astype(int), "score": sco.astype(float)})
    out_df = out_df.merge(arts, on="artist_id", how="left")  # Artist 이름 붙이기(있으면)

    payload = {
        "user_id": uid,
        "k": int(args.k),
        "cosine_score": bool(args.cosine_score),
        "pop_blend": float(blend),
        "recommendations": out_df.to_dict(orient="records"),
    }

    # 저장
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {args.out_json}")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out_csv, index=False)
        print(f"CSV saved:  → {args.out_csv}")


if __name__ == "__main__":
    main()
