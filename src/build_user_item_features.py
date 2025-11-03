import argparse, json, re, unicodedata
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# Note: 이 코드는 리랭커용 아이템 피처 유저 프로필을 생성하는 작업을 수행합니다.

# feature key management constant
INTER_USER = "user_id"
INTER_ITEM = "artist_id"
INTER_PLAYS = "plays"

ITEM_ID = "artist_id"
ITEM_TITLE = "artist_name"
ITEM_GENRES = "top_genres"

NUMERIC_CANDIDATES = [
    "danceability", "energy", "loudness", "tempo",
    "acousticness", "instrumentalness", "liveness", "speechiness", "valence",
    "key", "mode", "time_signature",
]
CATEGORICAL_CANDIDATES = ["explicit"]

# dir util
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

# data loader util
def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

# columns util
def pick_existing(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    return [c for c in candidates if c in df.columns]

# key type match util
def to_str_key(df: pd.DataFrame, key: str):
    df[key] = df[key].astype(str)

# bool mapping helper
def boolize_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float)
    m = s.astype(str).str.lower().map({
        "true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0, "1": 1, "0": 0
    })
    if 1.0 - m.isna().mean() >= 0.8:
        return m.fillna(0).astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0)

# norm helper for just in case
def normalize_key(x: str) -> str:
    if pd.isna(x):
        return ""
    x = unicodedata.normalize("NFKD", str(x)).lower()
    x = "".join(ch for ch in x if not unicodedata.combining(ch))
    x = re.sub(r"[\s\-_]+", "", x)
    x = re.sub(r"[^\w]", "", x)
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions_csv", required=True,
                    help="data/derived/als/splits/train.csv")
    ap.add_argument("--artist_features_csv", required=True,
                    help="data/derived/spotify_full/artist_features_weighted.csv")
    ap.add_argument("--out_dir", required=True,
                    help="data/derived/features")
    ap.add_argument("--topk_genre", type=int, default=3)

    # id map
    ap.add_argument("--id_map_csv", help="optional id map csv (internal item id -> spotify artist_id)")
    ap.add_argument("--id_map_old_col", default="old_item_id",
                    help="column in id_map_csv for internal item id")
    ap.add_argument("--id_map_new_col", default="artist_id",
                    help="column in id_map_csv for spotify artist_id")

    # name map
    ap.add_argument("--inter_item_col", default="artist_id",
                    help="interactions에서 아이템이 들어있는 컬럼명 (예: artist_name)")
    ap.add_argument("--use_name_join", action="store_true",
                    help="이름으로 items와 조인할 때 켬")

    # dir for als id map
    ap.add_argument("--artist_name_map_csv",
                    help="optional csv to map internal item id to artist name (e.g., artist_mapping.csv)")
    ap.add_argument("--artist_name_map_id_col", default="artist_id",
                    help="column in artist_name_map_csv for internal item id")
    ap.add_argument("--artist_name_map_name_col", default="Artist",
                    help="column in artist_name_map_csv for artist name")

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    # load
    inter_raw = read_csv(args.interactions_csv)
    items = read_csv(args.artist_features_csv)

    # schema check
    for need in [INTER_USER, INTER_PLAYS]:
        if need not in inter_raw.columns:
            raise ValueError(f"[interactions] missing column: {need}")
    if ITEM_ID not in items.columns or ITEM_TITLE not in items.columns:
        raise ValueError(f"[items] missing columns: need [{ITEM_ID}, {ITEM_TITLE}]")

    # key ready
    inter = inter_raw.copy()

    # method 1. id mapping
    mapped_by_id = False
    if args.id_map_csv:
        idmap = read_csv(args.id_map_csv)
        old_col, new_col = args.id_map_old_col, args.id_map_new_col

        if old_col not in idmap.columns or new_col not in idmap.columns:
            raise ValueError(f"[id_map_csv] columns not found: {old_col}, {new_col}")
        
        if args.inter_item_col not in inter.columns:
            raise ValueError(f"[interactions] missing item column: {args.inter_item_col}")
        
        inter[args.inter_item_col] = inter[args.inter_item_col].astype(str)
        
        idmap = idmap[[old_col, new_col]].dropna()
        idmap[old_col] = idmap[old_col].astype(str)
        idmap[new_col] = idmap[new_col].astype(str)

        inter = inter.merge(idmap, how="left", left_on=args.inter_item_col, right_on=old_col)
        inter[INTER_ITEM] = inter[new_col].where(inter[new_col].notna(), inter[args.inter_item_col].astype(str))
        inter.drop(columns=[old_col, new_col], inplace=True)
        
        mapped_by_id = True
    else:
        if args.inter_item_col not in inter.columns:
            raise ValueError(f"[interactions] missing item column: {args.inter_item_col}")

    # method 2. name mapping
    if args.artist_name_map_csv:
        name_map = read_csv(args.artist_name_map_csv)
        id_col = args.artist_name_map_id_col
        nm_col = args.artist_name_map_name_col

        if id_col not in name_map.columns or nm_col not in name_map.columns:
            raise ValueError(f"[artist_name_map_csv] columns not found: {id_col}, {nm_col}")
        
        if not mapped_by_id:
            inter[args.inter_item_col] = inter[args.inter_item_col].astype(str)
            name_map[id_col] = name_map[id_col].astype(str)
            inter = inter.merge(name_map[[id_col, nm_col]], how="left",
                                left_on=args.inter_item_col, right_on=id_col)
            inter["__mapped_name__"] = inter[nm_col]
            inter.drop(columns=[id_col, nm_col], inplace=True, errors="ignore")

    # final key decision
    use_name_join = args.use_name_join and not mapped_by_id
    if use_name_join:
        name_col = "__mapped_name__" if "__mapped_name__" in inter.columns else args.inter_item_col
        if name_col not in inter.columns:
            raise ValueError(f"[interactions] no name column available for name join: {name_col}")
        inter = inter[[INTER_USER, name_col, INTER_PLAYS]].rename(columns={name_col: "raw_name"})
    else:
        if not mapped_by_id:
            inter[INTER_ITEM] = inter[args.inter_item_col].astype(str)
        inter = inter[[INTER_USER, INTER_ITEM, INTER_PLAYS]]

    # type change for stable
    if INTER_USER in inter.columns:
        try:
            inter[INTER_USER] = inter[INTER_USER].astype(int)
        except Exception:
            pass

    # item_features: popularity + side feature
    num_cols = pick_existing(items, NUMERIC_CANDIDATES)
    cat_cols = pick_existing(items, CATEGORICAL_CANDIDATES)
    genre_col = ITEM_GENRES if ITEM_GENRES in items.columns else None

    it = items.copy()
    for c in cat_cols:
        it[c] = boolize_series(it[c])

    for c in num_cols:
        it[c] = pd.to_numeric(it[c], errors="coerce").fillna(0.0)

    if genre_col:
        it[genre_col] = it[genre_col].fillna("")

    # get popularity
    if use_name_join:
        tmp = inter.copy()
        tmp["join_key"] = tmp["raw_name"].astype(str).map(normalize_key)
        it["join_key"] = it[ITEM_TITLE].astype(str).map(normalize_key)
        
        tmp = tmp.merge(it[[ITEM_ID, "join_key"]].drop_duplicates("join_key"),
                        on="join_key", how="left")
        tmp = tmp.drop(columns=["join_key"])
        tmp[ITEM_ID] = tmp[ITEM_ID].astype(str)

        pop = (
            tmp.dropna(subset=[ITEM_ID])
               .groupby(ITEM_ID)
               .agg(item_pop_users=(INTER_USER, "nunique"),
                    item_pop_plays=(INTER_PLAYS, "sum"))
               .reset_index()
        )
    else:
        inter[INTER_ITEM] = inter[INTER_ITEM].astype(str)
        pop = (
            inter.groupby(INTER_ITEM)
                 .agg(item_pop_users=(INTER_USER, "nunique"),
                      item_pop_plays=(INTER_PLAYS, "sum"))
                 .reset_index()
                 .rename(columns={INTER_ITEM: ITEM_ID})
        )

    item_features = it.merge(pop, how="left", on=ITEM_ID)
    item_features["item_pop_users"] = item_features["item_pop_users"].fillna(0).astype(int)
    item_features["item_pop_plays"] = item_features["item_pop_plays"].fillna(0).astype(float)

    # make user profiles
    if use_name_join:
        df = inter[[INTER_USER, "raw_name", INTER_PLAYS]].copy()
        df["join_key"] = df["raw_name"].astype(str).map(normalize_key)
        item_features["join_key"] = item_features[ITEM_TITLE].astype(str).map(normalize_key)
        df = df.merge(item_features[[ITEM_ID, "join_key"] + num_cols + cat_cols + ([genre_col] if genre_col else [])],
                      on="join_key", how="left")
        df = df.drop(columns=["join_key"])
    else:
        df = inter[[INTER_USER, INTER_ITEM, INTER_PLAYS]].copy()
        df = df.merge(item_features[[ITEM_ID] + num_cols + cat_cols + ([genre_col] if genre_col else [])],
                      left_on=INTER_ITEM, right_on=ITEM_ID, how="left")

    df[INTER_PLAYS] = pd.to_numeric(df[INTER_PLAYS], errors="coerce").fillna(0.0)
    df = df[df[INTER_PLAYS] > 0]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    def _wavg(g: pd.DataFrame) -> pd.Series:
        if not len(num_cols):
            return pd.Series(dtype=float)
        
        w = g[INTER_PLAYS].astype(float).values.reshape(-1, 1)  # (n,1)
        X = g[num_cols].astype(float).values # (n,d)
        
        mask = ~np.isnan(X)
        w_masked = w * mask
        
        denom = w_masked.sum(axis=0)
        denom[denom == 0] = 1.0
        
        num = (np.nan_to_num(X) * w_masked).sum(axis=0)
        
        return pd.Series((num / denom), index=num_cols)

    if len(num_cols):
        up_num = df.groupby(INTER_USER, group_keys=False)[num_cols + [INTER_PLAYS]].apply(_wavg)
    else:
        up_num = pd.DataFrame(index=df[INTER_USER].unique())

    up_cat = {}
    for c in cat_cols:
        agg = df.groupby(INTER_USER, group_keys=False)[[c, INTER_PLAYS]] \
                .apply(lambda g: float(np.average(g[c].values, weights=g[INTER_PLAYS].values)))
        up_cat[c] = agg
    up_cat = pd.DataFrame(up_cat) if len(up_cat) else pd.DataFrame(index=up_num.index)

    # genre Top-K + entropy
    if genre_col:
        def explode_genre(sub: pd.DataFrame):
            bag = {}
            total = 0.0
            for _, r in sub.iterrows():
                w = float(r[INTER_PLAYS])
                s = str(r[genre_col]) if pd.notna(r[genre_col]) else ""

                tokens = []
                
                if s.startswith("[") and s.endswith("]"):
                    try:
                        import ast
                        tokens = [str(x).strip() for x in ast.literal_eval(s)]
                    except Exception:
                        tokens = []
                
                if not tokens:
                    tokens = [t.strip() for t in s.replace("|", ",").split(",") if t.strip()]
                
                if not tokens:
                    continue
                
                share = w / len(tokens)  # multi genre
                
                for t in tokens:
                    bag[t] = bag.get(t, 0.0) + share
                total += w

            if total <= 0 or not bag:
                return {}

            ssum = sum(bag.values())
            if ssum > 0:
                for k in list(bag.keys()):
                    bag[k] = bag[k] / ssum
            return bag

        gstats = {u: explode_genre(g) for u, g in df.groupby(INTER_USER)}
        cols = [f"user_genre_top{i+1}" for i in range(args.topk_genre)] + ["user_genre_entropy"]
        idx, rows = [], []

        for u, dist in gstats.items():
            idx.append(u)
            if not dist:
                rows.append([np.nan] * args.topk_genre + [0.0])
                continue
            top = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:args.topk_genre]
            top_names = [t[0] for t in top]
            p = np.array(list(dist.values()), dtype=float)
            p = p / p.sum() if p.sum() > 0 else p
            p = p[p > 0]
            entropy = float(-(p * np.log2(p)).sum())  # log2
            row = top_names + [entropy]
            while len(row) < len(cols):
                row.insert(len(row) - 1, np.nan)
            rows.append(row)
        up_genre = pd.DataFrame(rows, index=idx, columns=cols)
    else:
        up_genre = pd.DataFrame(index=up_num.index)

    # merge
    out = pd.DataFrame(index=up_num.index)
    if len(up_num):
        out = out.join(up_num, how="outer")
        out.rename(columns={c: f"uf_{c}" for c in up_num.columns}, inplace=True)

    if len(up_cat):
        out = out.join(up_cat, how="outer")
        out.rename(columns={c: f"uf_{c}" for c in up_cat.columns}, inplace=True)

    if not up_genre.empty:
        out = out.join(up_genre, how="outer")

    user_profiles = out.reset_index().rename(columns={"index": INTER_USER})

    # NaN filling
    for c in user_profiles.columns:
        if c == INTER_USER:
            continue
        if pd.api.types.is_numeric_dtype(user_profiles[c]):
            user_profiles[c] = user_profiles[c].fillna(0.0)

    # save
    item_out = str(Path(args.out_dir) / "item_features.csv")
    user_out = str(Path(args.out_dir) / "user_profiles.csv")
    meta_out = str(Path(args.out_dir) / "features_meta.json")

    item_features.to_csv(item_out, index=False)
    user_profiles.to_csv(user_out, index=False)

    meta = {
        "interaction_cols": {"user_col": INTER_USER,
                             "item_col": ("raw_name" if use_name_join else INTER_ITEM),
                             "plays_col": INTER_PLAYS},
        "item_id_col_in_input": ITEM_ID,
        "genre_col": genre_col,
        "numeric_item_feature_cols": [c for c in num_cols if c in item_features.columns],
        "categorical_item_feature_cols": [c for c in cat_cols if c in item_features.columns],
        "user_profile_cols": [c for c in user_profiles.columns if c != INTER_USER],
        "notes": {
            "join_strategy": ("name_join" if use_name_join else "id_join"),
            "id_map_used": bool(args.id_map_csv),
            "artist_name_map_used": bool(args.artist_name_map_csv),
            "numeric_user_profile": "plays 가중평균(NA 제외)",
            "categorical_user_profile": "0/1 비율(plays 가중평균)",
            "genre_topk": args.topk_genre,
            "item_popularity": ["item_pop_users", "item_pop_plays"],
            "genre_entropy_log_base": 2
        }
    }
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"item_features -> {item_out}")
    print(f"user_profiles -> {user_out}")
    print(f"features_meta -> {meta_out}")

if __name__ == "__main__":
    main()
