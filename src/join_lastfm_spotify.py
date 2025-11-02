import argparse, json
from pathlib import Path
import pandas as pd

# normalize artist name for join (hard norm)
def make_key_series(s: pd.Series, mode: str):
    if mode == "exact":
        return s.astype(str)
    elif mode == "loose":
        import unicodedata, re
        def norm(x):
            if pd.isna(x): return ""
            t = unicodedata.normalize("NFKD", str(x))
            t = "".join(ch for ch in t if not unicodedata.combining(ch))
            t = t.lower()
            t = re.sub(r"\s+", " ", t).strip()
            t = re.sub(r"[^a-z0-9]+", "", t)
            return t
        return s.map(norm)
    else:
        return s.astype(str).str.lower().str.strip()

def run(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # load last.fm for aggregate
    lf = pd.read_csv(args.lastfm_csv)

    cols = {c.lower(): c for c in lf.columns}

    user_col   = cols.get("username")
    artist_col = cols.get("artist")
    plays_col  = cols.get("plays") or cols.get("playcount")

    # depensive for both name (last.fm)
    if user_col is None or artist_col is None:
        raise ValueError(f'Expected "Username" and "Artist". Found: {list(lf.columns)}')
    if plays_col is None:
        lf["__ones__"] = 1; plays_col = "__ones__"

    agg = (lf.groupby([user_col, artist_col], as_index=False)[plays_col]
             .sum()
             .rename(columns={user_col:"Username", artist_col:"Artist", plays_col:"plays"}))
    if args.min_plays > 1:
        agg = agg[agg["plays"] >= args.min_plays].reset_index(drop=True)

    # load spotify features
    sp = pd.read_csv(args.spotify_features_csv)
    sc = {c.lower(): c for c in sp.columns}

    name_col = sc.get("artist_name")
    id_col   = sc.get("artist_id")

    # depensive for name and id (spotify)
    if name_col is None or id_col is None:
        raise ValueError('Spotify features must include "artist_name" and "artist_id".')

    # identify numeric feature columns (exclude ids)
    numeric_cols = sp.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    if id_col in numeric_cols:
        numeric_cols.remove(id_col)

    sp_key = sp[[name_col, id_col] + numeric_cols].copy()
    sp_key = sp_key.rename(columns={name_col:"artist_name", id_col:"artist_id"})

    # join keys
    agg["__key__"] = make_key_series(agg["Artist"], args.normalize)
    sp_key["__key__"] = make_key_series(sp_key["artist_name"], args.normalize)

    # deduplicate spotify rows per key by averaging numeric features
    means = sp_key.groupby(["__key__", "artist_id"], as_index=False)[numeric_cols].mean()

    # join
    joined = agg.merge(means, on="__key__", how="left").drop(columns="__key__")
    matched   = joined[joined["artist_id"].notna()].copy()
    unmatched = joined[joined["artist_id"].isna()].copy()

    # collapse duplicates that might arise from multiple artist_id sharing a key
    if not matched.empty:
        matched["artist_id"] = matched["artist_id"].astype(str)
        # Keep Artist (display name) from Last.fm. Artist_name on Spotify is not strictly needed here.
        # Sum plays when same (Username, Artist, artist_id) occurs, average features per (artist_id)
        base = (matched.groupby(["Username","Artist","artist_id"], as_index=False)["plays"].sum())
        feat = (matched.groupby(["artist_id"], as_index=False)[numeric_cols].mean())
        joined_feat = base.merge(feat, on="artist_id", how="left")
    else:
        base = matched.copy()
        joined_feat = matched.copy()

    # artist-level features table (1 row per artist_id)
    if not matched.empty:
        # try to recover a representative artist_name per artist_id using the most common artist
        repr_name = (matched.groupby(["artist_id","Artist"], as_index=False)["plays"]
                            .sum().sort_values(["artist_id","plays"], ascending=[True,False])
                            .drop_duplicates(["artist_id"])
                            .rename(columns={"Artist":"artist_name"}))[["artist_id","artist_name"]]
        artist_features = repr_name.merge(feat, on="artist_id", how="left") if 'feat' in locals() else repr_name
    else:
        artist_features = pd.DataFrame(columns=["artist_id","artist_name"] + numeric_cols)

    # save
    (base if not base.empty else matched)[["Username","Artist","artist_id","plays"]]\
        .to_csv(out_dir/"joined.csv", index=False)
    joined_feat.to_csv(out_dir/"joined_with_features.csv", index=False)

    # fallback save
    (unmatched.groupby("Artist", as_index=False)["plays"].sum()
             .sort_values("plays", ascending=False)
             .to_csv(out_dir/"unmatched.csv", index=False))

    artist_features.to_csv(out_dir/"artist_features_joined.csv", index=False)

    cov = {
        "lastfm_rows": int(len(lf)),
        "agg_pairs": int(len(agg)),
        "matched_rows": int(len(base)),
        "unmatched_artists": int(unmatched["Artist"].nunique()) if not unmatched.empty else 0,
        "joined_with_features_cols": ["Username","Artist","artist_id","plays"] + numeric_cols
    }
    with open(out_dir/"coverage.json", "w", encoding="utf-8") as f:
        json.dump(cov, f, ensure_ascii=False, indent=2)

    print(f"joined(just play cnt):                 {out_dir/'joined.csv'}")
    print(f"joined_with_features(with full feature):   {out_dir/'joined_with_features.csv'}")
    # Note: 정규화된 이름 기준으로 한번 더 뽑아두었으니 이걸 규칙 생성에 사용하시면 될듯 합니다.
    print(f"artist_features_joined(just artist with feature): {out_dir/'artist_features_joined.csv'}")
    print(f"unmatched(fallback):              {out_dir/'unmatched.csv'}")
    print(f"coverage(coverage for ppt):               {out_dir/'coverage.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lastfm_csv", type=str, default="data/lastfm/Last.fm_data.csv")
    ap.add_argument("--spotify_features_csv", type=str, default="data/derived/spotify_full/artist_features_weighted.csv")
    ap.add_argument("--out_dir", type=str, default="data/derived/lastfm_join_strict")
    ap.add_argument("--min_plays", type=int, default=1)
    ap.add_argument("--normalize", type=str, default="basic", choices=["basic","exact","loose"])
    args = ap.parse_args()
    run(args)
