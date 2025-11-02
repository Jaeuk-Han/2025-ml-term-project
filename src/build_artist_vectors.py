import argparse
import ast
import json
import unicodedata
from statistics import mode, StatisticsError
from pathlib import Path

import numpy as np
import pandas as pd


# Note: 이 코드는 artist 단위 벡터 제작코드로 spotify 데이터셋을 입력 받아 가중합을 통해 아티스트 당 하나의 특성 벡터를 출력합니다.
# 입력: artist.csv, track.csv, Last.fm_data.csv
# 출력: artist_features_weighted.csv(아티스트), artist_genre_aggregates.csv(장르별), coverage.json(last.fm과의 매칭 비율 사전 확인)

# text preprocessing util of artist name (simple norm because will normalize again)
def normalize_key(text: str) -> str:
    """Lowercase, strip accents, remove non-alnum (keep only [a-z0-9])."""
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text.lower() if ch.isalnum())

# string parse util
def parse_listlike(x):
    """Parse list-like strings such as "['a','b']" into list; pass lists through."""
    if isinstance(x, list):
        return x
    s = str(x)
    try:
        return ast.literal_eval(s)
    except Exception:
        s = s.strip().strip("[]")
        return [t.strip(" '\"") for t in s.split(",") if t.strip()]

# mode util
def mode_safe(series: pd.Series):
    vals = series.dropna().tolist()
    if not vals: # NaN filter
        return np.nan
    try:
        return mode(vals)
    except StatisticsError:
        return vals[0]

# weight avg util
def weighted_avg(df: pd.DataFrame, cols, wcol="w"):
    if df.empty: # NaN filter
        return {c: np.nan for c in cols}
    w = df[wcol].values.reshape(-1, 1)
    X = df[cols].values.astype(float)
    mask = ~np.isnan(X)
    w_masked = (w * mask)
    denom = w_masked.sum(axis=0)
    denom[denom == 0] = 1.0
    num = (np.nan_to_num(X) * w_masked).sum(axis=0)
    return dict(zip(cols, (num / denom)))



# core function
def build_artist_vectors(
    artists_csv,
    tracks_csv,
    lastfm_csv=None,
    out_features_csv="artist_features_weighted.csv",
    out_per_genre_csv="artist_genre_aggregates.csv",
    out_coverage_json="coverage.json",
):
    # data load
    artists = pd.read_csv(artists_csv)
    tracks = pd.read_csv(tracks_csv)

    # lower prevent missmatching
    artists.columns = [c.lower() for c in artists.columns]
    tracks.columns = [c.lower() for c in tracks.columns]

    # name-based mapping (coverage report if last,fm provided) for final ppt
    artists["artist_key"] = artists["name"].map(normalize_key)
    artists["id"] = artists["id"].astype(str)
    id2name = artists[["id", "name"]].rename(columns={"id": "artist_id", "name": "artist_name"})
    name_to_id = dict(zip(artists["artist_key"], id2name["artist_id"]))

    coverage = None
    if lastfm_csv is not None:
        lastfm = pd.read_csv(lastfm_csv)
        lastfm.columns = [c.lower() for c in lastfm.columns]
        artist_col = "artist" if "artist" in lastfm.columns else (
            "artist_name" if "artist_name" in lastfm.columns else lastfm.columns[1]
        )
        lastfm["artist_key"] = lastfm[artist_col].map(normalize_key)
        cov_mask = lastfm["artist_key"].map(name_to_id).notna()
        total_unique = max(1, lastfm["artist_key"].nunique())
        coverage = {
            "row_coverage_pct": round(float(cov_mask.mean() * 100), 2),
            "unique_artist_coverage_pct": round(
                100.0 * lastfm.loc[cov_mask, "artist_key"].nunique() / total_unique, 2
            ),
            "total_rows": int(len(lastfm)),
            "mapped_rows": int(cov_mask.sum()),
            "unique_artist_keys": int(lastfm["artist_key"].nunique()),
            "unique_artist_keys_mapped": int(lastfm.loc[cov_mask, "artist_key"].nunique()),
        }

    # tracks explode
    tracks["id_artists_list"] = tracks["id_artists"].map(parse_listlike)
    tracks_expl = tracks.explode("id_artists_list").rename(columns={"id_artists_list": "artist_id"})
    tracks_expl["artist_id"] = tracks_expl["artist_id"].astype(str)

    # co-artist genre filling reduce <unknown>
    # Base artist to genres
    # Note: Spotify artists.csv의 genres 탭이 비어있는 경우가 생각보다 많아서 결측치를 채우기 위한 방법론으로 추가한 파트입니다.
    # 그냥 최빈값이나 다른 방식으로 채우기에는 우리의 도메인에는 적합하지 않아 보입니다.
    # 보통 하나의 음악에 대해서 여러 아티스트가 참여하는 경우가 존재하는데, 이 아티스트들은 유사한 장르를 공유한다는 가정으로 
    # 결측치를 같이 작업한 아티스트들의 값으로 채웁니다.
    # 참고할 값이 없는 경우에는 <unknown>으로 fallback 해두었습니다.

    artists["genres_list"] = artists["genres"].map(parse_listlike)
    artist_genre_set = {aid: set(gs or []) for aid, gs in zip(artists["id"].astype(str), artists["genres_list"])}

    # track to list of artists map (needs track id)
    if "id" in tracks.columns:
        track_to_artists = dict(zip(tracks["id"], tracks["id_artists"].map(parse_listlike)))
        # For each track collect union of genres from present artists
        for tid, aids in track_to_artists.items():
            if not isinstance(aids, list) or len(aids) == 0:
                continue
            union_g = set().union(*[artist_genre_set.get(str(a), set()) for a in aids])
            if not union_g:
                continue
            # fill only artists that have empty genre set
            for a in aids:
                a = str(a)
                if not artist_genre_set.get(a):
                    artist_genre_set[a] = set(union_g)

    # helper for unknown genre
    def genres_for_artist(aid: str):
        gs = artist_genre_set.get(aid, set())
        return list(gs) if gs else ["<unknown>"]

    # numeric
    num_cols_all = [
        "danceability", "energy", "loudness", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "duration_ms", "popularity"
    ]

    num_cols = [c for c in num_cols_all if c in tracks_expl.columns]

    # categoric
    cat_candidates = ["key", "mode", "time_signature", "explicit"]
    cat_cols = [c for c in cat_candidates if c in tracks_expl.columns]

    # attach genres to each artist and track row and explode by genre
    tracks_expl["genres_for_artist"] = tracks_expl["artist_id"].map(genres_for_artist)
    tracks_expl = tracks_expl.explode("genres_for_artist").rename(columns={"genres_for_artist": "genre"})

    # aggregation into genre with mean and mode
    agg_dict = {c: "mean" for c in num_cols}
    for c in cat_cols:
        agg_dict[c] = mode_safe
    per_ag = tracks_expl.groupby(["artist_id", "genre"]).agg(agg_dict).reset_index()

    # Genre weights = share of tracks in that genre for an artist
    counts = tracks_expl.groupby(["artist_id", "genre"]).size().reset_index(name="n")
    counts["w"] = counts.groupby("artist_id")["n"].transform(lambda x: x / x.sum())
    per_ag = per_ag.merge(counts[["artist_id", "genre", "w"]], on=["artist_id", "genre"], how="left")

    # Weighted average across genres for final artist vector
    # Note: Proposal에 넣은 내용대로 연속형의 경우 각 장르의 평균으로 이산형의 경우 최빈값으로 넣었습니다.
    rows = []
    for aid, sub in per_ag.groupby("artist_id"):
        out = {"artist_id": aid}
        out.update(weighted_avg(sub, num_cols, "w"))
        if not sub.empty:
            top_row = sub.loc[sub["w"].idxmax()]
            for c in cat_cols:
                out[c] = top_row[c]
            out["top_genres"] = ",".join(sub.sort_values("w", ascending=False)["genre"].head(3).tolist())
        else:
            for c in cat_cols:
                out[c] = np.nan
            out["top_genres"] = ""
        rows.append(out)

    artist_features = pd.DataFrame(rows)

    # Attach artist names by comparing the spotify and last.fm
    artist_features = artist_features.merge(id2name, on="artist_id", how="left")
    first_cols = ["artist_id", "artist_name"]
    artist_features = artist_features[first_cols + [c for c in artist_features.columns if c not in first_cols]]

    per_ag = per_ag.merge(id2name, on="artist_id", how="left")
    pg_first = ["artist_id", "artist_name", "genre"]
    per_ag = per_ag[pg_first + [c for c in per_ag.columns if c not in pg_first]]

    # dir ensure
    Path(out_features_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_per_genre_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_coverage_json).parent.mkdir(parents=True, exist_ok=True)

    # Save the result
    artist_features.to_csv(out_features_csv, index=False)
    per_ag.to_csv(out_per_genre_csv, index=False)
    if coverage is not None:
        with open(out_coverage_json, "w", encoding="utf-8") as f:
            json.dump(coverage, f, ensure_ascii=False, indent=2)

    return out_features_csv, out_per_genre_csv, out_coverage_json



def main():
    # parser for CLI
    ap = argparse.ArgumentParser(
        description="Build artist vectors via name mapping + (co-artist-imputed) genre-weighted aggregation."
    )
    ap.add_argument("--artists_csv", required=True)
    ap.add_argument("--tracks_csv", required=True)
    ap.add_argument("--lastfm_csv", default=None, help="Optional; only for coverage report.")
    ap.add_argument("--out_features_csv", default="artist_features_weighted.csv")
    ap.add_argument("--out_per_genre_csv", default="artist_genre_aggregates.csv")
    ap.add_argument("--out_coverage_json", default="coverage.json")
    args = ap.parse_args()

    outs = build_artist_vectors(
        args.artists_csv,
        args.tracks_csv,
        args.lastfm_csv,
        args.out_features_csv,
        args.out_per_genre_csv,
        args.out_coverage_json,
    )
    print("Saved:", outs)


if __name__ == "__main__":
    main()
