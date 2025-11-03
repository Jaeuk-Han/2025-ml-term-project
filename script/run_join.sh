#!/usr/bin/env bash
# root에서 bash run_join.sh로 실행
set -euo pipefail

poetry run python src/join_lastfm_spotify.py \
  --lastfm_csv data/lastfm/Last.fm_data.csv \
  --spotify_features_csv data/derived/spotify_full/artist_features_weighted.csv \
  --out_dir data/derived/lastfm_join_strict \
  --min_plays 1 \
  --normalize basic