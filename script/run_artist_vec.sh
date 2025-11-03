#!/usr/bin/env bash
# root에서 bash run_artist_vec.sh로 실행

set -euo pipefail

poetry run python src/build_artist_vectors.py \
  --artists_csv data/spotify/artists.csv \
  --tracks_csv  data/spotify/tracks.csv \
  --lastfm_csv  data/lastfm/Last.fm_data.csv \
  --out_features_csv   data/derived/spotify_full/artist_features_weighted.csv \
  --out_per_genre_csv  data/derived/spotify_full/artist_genre_aggregates.csv \
  --out_coverage_json  data/derived/spotify_full/coverage.json
