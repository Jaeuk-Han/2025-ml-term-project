## 🗂️ Directory Structure

```text
ML_SHARE/
├─ data/                      # data 폴더 (용량 문제로 미포함)
│  ├─ lastfm/                 # 원본 Last.fm
│  │  └─ Last.fm_data.csv
│  ├─ spotify/                # 원본 Spotify (artists.csv, tracks.csv 등)
│  │  ├─ artists.csv
│  │  └─ tracks.csv
│  ├─ spotify_sample/         # 소용량 샘플
│  │  ├─ artists_sample.csv
│  │  └─ tracks_sample.csv
│  └─ derived/                # 모든 산출물(자동 생성)
│     ├─ spotify_full/        # build_artist_vectors.py 결과
│     ├─ lastfm_join_strict/  # join_lastfm_spotify.py 결과
│     └─ als_* / …            # train_implicit_als.py 결과(모델/지표/분할 등)
│
├─ script/                    # 실행 스크립트(파이프라인 4단계)
│  ├─ run_artist_vec.sh       # 1. Spotify → 아티스트 벡터
│  ├─ run_join.sh             # 2. Last.fm × Spotify 조인
│  ├─ run_train.sh            # 3. ALS 학습/평가
│  └─ run_infer.sh            # 4. 단일 유저 Top-K 추론
│
├─ src/
│  ├─ build_artist_vectors.py # Spotify artists/tracks > 장르가중 아티스트 벡터
│  ├─ join_lastfm_spotify.py  # Last.fm 집계 + Spotify 피처 조인
│  ├─ train_implicit_als.py   # ALS 학습/검증/테스트 + 옵션(BM25/TF-IDF, 코사인, pop-blend)
│  └─ predict_user.py         # 저장된 팩터로 특정 유저 Top-K 추천
│
├─ README.md
└─ pyproject.toml             # poetry 환경 (의존성 확인용)

```