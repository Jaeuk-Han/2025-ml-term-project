# 🎵 ML Term Project — Music Recommendation & Personalized Playlist 🎵

## 1) Overview
본 프로젝트는 <strong>Last.fm 재생 로그</strong>와 <strong>Spotify 메타/오디오 피처</strong>를 결합하여<br>
<strong>2-Stage 추천(ALS → LGBM LambdaRank 리랭커)</strong>로 유저별 아티스트를 추천하고,<br>
노트북에서 <strong>tempo/valence 취향과 상황(컨텍스트)</strong>을 반영해 <strong>개인화 플레이리스트(트랙 단위)</strong>를 생성합니다.<br>
최종 결과는 간단한 <strong>GUI</strong>(노트북 위젯)로 출력하여 사용자의 이해를 돕습니다.

## 2) Notes on Design
- **암묵적 피드백**: 미관찰=비선호 아님 → confidence(α), BM25, log1p로 신호 안정화  
- **Cosine score, pop_blend**: 희소/롱테일 환경에서 랭킹 일관성 및 안전성 강화  
- **리랭커 피처**: 유저×아이템 코사인/L2/absdiff, log-pop, ALS×cos 상호작용  
- **플레이리스트 확장**: 아티스트 추천을 트랙으로 확장, tempo/valence 유사도 반영

---

## 3) Final Scoring Design (Weighted Average)
최종 점수는 **ALS + 리랭커**의 결합 점수(Base)에 **tempo/valence 유사도**를 가중 평균으로 반영합니다.

$$\mathrm{FinalScore}=0.6\,\mathrm{Base\_score}+0.2\,\mathrm{Tempo\_sim}+0.2\,\mathrm{Valence\_sim}$$



| 구성요소        | 의미                                         | 비중 |
|----------------|----------------------------------------------|:---:|
| **Base_score** | ALS + 리랭커(α=0.5) 결합 점수                | 0.6 |
| **Tempo_sim**  | 유저 tempo 프로필과 트랙 tempo의 유사도      | 0.2 |
| **Valence_sim**| 유저 valence 프로필과 트랙 valence의 유사도  | 0.2 |

> 기본 비중은 다음과 같으며, 상황(운동/집중/휴식 등)에 따라 조정 가능합니다.

---

## 4) Pipeline (요약)
1) **Spotify → 아티스트 벡터 집계** (트랙 피처의 가중 평균)  
2) **Last.fm × Spotify 조인/정합**  
3) **ALS(implicit)** 학습(BM25, log1p, cosine score, pop_blend 옵션)  
4) **ALS 후보 Top-N** 덤프(Seen 제거)  
5) **유저/아이템 피처** 생성(오디오/장르/인기도, 유저 프로필)  
6) **리랭커 데이터셋** 구성 및 **LGBM LambdaRank** 학습  
7) **리랭크 추론 + α 블렌딩(기본 α=0.5)** → 최종 Top-K (아티스트)  
8) (노트북) **아티스트→트랙 확장 + tempo/valence 반영** → **개인화 플레이리스트**

---

## 5) Model Results (Test)

### 베이스라인 (ALS only)
- **Precision@10**: 0.5091  
- **MAP@10**: 0.3885  
- **NDCG@10**: 0.5751  

### 리랭커 + 블렌딩 (α=0.5)
- **Precision@10**: 0.5182  
- **MAP@10**: 0.7865  
- **NDCG@10**: 0.8980  

### 향상폭 (α=0.5 vs. ALS)

| Metric        | ALS    | Rerank(α=0.5) | Absolute Δ | Relative Δ |
|---|---:|---:|---:|---:|
| Precision@10  | 0.5091 | 0.5182 | +0.0091 | +1.8% |
| MAP@10        | 0.3885 | 0.7865 | +0.3981 | +102.5% |
| NDCG@10       | 0.5751 | 0.8980 | +0.3230 | +56.1% |

### α 블렌딩 스윕 (K=10)

| α   | Precision@10 | MAP@10 | NDCG@10 |
|---:|-------------:|-------:|--------:|
| 0.3 | 0.4727 | 0.7296 | 0.8420 |
| 0.4 | 0.4909 | 0.7789 | 0.8929 |
| **0.5** | **0.5182** | **0.7865** | **0.8980** |
| 0.6 | 0.5091 | 0.7856 | 0.8977 |
| 0.7 | 0.4909 | 0.7744 | 0.8932 |

> 해석: **α=0.5**에서 균형이 가장 우수하며, MAP/NDCG 개선 폭이 커서 정답 상위 노출 효과가 큼.


### 주요 관찰점
- **ALS 한계:** 정확도가 너무 안나와서 조사한 결과 ALS는 "사용자×아이템 **공발생(co‑occurrence)**"을 많이 볼수록 임베딩이 좋아지는데, 저희 데이터에 아티스트는 많지만 유저가 11명뿐이라 그래프가 **너무 희소**해서 모델이 아이템 간 관계를 풍부하게 못 배우는 것 같습니다.
- **리랭커 효과:** 부족한 정확도를 보완하고자 콘텐츠 기반 사이드 피처(오디오/장르/인기도)와 유저 프로필(가중 평균 + 장르 Top‑K/엔트로피)을 써서 순서를 다시 매기는 리랭커 개념을 도입했습니다.(RAG에서의 개념과 유사) 결론적으로 **상위 랭크 품질(nDCG/MAP)이 크게 개선**되었음을 확인 가능했습니다.

---

### 주요 방법론

- **BM25 가중치:** 양 극단의 값들 즉, heavy user나 초인기 아이템 편향을 눌러서 공정한 신호로 학습하게 하는 것을 통해 희소 데이터에서 순위 안정화에 도움.
- **Cosine 스코어:** 내적 대신 코사인 유사도로 점수화(벡터 크기 영향 제거)를 통한 일관된 랭킹.
- **log1p_plays:** 재생수에 로그 스케일 적용하는 것을 통해 과도한 카운트를 압축하고 일반화 향상, 쏠림 완화.
- **pop_blend=0.1:** 최종 점수에 인기도를 10% 섞어 비정상 상위 노출 완충.
- **유저 프로필:** 오디오 피처 **plays 가중 평균** + **장르 Top‑K/엔트로피(취향 집중도)**  
- **아이템 피처:** 오디오/장르/explicit + **인기도 통계(유니크 유저 수, 총 재생수)**  
- **파생 피처:** 유저–아이템 **코사인 유사도/L2/절대차 평균**, **ALS rank 역수**, **log(pop)**, **ALS×cos 상호작용**  
- **리랭커:** LightGBM(LambdaRank) + **α 블렌딩**으로 협업(ALS)과 콘텐츠를 **균형 결합**

---

## 6) How to Run

### (A) CLI Quickstart
```bash
bash run_all.sh # 전체 모델링 과정 파이프라인 스크립트 (poetry 기반)
```

### (B) Notebook (GUI) — `ML_final.ipynb`
- 입력: `data/derived/rerank/test_reranked_topk_a05.csv`, `data/spotify/artists.csv`
- 가중치(Base/Tempo/Valence) 및 상황 옵션 설정 → **플레이리스트 생성**(트랙 리스트 표시)

---

## 7) 🗂️Directory Structure

```text
~/Project/ml_term_project/
├─ data/
│  ├─ lastfm/
│  │  └─ Last.fm_data.csv                      # 원본 Last.fm 스크로블 로그
│  ├─ spotify/
│  │  ├─ artists.csv                           # 원본 Spotify 아티스트 메타
│  │  └─ tracks.csv                            # 원본 Spotify 트랙/오디오 피처
│  ├─ spotify_sample/
│  │  ├─ artists_sample.csv                    # 소용량 샘플(디버깅)
│  │  └─ tracks_sample.csv
│  └─ derived/
│     ├─ spotify_full/                         # (1) Spotify → 아티스트 벡터
│     │  └─ artist_features_weighted.csv       # 아티스트 단위 집계 피처
│     ├─ lastfm_join_strict/                   # (2) Last.fm × Spotify 조인(선택)
│     │  ├─ joined.csv                         # 규격화된 user–artist–plays 테이블
│     │  └─ joined_with_features.csv           # joined + 일부 side feature
│     ├─ als/                                  # (3) ALS 학습/분할/매핑
│     │  ├─ als_model.npz                      # 학습된 ALS 팩터(유저/아이템)
│     │  ├─ metrics.json                       # train/val/test 지표 요약
│     │  ├─ splits/                            # 데이터 분할(학습/평가 기준)
│     │  │  ├─ train.csv
│     │  │  ├─ val.csv
│     │  │  └─ test.csv
│     │  ├─ artist_mapping.csv                 # artist_id ↔ Artist(이름) 맵
│     │  └─ user_mapping.csv                   # user_id ↔ Username 맵
│     ├─ features/                              # (4) 리랭커용 유저/아이템 피처
│     │  ├─ item_features.csv                   # 아이템: 오디오/메타 + 인기도(pop)
│     │  ├─ user_profiles.csv                   # 유저: plays-가중 평균 + 장르 top-k
│     │  └─ features_meta.json                  # 피처 스키마/키/주의사항 메타
│     ├─ candidates/                            # (5) ALS 후보 덤프(리랭커 입력)
│     │  ├─ val_candidates.parquet              # user_id, artist_id, als_score, …
│     │  └─ test_candidates.parquet
│     └─ rerank/                                # (6) 리랭커 학습/추론/평가 결과
│        ├─ val_dataset.parquet                 # 후보+피처(join) 학습 테이블(VAL)
│        ├─ test_dataset.parquet                # 후보+피처(TEST)
│        ├─ val_dataset.features.json           # 리랭커 입력 피처 목록/그룹 정의
│        ├─ lgbm/
│        │  ├─ lgbm_reranker.pkl                # 학습된 LGBM LambdaRank 모델
│        │  ├─ report.json                      # train/eval 지표/설정
│        │  └─ feature_importance.csv           # 피처 중요도
│        ├─ test_reranked_topk_a03.csv          # α=0.3 블렌딩 Top-K(최종 점수)
│        ├─ test_reranked_topk_a04.csv
│        ├─ test_reranked_topk_a05.csv
│        ├─ test_reranked_topk_a06.csv
│        ├─ test_reranked_topk_a07.csv
│        ├─ test_reranked_topk_best.csv         # 스윕 중 best(선택)
│        └─ alpha_sweep_k10.jsonl               # α 스윕 평가 로그(k=10)
│
├─ script/                                      # 실행 스크립트
│  ├─ run_artist_vec.sh                         # 1. Spotify → 아티스트 벡터
│  ├─ run_join.sh                               # 2. Last.fm × Spotify 조인
│  ├─ run_train_als.sh                          # 3. ALS 학습/평가/분할 생성
│  ├─ run_dump_candidates.sh                    # 4. ALS 후보 TopN 덤프
│  ├─ run_build_features.sh                     # 5. 유저/아이템 피처 생성
│  ├─ run_make_rerank_dataset.sh                # 6. 리랭커 학습 데이터 테이블 생성
│  ├─ run_train_reranker.sh                     # 7. LGBM 리랭커 학습/리포트
│  ├─ run_rerank_infer.sh                       # 8. 리랭크+ALS 블렌딩 Top-K 생성
│  └─ run_eval_sweep.sh                         # 9. α 스윕 평가(JSONL 집계/정렬)
│
├─ src/                                         # 단계별 파이썬 스크립트
│  ├─ build_artist_vectors.py                   # Spotify → 장르가중 아티스트 벡터
│  ├─ join_lastfm_spotify.py                    # Last.fm 집계 + Spotify 조인
│  ├─ train_implicit_als.py                     # ALS 학습/검증/테스트 + 매핑 저장
│  ├─ predict_user.py                           # ALS 팩터 기반 단일 유저 추천
│  ├─ build_user_item_features.py               # 리랭커용 item/user 피처 생성
│  ├─ dump_candidates_from_als.py               # ALS 후보(topN) 덤프
│  ├─ make_rerank_dataset.py                    # 후보+피처→리랭커 학습 테이블
│  ├─ train_reranker_lgbm.py                    # LGBM(LambdaRank) 학습/리포트
│  ├─ rerank_infer.py                           # 리랭커+ALS 블렌딩 추론(topK)
│  ├─ eval_topk_csv.py                          # CSV/Parquet Top-K 평가(Precision/MAP/NDCG)
│  └─ ML_final.ipynb                            # 최종 점수 산출 후 플레이리스트 GUI 출력
│
├─ run_all.sh                                   # 전체 파이프라인 실행코드
├─ README.md                                    # 파이프라인/명령어/결과 요약
└─ pyproject.toml                               # poetry 의존성
```

---

## 8) reference

[Yifan Hu, Yehuda Koren, Chris Volinsky. [IEEE] Collaborative Filtering for Implicit Feedback Datasets](https://ieeexplore.ieee.org/document/4781121)

[LightGBM Documentation](https://lightgbm.readthedocs.io/en/stable/)

[implicit (benfred) GitHub](https://benfred.github.io/implicit/)

[Tech Blog posts on implicit CF & LambdaRank 1](https://medium.com/radon-dev/als-implicit-collaborative-filtering-5ed653ba39fe)

[Tech Blog posts on implicit CF & LambdaRank 2](https://blog.reachsumit.com/posts/2022/09/explicit-implicit-cf/)