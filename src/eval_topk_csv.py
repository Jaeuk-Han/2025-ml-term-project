import argparse, json, os
import numpy as np, pandas as pd

# Note: 파라미터(alpha) 튜닝을 위한 간단한 평가 코드입니다.
# Ps. 현재 alpha는 0.5~7 사이가 적절해보임

# Precision@K
# Top-K 추천 중 정답이 몇 개 들어갔는가.
def _precision_at_k(rel, k):
    k=min(k,len(rel)); return 0.0 if k<=0 else float(np.sum(rel[:k])/k)

# MAP@K (Mean Average Precision at K)
# 정답 하나가 나올 때마다 그 시점까지의 Precision을 누적 평균. 정답을 앞쪽에 더 많이 두면 올라감.
def _ap_at_k(rel, k):
    k=min(k,len(rel)); hits=0; ap=0.0; pos=max(1,int(np.sum(rel)))
    
    for i in range(k):
        if rel[i]>0: hits+=1; ap+=hits/(i+1.0)
    
    return float(ap/pos)

# NDCG@K (Normalized Discounted Cumulative Gain at K)
# 앞순위에 더 큰 가중을 주면서 이상적인 순서와 비교해 정규화한 점수.
def _ndcg_at_k(rel, k):
    k=min(k,len(rel)); 

    gains=(2**rel[:k]-1); disc=1.0/np.log2(np.arange(2,k+2))
    
    dcg=float(np.sum(gains*disc))
    
    ideal=np.sort(rel)[::-1][:k]; idcg=float(np.sum((2**ideal-1)*disc))
    
    return float(dcg/idcg) if idcg>0 else 0.0

# load label
def load_label_set(splits_dir, split):
    df=pd.read_csv(os.path.join(splits_dir,f"{split}.csv"),usecols=["user_id","artist_id"])
    df["user_id"]=df["user_id"].astype(int); df["artist_id"]=df["artist_id"].astype(int)

    return set(map(tuple, df[["user_id","artist_id"]].to_numpy()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="candidates or reranked csv/parquet")
    ap.add_argument("--splits_dir", required=True)
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    ap.add_argument("--score_col", default="final_score", help="als_score / final_score / ranker_score")
    ap.add_argument("--k", type=int, default=10)
    args=ap.parse_args()

    df=pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)
    
    if "final_score" not in df.columns and args.score_col=="final_score":
        raise SystemExit("final_score not found; pass --score_col als_score for ALS baseline")
    labels=load_label_set(args.splits_dir, args.split)

    if "label" not in df.columns:
        df["label"]=df.apply(lambda r: 1 if (int(r["user_id"]),int(r["artist_id"])) in labels else 0, axis=1)

    out=[]

    for uid,g in df.groupby("user_id"):
        g=g.sort_values(args.score_col, ascending=False)
        rel=g["label"].astype(int).values
        out.append([_precision_at_k(rel,args.k), _ap_at_k(rel,args.k), _ndcg_at_k(rel,args.k)])
    
    out=np.array(out)

    print(json.dumps({
        "k": args.k,
        "users": int(len(out)),
        "precision@k": float(out[:,0].mean() if len(out) else 0.0),
        "map@k": float(out[:,1].mean() if len(out) else 0.0),
        "ndcg@k": float(out[:,2].mean() if len(out) else 0.0),
    }, ensure_ascii=False, indent=2))

if __name__=="__main__":
    main()
