import math
from typing import List, Dict

def keyword_recall(pred: str, keywords: List[str]) -> float:
    if not keywords: return 1.0
    pred_l = pred.lower()
    hit = sum(1 for k in keywords if k.lower() in pred_l)
    return hit / len(keywords)

def keyword_frontload_score(pred: str, keywords: List[str]) -> float:
    if not keywords: return 1.0
    pred_l = pred.lower()
    first_70 = pred_l[:70]
    hits_front = sum(1 for k in keywords if k.lower() in first_70)
    return hits_front / max(1, len(keywords))

def length_score(pred: str, lo: int = 110, hi: int = 160) -> float:
    n = len(pred)
    if lo <= n <= hi: return 1.0
    # smooth penalty
    dist = min(abs(n - lo), abs(n - hi)) if n < lo or n > hi else 0
    return max(0.0, 1.0 - dist/80.0)

def score_record(rec: Dict) -> Dict:
    pred = rec["seo_caption"]
    kws  = rec.get("keywords", [])
    return {
        "image_id": rec["image_id"],
        "kw_recall": keyword_recall(pred, kws),
        "kw_frontload": keyword_frontload_score(pred, kws),
        "len_score": length_score(pred),
    }
