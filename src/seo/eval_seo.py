# this file evaluates the SEO quality of generated captions using heuristic metrics
import json
from statistics import mean
try:
    from src.seo.metrics import score_record
except ModuleNotFoundError:
    from metrics import score_record

data = json.load(open("outputs/captions_seo.json", "r", encoding="utf-8"))
scores = [score_record(r) for r in data]

def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return round(mean(vals), 3) if vals else float("nan")
print("SEO quality (heuristic):")
print("  keyword_recall :", _safe_mean([s["kw_recall"]    for s in scores]))
print("  frontload_score:", _safe_mean([s["kw_frontload"] for s in scores]))
print("  length_score   :", _safe_mean([s["len_score"]    for s in scores]))