# this file evaluates the SEO quality of generated captions using heuristic metrics
import json
from statistics import mean
from src.seo.metrics import score_record

data = json.load(open("outputs/captions_seo.json", "r", encoding="utf-8"))
scores = [score_record(r) for r in data]
print("SEO quality (heuristic):")
print("  keyword_recall :", round(mean(s["kw_recall"]   for s in scores), 3))
print("  frontload_score:", round(mean(s["kw_frontload"]for s in scores), 3))
print("  length_score   :", round(mean(s["len_score"]   for s in scores), 3))
