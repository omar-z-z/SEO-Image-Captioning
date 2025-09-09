import argparse
import json
from pathlib import Path

try:
    import nltk
    from nltk.translate.bleu_score import corpus_bleu
except Exception:
    nltk = None

def load_refs(manifest_path):
    manifest = json.load(open(manifest_path, "r", encoding="utf8"))
    return {it["image_id"]:[c.split() for c in it["captions"]] for it in manifest["items"]}

def load_preds(pred_path):
    preds = {}
    with open(pred_path, "r", encoding="utf8") as f:
        for line in f:
            obj = json.loads(line.strip())
            preds[obj["image_id"]] = obj["caption"].split()
    return preds

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred", default="outputs/predictions.jsonl")
    p.add_argument("--ref", default="data/manifests/flickr8k.json")
    args = p.parse_args()
    refs = load_refs(args.ref)
    preds = load_preds(args.pred)
    common = list(set(refs.keys()) & set(preds.keys()))
    common.sort()
    if not common:
        print("No overlapping image IDs between reference and predictions.")
        raise SystemExit(1)
    references = [refs[i] for i in common]
    hypotheses = [preds[i] for i in common]
    if nltk is None:
        print("NLTK not available. Install NLTK and download punkt to compute BLEU.")
        print("Sample predictions (first 5):")
        for i in common[:5]:
            print("ID:", i)
            print("REF:", [" ".join(r) for r in refs[i][:3]])
            print("PRED:", " ".join(preds[i]))
            print("-"*30)
    else:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            print("Downloading punkt tokenizers...")
            nltk.download("punkt")
        bleu = corpus_bleu(references, hypotheses)
        print(f"Corpus BLEU: {bleu:.4f}")
        print("\nSample predictions (first 5):")
        for i in common[:5]:
            print("ID:", i)
            print("REF:", [" ".join(r) for r in refs[i][:3]])
            print("PRED:", " ".join(preds[i]))
            print("-"*30)
