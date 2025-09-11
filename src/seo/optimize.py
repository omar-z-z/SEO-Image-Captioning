# this is the main file to optimize captions for SEO using a pre-trained language model
import json
from typing import List, Dict, Optional
import argparse
from pathlib import Path
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# _SEO_SYSTEM_PROMPT: a fixed instruction that teaches the model to keep one sentence, ~110–160 chars,
# front-load 1–2 keywords, no hashtags, natural grammar.
# Think of it as your style guide baked into the prompt.
_SEO_SYSTEM_PROMPT = (
    "Rewrite the image caption for SEO. Requirements:\n"
    "- Keep it natural and concise (110-160 chars when possible)\n"
    "- Front-load 1–2 important keywords, include others if natural\n"
    "- Avoid keyword stuffing; keep grammar clean\n"
    "- Use present-tense, active voice; no hashtags\n"
    "- Output ONE sentence only.\n"
)

def _build_prompt(caption: str, keywords: Optional[List[str]] = None) -> str:
    kws = ", ".join(keywords) if keywords else ""
    return (
        f"{_SEO_SYSTEM_PROMPT}\n"
        f"Caption: {caption}\n"
        f"Keywords: {kws}\n"
        f"SEO Caption:"
    )

class SeoRewriter:
    def __init__(self, model_name: str = "t5-small", device: Optional[str] = None, max_new_tokens: int = 48):
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def rewrite(self, caption: str, keywords: Optional[List[str]] = None) -> str:
        prompt = _build_prompt(caption, keywords)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            do_sample=False,
            num_beams=4,
            length_penalty=0.9,
            max_new_tokens=self.max_new_tokens,
            early_stopping=True,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        # Guardrails: enforce single sentence and trim
        text = text.replace("  ", " ").strip()
        if "." in text:
            text = text.split(".")[0].strip() + "."
        if len(text) > 180:
            text = text[:180].rsplit(" ", 1)[0].rstrip(".,;:") + "."
        return text

def load_base(base_json: Path) -> List[Dict]:
    with open(base_json, "r", encoding="utf-8") as f:
        return json.load(f)

def load_keywords(csv_path: Optional[Path]) -> Dict[str, List[str]]:
    if not csv_path or not csv_path.exists():
        return {}
    import csv
    m = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ks = [k.strip() for k in row["keywords"].split(",") if k.strip()]
            m[row["image_id"]] = ks
    return m

def run_optimize(
    base_json: Path,
    out_json: Path,
    keywords_csv: Optional[Path] = None,
    model_name: str = "t5-small",
):
    items = load_base(base_json)
    kw_map = load_keywords(keywords_csv)
    rewriter = SeoRewriter(model_name=model_name)
    out = []
    for it in items:
        img = it["image_id"]
        cap = it["caption"]
        kws = kw_map.get(img, None)
        seo = rewriter.rewrite(cap, kws)
        out.append({"image_id": img, "base_caption": cap, "keywords": kws or [], "seo_caption": seo})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/seo/flickr8kSEO.json")
    p.add_argument("--out",  default="outputs/captions_seo.json")
    p.add_argument("--keywords_csv", default="data/seo/keywords.csv")
    p.add_argument("--model", default="t5-small") 
    args = p.parse_args()

    run_optimize(
        base_json=Path(args.base),
        out_json=Path(args.out),
        keywords_csv=Path(args.keywords_csv) if args.keywords_csv else None,
        model_name=args.model,
    )

if __name__ == "__main__":
    main()
