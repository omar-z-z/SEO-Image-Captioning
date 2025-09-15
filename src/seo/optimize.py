#!/usr/bin/env python3
"""
SEO Caption Optimizer

- Loads base captions (JSON) and optional per-image keywords (CSV).
- Rewrites each caption with a T5 model into one, natural SEO-friendly sentence.
- Front-loads a compact keyword phrase when natural.
- Writes a JSON list with: image_id, base_caption, keywords (used), seo_caption.
- Uses heuristics to extract keywords if not provided.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast

# --------------------------- Prompting ---------------------------------------------------------

_FEW_SHOTS = [
    {
        "cap": "A child in a pink dress is climbing up a set of stairs in an entry way.",
        "kws": ["child", "stairs"],
        "out": "Child on stairs in a bright entryway, pink dress drawing focus as she climbs with playful confidence."
    },
    {
        "cap": "A black dog and a white dog with brown spots are staring at each other in the street.",
        "kws": ["dogs", "street"],
        "out": "Dogs facing off in the street, a black pup and a white one with brown spots locking an alert, curious gaze."
    },
]

_SEO_SYSTEM_PROMPT = (
    "You rewrite image captions for SEO.\n"
    "Rules:\n"
    "- Natural, concise (aim 110–160 chars)\n"
    "- Front-load 1–2 important keywords when natural\n"
    "- No keyword stuffing, clean grammar\n"
    "- Present tense, active voice, no hashtags\n"
    "- Write exactly one sentence.\n"
    "Do not mention these rules or the words 'one sentence' in your output."
)

def _build_prompt(caption: str, keywords: Optional[List[str]] = None) -> str:
    # Few-shot prompt that nudges the model toward our style.
    kws = ", ".join(keywords) if keywords else "none"
    fewshot_block = "\n".join(
        "Input caption: " + ex["cap"] + "\n"
        "Keywords: " + (", ".join(ex["kws"]) if ex["kws"] else "none") + "\n"
        "Output: <answer>" + ex["out"] + "</answer>\n"
        for ex in _FEW_SHOTS
    )
    return (
        f"{_SEO_SYSTEM_PROMPT}\n\n"
        f"{fewshot_block}\n"
        f"Input caption: {caption}\n"
        f"Keywords: {kws}\n"
        "Output: <answer>"
    )

# --------------------------- Heuristics --------------------------------------------------------

STOPWORDS = {
    "a","an","the","and","or","of","in","on","with","to","at","for","from","by","as","is","are","was","were",
    "this","that","these","those","it","its","his","her","their","our","your","my","up","down","over","under"
}

# common color words—usually not helpful as leading "keywords"
COLORS = {
    "black","white","brown","pink","red","green","blue","yellow","orange","purple",
    "grey","gray","gold","silver","beige","tan","teal","maroon","navy"
}

def _clean_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", text)

def _extract_keywords(caption: str) -> List[str]:
    """
    Try to form a compact phrase (preferred) like 'Child on stairs' or 'Dogs in street'.
    If not possible, fall back to 1–2 meaningful nouns (skipping stopwords/colors).
    Returns Title-case phrases for nicer leading.
    """
    words = _clean_tokens(caption)

    # 1) Pattern: <subject> (in|on|at|by|under|near|beside|before) <place/object>
    m = re.search(
        r"\b([A-Za-z][A-Za-z\-']+)\s+(in|on|at|by|under|near|beside|before)\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z\-']+)",
        caption, flags=re.IGNORECASE
    )
    if m:
        subj, prep, obj = (m.group(1).lower(), m.group(2).lower(), m.group(3).lower())
        if subj not in STOPWORDS | COLORS and obj not in STOPWORDS | COLORS:
            return [f"{subj.capitalize()} {prep} {obj}"]

    # 2) Verb-ing hook: 'climbing', 'finger-painting', ...
    m2 = re.search(r"\b([A-Za-z][A-Za-z\-']+ing)\b", caption)
    if m2:
        v = m2.group(1).lower()
        if v not in STOPWORDS and v not in COLORS:
            nouns = [w.lower() for w in words if w.lower() not in STOPWORDS | COLORS and len(w) > 2]
            subj = nouns[0].capitalize() if nouns else ""
            if subj:
                return [f"{subj} {v}"]

    # 3) Fallback: first 1–2 non-trivial nouns
    nouns: List[str] = []
    for w in (w.lower() for w in words):
        if w in STOPWORDS or w in COLORS or len(w) < 3:
            continue
        if w not in nouns:
            nouns.append(w)
        if len(nouns) >= 2:
            break
    if nouns:
        return [f"{nouns[0].capitalize()} in {nouns[1]}"] if len(nouns) >= 2 else [nouns[0].capitalize()]

    return []

def _frontload(sentence: str, keywords: List[str]) -> str:
    """
    If we have a compact keyword phrase, prepend it plus an em dash.
    Otherwise just capitalize the sentence.
    """
    s = re.sub(r"\s+", " ", sentence).strip()
    if not s:
        return s
    if keywords:
        lead = keywords[0].strip().rstrip(".,;:")
        if not s.lower().startswith(lead.lower()):
            s = f"{lead} — {s}"
    if s and not s[0].isupper():
        s = s[0].upper() + s[1:]
    return s

def _polish(sentence: str, max_len: int = 180) -> str:
    # Enforce single-sentence, tidy punctuation, and max length.
    s = re.sub(r"\s+", " ", sentence).strip()
    s = (s.split(".")[0] + ".") if "." in s else (s.rstrip("!?,;:") + ".")
    if len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0].rstrip(".,;:") + "."
    return s

# --------------------------- Model wrapper -----------------------------------------------------

class SeoRewriter:
    def __init__(self, model_name: str = "google/flan-t5-small",
                 device: Optional[str] = None, max_new_tokens: int = 64):
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def _generate(self, prompt: str, sample: bool = False) -> str:
        # Single call to T5.generate + answer tag stripping + light cleanup.
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        out = self.model.generate(
            **inputs,
            do_sample=sample,
            top_p=0.9 if sample else None,
            temperature=0.8 if sample else None,
            num_beams=4 if not sample else 1,
            no_repeat_ngram_size=3,
            length_penalty=0.9,
            max_new_tokens=self.max_new_tokens,
            early_stopping=True,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        text = (m.group(1) if m else text).strip()
        return re.sub(r"^[\s\-\:\.\•]+", "", text)  # strip bullets/dashes

    @torch.inference_mode()
    def rewrite(self, caption: str, keywords: Optional[List[str]] = None) -> tuple[str, List[str]]:
        """
        Returns (seo_sentence).

        - If external keywords are provided, use them.
        - Otherwise auto-extract a compact phrase and use that.
        - Second pass (sampled) only if the first pass is too short.
        """
        auto_kws = _extract_keywords(caption)
        kws_used = keywords if (keywords and len(keywords) > 0) else auto_kws

        # Pass A: few-shot prompt, deterministic
        s1 = self._generate(_build_prompt(caption, kws_used), sample=False)
        s1 = _polish(_frontload(s1, kws_used))

        # If too short, try a more forceful paraphrase with sampling
        if len(s1) < 100:
            p2 = (
                "Paraphrase the caption for SEO as one natural sentence (110–160 chars). "
                "Vary the wording; do not echo the original phrasing. "
                "Front-load 1–2 keywords if natural; no hashtags.\n"
                f"Caption: {caption}\n"
                f"Keywords: {', '.join(kws_used) if kws_used else 'none'}\n"
                "Output only the sentence between <answer> and </answer>.\n<answer>"
            )
            s2 = self._generate(p2, sample=True)
            s2 = _polish(_frontload(s2, kws_used))
            if len(s2) >= 100:
                return s2

        return s1

# --------------------------- I/O --------------------------------------------------------------

def load_base(base_json: Path) -> List[Dict]:
    with base_json.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_keywords(csv_path: Optional[Path]) -> Dict[str, List[str]]:
    """
    CSV schema:
      image_id,keywords
      1000268201_693b08cb0e.jpg,"child, stairs"
    """
    if not csv_path or not csv_path.exists():
        return {}
    kw_map: Dict[str, List[str]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ks = [k.strip() for k in row.get("keywords", "").split(",") if k.strip()]
            if row.get("image_id"):
                kw_map[row["image_id"]] = ks
    return kw_map

def run_optimize(
    base_json: Path,
    out_json: Path,
    keywords_csv: Optional[Path] = None,
    model_name: str = "google/flan-t5-small",
):
    items = load_base(base_json)
    kw_map = load_keywords(keywords_csv)
    rewriter = SeoRewriter(model_name=model_name)

    out = []
    for it in tqdm(items, desc="Optimizing captions", unit="caption"):
        img = it["image_id"]
        cap = it["caption"]
        manual_kws = kw_map.get(img, None)

        seo_sentence = rewriter.rewrite(cap, manual_kws)

        out.append({
            "image_id": img,
            "base_caption": cap,
            "keywords": [],   # <-- always include the actually used keywords
            "seo_caption": seo_sentence
        })
        # if len(out) == 50:
        #     break  # FOR DEBUG

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

# --------------------------- CLI --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/seo/flickr8kSEO.json")
    p.add_argument("--out",  default="outputs/captions_seo.json")
    p.add_argument("--keywords_csv", default="data/seo/keywords.csv")
    p.add_argument("--model", default="google/flan-t5-small")
    args = p.parse_args()

    run_optimize(
        base_json=Path(args.base),
        out_json=Path(args.out),
        keywords_csv=Path(args.keywords_csv) if args.keywords_csv else None,
        model_name=args.model,
    )

if __name__ == "__main__":
    main()
