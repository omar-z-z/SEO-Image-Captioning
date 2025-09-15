# this is the main file to optimize captions for SEO using a pre-trained language model
import json
from typing import List, Dict, Optional
import argparse
from pathlib import Path
import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from tqdm import tqdm
import re
import difflib

# _SEO_SYSTEM_PROMPT: a fixed instruction that teaches the model to keep one sentence, ~110–160 chars,
# front-load 1–2 keywords, no hashtags, natural grammar.
# Think of it as your style guide baked into the prompt.
# _SEO_SYSTEM_PROMPT = (
#     "You are an assistant that rewrites image captions for SEO.\n"
#     "Requirements:\n"
#     "• Keep it natural and concise (aim 110–160 chars)\n"
#     "• Front-load 1–2 important keywords; include others only if natural\n"
#     "• Avoid keyword stuffing; clean grammar\n"
#     "• Present tense, active voice; no hashtags\n"
#     "• Output ONE sentence only\n"
# )

# def _build_prompt(caption: str, keywords: Optional[List[str]] = None) -> str:
#     kws = ", ".join(keywords) if keywords else ""
#     return (
#         f"{_SEO_SYSTEM_PROMPT}\n"
#         f"Caption: {caption}\n"
#         f"Keywords: {kws}\n"
#         f"SEO Caption:"
#     )
# def _build_prompt(caption: str, keywords: Optional[List[str]] = None) -> str:
#     kws = ", ".join(keywords) if keywords else "none"
#     return (
#         f"Instruction:\n{_SEO_SYSTEM_PROMPT}\n\n"
#         f"Input caption:\n{caption}\n\n"
#         f"Keywords (optional):\n{kws}\n\n"
#         "Output:\n"
#         "Return ONLY the rewritten sentence between <answer> and </answer>.\n"
#         "<answer>"
#     )
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
    }
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
    kws = ", ".join(keywords) if keywords else "none"
    examples = []
    for ex in _FEW_SHOTS:
        examples.append(
            "Input caption: " + ex["cap"] + "\n"
            "Keywords: " + (", ".join(ex["kws"]) if ex["kws"] else "none") + "\n"
            "Output: <answer>" + ex["out"] + "</answer>\n"
        )
    fewshot_block = "\n".join(examples)

    return (
        f"{_SEO_SYSTEM_PROMPT}\n\n"
        f"{fewshot_block}\n"
        f"Input caption: {caption}\n"
        f"Keywords: {kws}\n"
        "Output: <answer>"
    )
    

    
def _looks_bad(s: str) -> bool:
    s_clean = s.lower().strip()
    if not s_clean or len(s_clean) < 12:
        return True
    if re.fullmatch(r"(one\s+sentence\.?)", s_clean):
        return True
    # discourage echoes of meta words
    if any(w in s_clean for w in ["instruction:", "output:", "<answer", "keywords:", "rules:"]):
        return True
    return False

def _rule_based_fallback(caption: str, keywords: Optional[List[str]]) -> str:
    # very small polish: trim, collapse spaces, lead with a keyword if possible
    cap = re.sub(r"\s+", " ", caption).strip().rstrip(".")
    lead = ""
    if keywords:
        # pick the first short keyword as a lead
        lead = keywords[0].strip()
        if lead and lead.lower() not in cap.lower():
            cap = f"{lead.capitalize()} — {cap}"
    sent = cap[0].upper() + cap[1:] + "."
    if len(sent) > 180:
        sent = sent[:180].rsplit(" ", 1)[0].rstrip(".,;:") + "."
    return sent

STOPWORDS = {
    "a","an","the","and","or","of","in","on","with","to","at","for","from","by","as","is","are","was","were",
    "this","that","these","those","it","its","his","her","their","our","your","my","up","down","over","under"
}

def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.lower().strip(), b=b.lower().strip()).ratio()

def _extract_keywords(caption: str, k: int = 2):
    # very light heuristic: take first 2 non-stopword nouns-ish tokens
    words = re.findall(r"[A-Za-z][A-Za-z\-']+", caption)
    cand = [w.lower() for w in words if w.lower() not in STOPWORDS and len(w) > 2]
    # dedupe while preserving order
    seen = set(); out=[]
    for w in cand:
        if w not in seen:
            out.append(w); seen.add(w)
        if len(out) >= k: break
    # return capitalized for leading
    return [w.capitalize() for w in out]

def _frontload(seo: str, keywords):
    if not keywords: 
        return seo[0].upper() + seo[1:] if seo else seo
    lead = ", ".join(keywords[:2])
    # If already starts with a keyword, just capitalize
    starts_with_kw = any(seo.lower().startswith(k.lower()) for k in keywords[:2])
    if starts_with_kw:
        s = seo
    else:
        s = f"{lead} — {seo.lstrip()}"
    # Clean spacing/case
    s = re.sub(r"\s+", " ", s).strip()
    if s and not s[0].isupper():
        s = s[0].upper() + s[1:]
    return s

def _polish(seo: str, min_len: int = 90, max_len: int = 180):
    s = re.sub(r"\s+", " ", seo).strip()
    # force single sentence
    if "." in s:
        s = s.split(".")[0].strip() + "."
    else:
        s = s.rstrip("!?,;:") + "."
    # length bounds
    if len(s) > max_len:
        s = s[:max_len].rsplit(" ", 1)[0].rstrip(".,;:") + "."
    return s

import re, difflib

# Keep your STOPWORDS set; add common color words so they won't lead the sentence
COLORS = {
    "black","white","brown","pink","red","green","blue","yellow","orange","purple",
    "grey","gray","gold","silver","beige","tan","teal","maroon","navy"
}

def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a.lower().strip(), b=b.lower().strip()).ratio()

def _clean_tokens(text: str):
    return re.findall(r"[A-Za-z][A-Za-z\-']+", text)

def _extract_keywords(caption: str):
    """
    Prefer a compact phrase like 'Child on stairs', 'Dogs in street', 'Girl finger-painting'.
    Falls back to 1–2 meaningful nouns (not colors / stopwords).
    """
    words = _clean_tokens(caption)
    lw = [w.lower() for w in words]

    # 1) Try to capture a simple pattern: <subject> (in|on|at|by|under|near) <place/object>
    m = re.search(
        r"\b([A-Za-z][A-Za-z\-']+)\s+(in|on|at|by|under|near|beside|before)\s+(?:a|an|the)?\s*([A-Za-z][A-Za-z\-']+)",
        caption, flags=re.IGNORECASE
    )
    if m:
        subj, prep, obj = m.group(1).lower(), m.group(2).lower(), m.group(3).lower()
        if subj not in STOPWORDS | COLORS and obj not in STOPWORDS | COLORS:
            # pluralize heuristic: if subj ends with 's' already, keep it
            subj_cap = subj.capitalize()
            obj_cap  = obj if obj.endswith('s') else obj
            return [f"{subj_cap} {prep} {obj}"]

    # 2) Verb-ing hook: pick a leading activity phrase (e.g., 'finger-painting', 'climbing')
    m2 = re.search(r"\b([A-Za-z][A-Za-z\-']+ing)\b", caption)
    if m2:
        v = m2.group(1).lower()
        if v not in STOPWORDS and v not in COLORS:
            # try to find a subject near the verb (previous noun)
            nouns = [w for w in lw if w not in STOPWORDS | COLORS and len(w) > 2]
            subj = nouns[0].capitalize() if nouns else ""
            if subj:
                return [f"{subj} {v}"]

    # 3) Fallback: first 1–2 meaningful nouns (skip colors/stopwords)
    nouns = []
    for w in lw:
        if w in STOPWORDS or w in COLORS or len(w) < 3:
            continue
        if w not in nouns:
            nouns.append(w)
        if len(nouns) >= 2:
            break
    if nouns:
        # Craft a compact phrase if we have two nouns: 'Dogs street' -> 'Dogs in street'
        if len(nouns) >= 2:
            return [f"{nouns[0].capitalize()} in {nouns[1]}"]
        return [nouns[0].capitalize()]

    return []  # nothing found

def _frontload(seo: str, keywords):
    """
    Use a single concise phrase as the lead (no commas), then an em dash.
    Example: 'Child on stairs — …' or just capitalize sentence if no keywords.
    """
    s = re.sub(r"\s+", " ", seo).strip()
    if not s:
        return s
    if keywords:
        lead = keywords[0].strip().rstrip(".,;:")  # one compact phrase
        # Avoid duplicate lead if the sentence already starts with it (case-insensitive)
        if not s.lower().startswith(lead.lower()):
            s = f"{lead} — {s}"
    # Capitalize first letter of the sentence
    if s and not s[0].isupper():
        s = s[0].upper() + s[1:]
    return s



class SeoRewriter:
    def __init__(self, model_name: str = "google/flan-t5-small",
                 device: Optional[str] = None, max_new_tokens: int = 48):
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def rewrite(self, caption: str, keywords: Optional[List[str]] = None) -> str:
        # ensure we always have some keywords for front-loading
        auto_kws = _extract_keywords(caption)
        kws = (keywords or []) or auto_kws

        def _gen(prompt: str, sample: bool = False) -> str:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            out = self.model.generate(
                **inputs,
                do_sample=sample,
                top_p=0.9 if sample else None,
                temperature=0.8 if sample else None,
                num_beams=4 if not sample else 1,
                no_repeat_ngram_size=3,
                length_penalty=0.9,
                max_new_tokens=max(self.max_new_tokens, 64),
                early_stopping=True,
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
            text = (m.group(1) if m else text).strip()
            text = re.sub(r"^[\s\-\:\.\•]+", "", text)
            return text

        # Pass A: current few-shot prompt
        p1 = _build_prompt(caption, kws)
        s1 = _gen(p1, sample=False)
        s1 = _polish(_frontload(s1, kws))

        # If too similar to base or very short, try a second, more forceful pass
        if _similar(s1, caption) >= 0.80 or len(s1) < 100:
            p2 = (
                "Paraphrase the caption for SEO as one natural sentence (110–160 chars). "
                "Vary the wording significantly; do not echo the original phrasing. "
                "Front-load 1–2 keywords if natural; no hashtags.\n"
                f"Caption: {caption}\n"
                f"Keywords: {', '.join(kws) if kws else 'none'}\n"
                "Output only the sentence between <answer> and </answer>.\n<answer>"
            )
            s2 = _gen(p2, sample=True)
            s2 = _polish(_frontload(s2, kws))
            # pick the better one (less similar to base, but still valid)
            if (_similar(s2, caption) < _similar(s1, caption)) and len(s2) >= 90:
                return s2

        return s1

    # @torch.inference_mode()
    # def rewrite(self, caption: str, keywords: Optional[List[str]] = None) -> str:
    #     def _gen(prompt: str) -> str:
    #         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
    #         out = self.model.generate(
    #             **inputs,
    #             do_sample=False,
    #             num_beams=4,
    #             no_repeat_ngram_size=3,
    #             length_penalty=0.9,
    #             max_new_tokens=self.max_new_tokens,
    #             early_stopping=True,
    #         )
    #         text = self.tokenizer.decode(out[0], skip_special_tokens=True)
    #         m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    #         text = (m.group(1) if m else text).strip()
    #         text = re.sub(r"^[\s\-\:\.\•]+", "", text)     # strip list markers
    #         text = re.sub(r"\s+", " ", text).strip()
    #         # force a single sentence
    #         if "." in text:
    #             text = text.split(".")[0].strip() + "."
    #         else:
    #             text = text.rstrip("!?,;:") + "."
    #         if len(text) > 180:
    #             text = text[:180].rsplit(" ", 1)[0].rstrip(".,;:") + "."
    #         return text

    #     # Pass A: few-shot prompt
    #     p1 = _build_prompt(caption, keywords)
    #     s1 = _gen(p1)

    #     if not _looks_bad(s1):
    #         return s1

    #     # Pass B: simpler wording to avoid parroting edge cases
    #     alt = (
    #         "Rewrite this image caption for SEO as a single natural sentence (110–160 chars), "
    #         "front-loading 1–2 keywords if natural, no hashtags.\n"
    #         f"Caption: {caption}\n"
    #         f"Keywords: {', '.join(keywords) if keywords else 'none'}\n"
    #         "Return only the sentence:\n<answer>"
    #     )
    #     s2 = _gen(alt)
    #     if not _looks_bad(s2):
    #         return s2

    #     # Final fallback: rule-based polish so you never emit junk
    #     return _rule_based_fallback(caption, keywords or [])
    # @torch.inference_mode()
    # def rewrite(self, caption: str, keywords: Optional[List[str]] = None) -> str:
    #     prompt = _build_prompt(caption, keywords)
    #     inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
    #     out = self.model.generate(
    #         **inputs,
    #         do_sample=False,           # deterministic
    #         num_beams=4,
    #         length_penalty=0.9,
    #         max_new_tokens=self.max_new_tokens,
    #         early_stopping=True,
    #     )
    #     text = self.tokenizer.decode(out[0], skip_special_tokens=True)

    #     # Pull text between <answer>...</answer>
    #     m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    #     text = (m.group(1) if m else text).strip()

    #     # Guardrails: single sentence, reasonable length, clean leading punctuation
    #     text = re.sub(r"^[\s\-\:\.\•]+", "", text)         # remove leading list markers
    #     text = re.sub(r"\s+", " ", text).strip()
    #     if "." in text:
    #         text = text.split(".")[0].strip() + "."
    #     if len(text) > 180:
    #         text = text[:180].rsplit(" ", 1)[0].rstrip(".,;:") + "."

    #     return text

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
    model_name: str = "google/flan-t5-small",
):
    items = load_base(base_json)
    kw_map = load_keywords(keywords_csv)
    rewriter = SeoRewriter(model_name=model_name)
    out = []

    for it in tqdm(items, desc="Optimizing captions", unit="caption"):
        img = it["image_id"]
        cap = it["caption"]
        kws = kw_map.get(img, None)
        seo = rewriter.rewrite(cap, kws)
        out.append({
            "image_id": img,
            "base_caption": cap,
            "keywords": kws or [],
            "seo_caption": seo
        })
        if len(out) == 100:
            break  # for quick testing

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

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
