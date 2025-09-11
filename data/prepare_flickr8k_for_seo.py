import json
from pathlib import Path

raw = Path("data/data/raw/flickr8k")
tok = raw / "captions.txt"         
imgs = raw / "Images"

assert tok.exists(), f"captions.txt not found at: {tok}"

caps = {} 
for line in tok.read_text(encoding="utf8").splitlines():
    line = line.strip()
    if not line:
        continue
    if line.lower().startswith("image,caption"):
        continue
    # Split on the FIRST comma only
    try:
        fname, text = line.split(",", 1)
    except ValueError:
        continue

    fname = fname.strip()
    if not fname.lower().endswith(".jpg"):
        fname = fname + ".jpg"
    clean = text.strip().strip('"').strip()
    if clean:
        caps.setdefault(fname, []).append(clean)

# Choose one caption per image => longest = usually most informative
def pick_caption(candidates):
    return max(candidates, key=len)

items = []
for fname, cand_list in caps.items():
    if (imgs / fname).exists():
        items.append({"image_id": fname, "caption": pick_caption(cand_list)})

items.sort(key=lambda r: r["image_id"].lower())

out_path = Path("data/seo/flickr8kSEO.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf8")
print(f"Wrote {out_path} with {len(items)} images")
