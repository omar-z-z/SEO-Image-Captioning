import json, random, sys
from pathlib import Path
raw = Path("data/data/raw/flickr8k")
tok = raw/"captions.txt"
imgs = raw/"Images"
assert tok.exists(), "put captions.txt in data/raw/flickr8k"
caps = {}
for L in tok.read_text(encoding="utf8").splitlines():
    if not L.strip(): continue
    if (L == "image,caption"): continue
    key, text = (L.split(".jpg,")) #  if "," in L else L.split("\t",1)
    fname = key + ".jpg"#.split("#")[0]
    caps.setdefault(fname,[]).append(text.strip())
items=[]
for f,cs in caps.items():
    p = imgs/f
    if p.exists():
        items.append({"image_id":f, "image_path":str(p), "captions":cs})
if len(items)>500:
    # random.seed(42)
    items = random.sample(items,500)
out={"dataset":"Flickr8k","n_images":len(items),"items":items}
outfile=Path("data/manifests/flickr8k.json")
outfile.parent.mkdir(parents=True,exist_ok=True)
outfile.write_text(json.dumps(out,indent=2),encoding="utf8")
print("Wrote",outfile,"with",len(items),"images")