import json, random, argparse
from pathlib import Path


p = argparse.ArgumentParser()
p.add_argument("--limit", type=int, default=None)
p.add_argument("--number_of_files", type=int, default=1)
args = p.parse_args()
raw = Path("data/data/raw/flickr8k")
tok = raw/"captions.txt"
imgs = raw/"Images"
assert tok.exists(), "put captions.txt in data/raw/flickr8k"
caps = {}
for L in tok.read_text(encoding="utf8").splitlines():
    if not L.strip(): continue
    if (L == "image,caption"): continue
    key, text = (L.split(".jpg,"))
    fname = key + ".jpg"
    clean = text.strip().strip('"')
    caps.setdefault(fname, []).append(clean)
items=[]
for f,cs in caps.items():
    p = imgs/f
    if p.exists():
        items.append({"image_id":f, "image_path":str(p), "captions":cs})

if getattr(args, "number_of_files", 1):
    for i in range(args.number_of_files): 
        if getattr(args, "limit", None):
            items = random.sample(items, args.limit)
        out={"dataset":"Flickr8k","n_images":len(items),"items":items}
        outfile=Path("data/manifests/flickr8k"+str(i+1)+"F"+str(len(items))+"I.json")
        outfile.parent.mkdir(parents=True,exist_ok=True)
        outfile.write_text(json.dumps(out,indent=2),encoding="utf8")
        print("Wrote",outfile,"with",len(items),"images")