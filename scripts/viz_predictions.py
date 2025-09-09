import argparse
import json
import math
import random
import textwrap
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

# loading the manifest
def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf8") as f:
        man = json.load(f)
    id2path = {}
    for it in man.get("items", []):
        iid = it.get("image_id")
        ipath = it.get("image_path")
        if iid and ipath:
            id2path[iid] = ipath
    return id2path

# loading the predictions
def load_predictions(pred_path):
    preds = []
    with open(pred_path, "r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                preds.append({"image_id": obj.get("image_id"), "caption": obj.get("caption", "")})
            except Exception:
                pass
    return preds


def best_grid(n, max_cols=4):
    ncols = min(max_cols, max(1, n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


def make_grid(samples, save_path=None, dpi=150, wrap_width=45, title=None):
    if not samples:
        print("No samples to visualize.")
        return

    n = len(samples)
    nrows, ncols = best_grid(n, max_cols=4)
    fig_w, fig_h = ncols * 4, nrows * 4 

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    plt.subplots_adjust(hspace=1)
    axes_flat = [ax for row in axes for ax in row]

    for ax, sample in zip(axes_flat, samples):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        try:
            img = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to open:\n{sample['image_path']}", ha="center", va="center", wrap=True)
            continue
        ax.imshow(img)
        # Caption under the image
        cap = textwrap.fill(sample.get("caption", "").strip(), width=45)
        ax.set_xlabel(cap, fontsize=9, labelpad=6)


        iid = sample.get("image_id", "")
        if iid:
            ax.text(0.01, 0.99, str(iid), transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.6, boxstyle="round,pad=0.2"))

    for ax in axes_flat[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved grid to {save_path}")

    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Path to predictions.jsonl (from generate.py)")
    ap.add_argument("--manifest", required=True, help="Path to manifest.json used for training/inference")
    ap.add_argument("--num", type=int, default=20, help="Number of random samples to visualize")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--save", default=None, help="Optional path to save the grid image (e.g., outputs/viz.png)")
    args = ap.parse_args()

    random.seed(args.seed)

    id2path = load_manifest(args.manifest)
    preds = load_predictions(args.pred)

    # Join predictions with image paths
    joined = []
    missing = 0
    for p in preds:
        iid = p.get("image_id")
        if iid in id2path:
            joined.append({"image_id": iid, "image_path": id2path[iid], "caption": p.get("caption", "")})
        else:
            missing += 1

    if not joined:
        print("No predictions matched with manifest image paths. Check that image_id values align.")
        return

    k = min(args.num, len(joined))
    samples = random.sample(joined, k)

    title = f"Random {k} predictions"
    make_grid(samples, save_path=args.save, title=title)


if __name__ == "__main__":
    main()
