# This file is for re-evaluating the saved checkpoints, because I forgot to log the results when I ran the models.
import argparse, json, os, math, hashlib
from pathlib import Path

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms, models
from PIL import Image

# ---------- CPU tuning ----------
def tune_cpu():
    n = os.cpu_count() or 4
    torch.set_num_threads(max(1, n - 1))
    torch.set_num_interop_threads(max(1, n // 2))
    try:
        torch.backends.mkldnn.enabled = True
    except Exception:
        pass

# ---------- tokenizer (same API as training) ----------
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>":0, "<s>":1, "</s>":2, "<unk>":3}
        self.idx2word = {v:k for k,v in self.word2idx.items()}
    def build_from_vocab(self, vocab_dict):
        self.word2idx = dict(vocab_dict)
        self.idx2word = {v:k for k,v in self.word2idx.items()}
    def encode(self, s, max_len=20):
        toks = s.lower().split()[: max_len-2]
        ids = [self.word2idx.get(w, self.word2idx["<unk>"]) for w in toks]
        out = [self.word2idx["<s>"]] + ids + [self.word2idx["</s>"]]
        if len(out) < max_len: out += [0] * (max_len - len(out))
        return out[:max_len]

# ---------- datasets ----------
class ImgOnly(Dataset):
    def __init__(self, items, transform):
        self.items, self.tf = items, transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p = self.items[i]["image_path"]
        img = Image.open(p).convert("RGB")
        return self.tf(img)

# ---------- models ----------
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            res = models.resnet18(weights=weights)
        except Exception:
            res = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(res.children())[:-1])
        for p in self.backbone.parameters(): p.requires_grad = False
    def forward(self, x):
        f = self.backbone(x)
        return f.view(f.size(0), -1)  # (B,512)

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=256, img_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + img_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
    def forward(self, img_feats, captions_in):
        emb = self.embed(captions_in)                               # (B,T,E)
        img_ext = img_feats.unsqueeze(1).expand(-1, emb.size(1), -1)# (B,T,512)
        out,_ = self.lstm(torch.cat([img_ext, emb], dim=2))         # (B,T,H)
        return self.fc(out)                                         # (B,T,V)

# ---------- feature extraction with caching ----------
@torch.inference_mode()
def extract_feats(manifest_items, batch_size, num_workers, device, cache_dir, img_size):
    # normalization aligned with training
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        mean, std = weights.meta["mean"], weights.meta["std"]
    except Exception:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    key_str = "|".join(it["image_path"] for it in manifest_items) + f"|{img_size}|{mean}|{std}"
    cache_key = hashlib.md5(key_str.encode()).hexdigest()
    cache_path = Path(cache_dir) / f"feats_res18_{cache_key}.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location=device)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = ImgOnly(manifest_items, tf)

    dl_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if num_workers and num_workers > 0:
        dl_kwargs.update(dict(prefetch_factor=4, persistent_workers=True, pin_memory=False))

    loader = DataLoader(ds, **dl_kwargs)
    enc = ImageEncoder().to(device).eval()

    feats = []
    for x in loader:
        x = x.to(device, non_blocking=False)
        feats.append(enc(x))
    feats = torch.cat(feats, dim=0)  # (N,512)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, cache_path)
    return feats

# ---------- evaluate one checkpoint ----------
@torch.inference_mode()
def eval_one_ckpt(ckpt_path, feats, items, max_len, batch_size, num_workers, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    tok = SimpleTokenizer(); tok.build_from_vocab(ckpt["vocab"])

    # deterministic first caption targets
    caps = []
    for it in items:
        cap = it["captions"][0]
        caps.append(torch.tensor(tok.encode(cap, max_len=max_len), dtype=torch.long))
    caps = torch.stack(caps)  # (N,T)

    dec = DecoderLSTM(len(tok.word2idx), 128, 256, 512).to(device).eval()
    dec.load_state_dict(ckpt["decoder"])

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="sum")
    N, T = caps.size()
    ds = TensorDataset(torch.arange(N))

    dl_kwargs = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if num_workers and num_workers > 0:
        dl_kwargs.update(dict(prefetch_factor=6, persistent_workers=True, pin_memory=False))
    loader = DataLoader(ds, **dl_kwargs)

    total_tokens = 0
    total_correct = 0
    total_loss = 0.0

    for (idx_batch,) in loader:
        idx_batch = idx_batch.to(device, non_blocking=False)
        f = feats.index_select(0, idx_batch)            # (B,512)
        y = caps.index_select(0, idx_batch).to(device)  # (B,T)
        y_in, y_tgt = y[:, :-1], y[:, 1:]

        logits = dec(f, y_in)                           # (B,T-1,V)
        mask = (y_tgt != 0)
        B, TT, V = logits.shape
        loss = criterion(logits.reshape(-1, V), y_tgt.reshape(-1))
        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        total_correct += (pred.eq(y_tgt) & mask).sum().item()
        total_tokens  += mask.sum().item()

    loss_tok = total_loss / max(1, total_tokens)
    ppl = math.exp(loss_tok) if loss_tok < 50 else float("inf")
    acc = total_correct / max(1, total_tokens)
    return {
        "checkpoint": os.path.basename(ckpt_path),
        "tokens": int(total_tokens),
        "loss_per_token": loss_tok,
        "perplexity": ppl,
        "token_accuracy": acc,
        "epoch_tag": ckpt.get("epoch", None),
    }

# ---------- main ----------
def main():
    tune_cpu()

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/flickr8k1F8091I.json",
                    help="default manifest used if --manifests is not supplied")
    ap.add_argument("--manifests", nargs="*", default=None,
                    help="optional list aligned 1:1 with --ckpts")
    ap.add_argument("--ckpts", nargs="+", required=True,
        help="paths to checkpoints, e.g. checkpoints/baseline_1.pt ...")
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=20)
    ap.add_argument("--subset", type=int, default=0, help="evaluate only first N images")
    ap.add_argument("--cache_dir", default=".cache_feats")
    ap.add_argument("--img_size", type=int, default=192)
    args = ap.parse_args()

    if args.manifests is not None and len(args.manifests) != len(args.ckpts):
        raise SystemExit("len(--manifests) must equal len(--ckpts)")

    device = "cpu"

    def load_items(mp):
        m = json.load(open(mp, "r", encoding="utf8"))
        return m["items"]

    for i, ck in enumerate(args.ckpts):
        if not os.path.exists(ck):
            print(f"[skip] missing: {ck}")
            continue

        manifest_path = args.manifests[i] if args.manifests else args.manifest
        items = load_items(manifest_path)
        if args.subset and args.subset < len(items):
            items = items[:args.subset]

        feats = extract_feats(items, batch_size=256, num_workers=args.num_workers,
                              device=device, cache_dir=args.cache_dir, img_size=args.img_size)
        feats = feats.to(device, non_blocking=False)

        m = eval_one_ckpt(ck, feats, items, args.max_len, args.batch_size, args.num_workers, device)
        print(f"{m['checkpoint']}: loss/token={m['loss_per_token']:.4f}, "
              f"ppl={m['perplexity']:.2f}, acc={m['token_accuracy']:.4f}, "
              f"epoch_tag={m['epoch_tag']}, tokens={m['tokens']}")

if __name__ == "__main__":
    main()
