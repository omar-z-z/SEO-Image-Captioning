# In this training i trained the model 1 time, with 10 epochs, and saved the model after the training session.
# The checkpoint is named baseline.pt.
# I trained the model with the manifest file located at data/manifests/flickr8k1F8091I.json.
import argparse
import json
import os
import random
import hashlib
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ---------------- Tokenizer ----------------
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>":0, "<s>":1, "</s>":2, "<unk>":3}
        self.idx2word = {v:k for k,v in self.word2idx.items()}

    def build(self, captions_list, min_freq=2):
        freq = defaultdict(int)
        for caps in captions_list:
            for c in caps:
                for w in c.lower().split():
                    freq[w]+=1
        idx = max(self.word2idx.values())+1
        for w,cnt in freq.items():
            if cnt >= min_freq and w not in self.word2idx:
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                idx+=1

    def encode(self, s, max_len=20):
        toks = s.lower().split()[: max_len-2]
        ids = [self.word2idx.get(w, self.word2idx["<unk>"]) for w in toks]
        return [self.word2idx["<s>"]] + ids + [self.word2idx["</s>"]]

# ---------------- Datasets ----------------
class CaptionDataset(Dataset):
    """Uses ALL captions per image; yields (image_tensor, caption_ids)."""
    def __init__(self, manifest, tokenizer, transform=None, max_len=20):
        self.tok = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.samples = []
        for it in manifest["items"]:
            for cap in it["captions"]:
                self.samples.append({"image_path": it["image_path"], "caption": cap})

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s["image_path"]).convert("RGB")
        if self.transform: img = self.transform(img)
        ids = self.tok.encode(s["caption"], max_len=self.max_len)
        if len(ids) < self.max_len: ids += [0]*(self.max_len - len(ids))
        return img, torch.tensor(ids[:self.max_len], dtype=torch.long)

class ImageOnlyDataset(Dataset):
    """For precomputing features: yields (image_path, image_tensor)."""
    def __init__(self, image_paths, transform):
        self.paths = image_paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p = self.paths[i]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return p, img

class CaptionFeatDataset(Dataset):
    """Loads precomputed features; yields (feat_vector, caption_ids)."""
    def __init__(self, manifest, tokenizer, feats_dir, max_len=20):
        self.tok = tokenizer
        self.max_len = max_len
        self.feats_dir = feats_dir
        self.samples = []
        for it in manifest["items"]:
            for cap in it["captions"]:
                self.samples.append({"image_path": it["image_path"], "caption": cap})

    def _feat_path(self, image_path):
        h = hashlib.md5(image_path.encode("utf-8")).hexdigest()
        return os.path.join(self.feats_dir, f"{h}.pt")

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        fpath = self._feat_path(s["image_path"])
        feats = torch.load(fpath)  # [512]
        ids = self.tok.encode(s["caption"], max_len=self.max_len)
        if len(ids) < self.max_len: ids += [0]*(self.max_len - len(ids))
        return feats, torch.tensor(ids[:self.max_len], dtype=torch.long)

def collate_fn(batch):
    xs, caps = zip(*batch)
    xs = torch.stack(xs)
    caps = torch.stack(caps)
    return xs, caps

# ---------------- Models ----------------
class ImageEncoder(nn.Module):
    """ResNet18 backbone frozen; outputs 512-d features."""
    def __init__(self, feat_dim=512):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            res = models.resnet18(weights=weights)
        except Exception:
            res = models.resnet18(pretrained=True)
        modules = list(res.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        for p in self.backbone.parameters(): p.requires_grad = False
    def forward(self,x):
        f = self.backbone(x)
        return f.view(f.size(0), -1)  # [B,512]

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=256, img_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim + img_dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc    = nn.Linear(hidden, vocab_size)
        # NEW: image-conditioned init for (h0, c0)
        self.h_proj = nn.Linear(img_dim, hidden)
        self.c_proj = nn.Linear(img_dim, hidden)

    def forward(self, img_feats, captions):
        emb = self.embed(captions)                                    # [B,T,E]
        img_ext = img_feats.unsqueeze(1).expand(-1, emb.size(1), -1)  # [B,T,IMG]
        inp = torch.cat([img_ext, emb], dim=2)                        # [B,T,IMG+E]
        # image-conditioned initial hidden/cell
        h0 = torch.tanh(self.h_proj(img_feats)).unsqueeze(0)          # [1,B,H]
        c0 = torch.tanh(self.c_proj(img_feats)).unsqueeze(0)          # [1,B,H]
        out, _ = self.lstm(inp, (h0, c0))                             # [B,T,H]
        out = self.dropout(out)
        return self.fc(out)                                           # [B,T,V]
# ---------------- Utils ----------------
@torch.no_grad()
def validate(encoder, decoder, loader, device, criterion, using_feats=False):
    decoder.eval()
    total = 0.0; n = 0
    for xs, caps in loader:
        xs = xs.to(device); caps = caps.to(device)
        feats = xs if using_feats else encoder(xs)
        logits = decoder(feats, caps[:, :-1])
        loss = criterion(logits.view(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
        total += loss.item(); n += 1
    return total / max(1, n)

@torch.no_grad()
def precompute_features(items, transform, encoder, device, feats_dir, batch_size=64, num_workers=0):
    os.makedirs(feats_dir, exist_ok=True)
    # unique image paths
    all_paths = sorted({it["image_path"] for it in items})
    ds = ImageOnlyDataset(all_paths, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        persistent_workers=(num_workers>0), collate_fn=lambda b: list(zip(*b)))
    encoder.eval()
    done = 0
    for batch in loader:
        paths, imgs = batch
        imgs = torch.stack(imgs).to(device)
        feats = encoder(imgs).cpu()  # [B,512]
        for p, f in zip(paths, feats):
            h = hashlib.md5(p.encode("utf-8")).hexdigest()
            out = os.path.join(feats_dir, f"{h}.pt")
            if not os.path.exists(out):
                torch.save(f, out)
        done += len(paths)
        if done % 256 == 0:
            print(f"Precompute: {done}/{len(all_paths)} images cached")

def train(args):
    # Optional: let PyTorch use more CPU threads if available.
    # torch.set_num_threads(max(1, os.cpu_count() or 1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.load(open(args.manifest, "r", encoding="utf8"))

    # ---------- split by image
    items = manifest["items"]
    random.shuffle(items)
    cut = int(0.9 * len(items))
    train_items = items[:cut]; val_items = items[cut:]

    # tokenizer (train only)
    tok = SimpleTokenizer()
    tok.build([it["captions"] for it in train_items], min_freq=2)
    print(f"Vocab size: {len(tok.word2idx)} (min_freq=2)")

    # transforms (224 for best compatibility with ResNet18 pretrain)
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        mean, std = weights.meta["mean"], weights.meta["std"]
    except Exception:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # models
    encoder = ImageEncoder().to(device)
    encoder.eval()  # freeze BN

    # ---- optional precompute path
    using_feats = args.feats_dir is not None and len(args.feats_dir) > 0
    if using_feats:
        print(f"Precomputing (or using cached) features into: {args.feats_dir}")
        precompute_features(train_items + val_items, val_transform, encoder, device,
                            args.feats_dir, batch_size=args.feats_bs, num_workers=args.num_workers)
        train_ds = CaptionFeatDataset({"items": train_items}, tok, feats_dir=args.feats_dir, max_len=args.max_len)
        val_ds   = CaptionFeatDataset({"items": val_items},   tok, feats_dir=args.feats_dir, max_len=args.max_len)
    else:
        train_ds = CaptionDataset({"items": train_items}, tok, transform=train_transform, max_len=args.max_len)
        val_ds   = CaptionDataset({"items": val_items},   tok, transform=val_transform,   max_len=args.max_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, persistent_workers=(args.num_workers>0),
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, persistent_workers=(args.num_workers>0),
        collate_fn=collate_fn
    )

    decoder = DecoderLSTM(vocab_size=len(tok.word2idx), embed_dim=128, hidden=256, img_dim=512).to(device)

    # loss/opt/sched
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # resume (decoder/opt only; keep current tokenizer!)
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if "decoder" in ckpt: decoder.load_state_dict(ckpt["decoder"])
        if "optimizer" in ckpt: optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)

    best_val = float("inf"); last_epoch = start_epoch
    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        tot = 0.0; steps = 0
        for xs, caps in train_loader:
            xs = xs.to(device); caps = caps.to(device)
            feats = xs if using_feats else encoder(xs)
            logits = decoder(feats, caps[:, :-1])
            loss = criterion(logits.view(-1, logits.size(-1)), caps[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            tot += loss.item(); steps += 1

        train_avg = tot / max(1, steps)
        val_loss = validate(encoder, decoder, val_loader, device, criterion, using_feats=using_feats)
        scheduler.step()
        last_epoch = epoch + 1
        print(f"Epoch {last_epoch}/{args.epochs}  train_loss={train_avg:.4f}  val_loss={val_loss:.4f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            os.makedirs(os.path.dirname(args.ckpt_dir) or ".", exist_ok=True)
            ckpt = {
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "vocab": tok.word2idx,
                "epoch": last_epoch
            }
            torch.save(ckpt, args.ckpt_dir)
            print(f"Saved best checkpoint to {args.ckpt_dir} (val_loss={best_val:.4f})")

    print("Training complete.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifests/flickr8k1F8091I.json")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)          # CPU-friendly default
    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--ckpt_dir", default="checkpoints/baseline.pt")
    p.add_argument("--resume", type=str, default=None)

    # Speed knobs
    p.add_argument("--feats_dir", type=str, default="")           # set to a folder to enable caching
    p.add_argument("--feats_bs", type=int, default=64)            # batch size for precompute pass
    p.add_argument("--num_workers", type=int, default=0)          # try 2 on Windows; >=4 on Linux

    args = p.parse_args()
    train(args)
