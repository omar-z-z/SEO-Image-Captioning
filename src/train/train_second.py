#!/usr/bin/env python3
import argparse
import json
import os
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image


# --- tiny tokenizer (word-level)
class SimpleTokenizer:
    def __init__(self):
        self.word2idx = {"<pad>":0, "<s>":1, "</s>":2, "<unk>":3}
        self.idx2word = {v:k for k,v in self.word2idx.items()}

    def build(self, captions_list, min_freq=2):
        """Build vocab from a list of list-of-captions, keeping tokens with freq >= min_freq."""
        freq = defaultdict(int)
        for caps in captions_list:
            for c in caps:
                for w in c.lower().split():
                    freq[w] += 1
        idx = max(self.word2idx.values()) + 1
        for w, cnt in freq.items():
            if cnt >= min_freq and w not in self.word2idx:
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                idx += 1

    def encode(self, s, max_len=20):
        toks = s.lower().split()[: max_len - 2]
        ids = [self.word2idx.get(w, self.word2idx["<unk>"]) for w in toks]
        return [self.word2idx["<s>"]] + ids + [self.word2idx["</s>"]]


# --- dataset (uses ALL captions: one sample per caption)
class CaptionDataset(Dataset):
    def __init__(self, manifest, tokenizer, transform=None, max_len=20):
        self.tok = tokenizer
        self.transform = transform
        self.max_len = max_len
        self.samples = []
        for it in manifest["items"]:
            for cap in it["captions"]:
                self.samples.append({"image_path": it["image_path"], "caption": cap})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        ids = self.tok.encode(s["caption"], max_len=self.max_len)
        if len(ids) < self.max_len:
            ids = ids + [0] * (self.max_len - len(ids))
        return img, torch.tensor(ids[: self.max_len], dtype=torch.long)


def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)
    caps = torch.stack(caps)
    return imgs, caps


# --- model
class ImageEncoder(nn.Module):
    # ResNet18 backbone frozen
    def __init__(self, feat_dim=512):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            res = models.resnet18(weights=weights)
        except Exception:
            res = models.resnet18(pretrained=True)
        modules = list(res.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.backbone(x)
        return f.view(f.size(0), -1)


class DecoderLSTM(nn.Module):
    # small dropout for regularization
    def __init__(self, vocab_size, embed_dim=128, hidden=256, img_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + img_dim, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, vocab_size)

    def forward(self, img_feats, captions):
        emb = self.embed(captions)                                     # [B,T,E]
        img_ext = img_feats.unsqueeze(1).expand(-1, emb.size(1), -1)   # [B,T,IMG]
        inp = torch.cat([img_ext, emb], dim=2)                         # [B,T,IMG+E]
        out, _ = self.lstm(inp)                                        # [B,T,H]
        out = self.dropout(out)
        logits = self.fc(out)                                          # [B,T,V]
        return logits


@torch.no_grad()
def validate(encoder, decoder, loader, device, criterion):
    decoder.eval()
    total = 0.0
    count = 0
    for imgs, caps in loader:
        imgs = imgs.to(device)
        caps = caps.to(device)
        feats = encoder(imgs)
        logits = decoder(feats, caps[:, :-1])
        loss = criterion(logits.view(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
        total += loss.item()
        count += 1
    return total / max(1, count)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = json.load(open(args.manifest, "r", encoding="utf8"))

    # ---------- split by image (not by caption)
    items = manifest["items"]
    random.shuffle(items)
    cut = int(0.9 * len(items))
    train_items = items[:cut]
    val_items = items[cut:]

    # tokenizer from TRAIN captions only (min_freq=2)
    train_caps_list = [it["captions"] for it in train_items]
    tok = SimpleTokenizer()
    tok.build(train_caps_list, min_freq=2)
    print(f"Vocab size: {len(tok.word2idx)} (min_freq=2)")

    # transforms
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        mean, std = weights.meta["mean"], weights.meta["std"]
    except Exception:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # datasets/loaders (use ALL captions per image)
    train_manifest = {"items": train_items}
    val_manifest = {"items": val_items}
    train_ds = CaptionDataset(train_manifest, tok, transform=train_transform, max_len=args.max_len)
    val_ds   = CaptionDataset(val_manifest, tok, transform=val_transform,   max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # models
    encoder = ImageEncoder().to(device)
    encoder.eval()  # freeze BN running stats
    decoder = DecoderLSTM(vocab_size=len(tok.word2idx), embed_dim=128, hidden=256, img_dim=512).to(device)

    # loss/opt/sched
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # resume (decoder/opt only; vocab from checkpoint if present)
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if "vocab" in ckpt:
            tok.word2idx = ckpt["vocab"]
            tok.idx2word = {v: k for k, v in tok.word2idx.items()}
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)

    last_epoch = start_epoch
    best_val = float("inf")

    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        total_loss = 0.0
        steps = 0
        for imgs, caps in train_loader:
            imgs = imgs.to(device)
            caps = caps.to(device)
            feats = encoder(imgs)
            logits = decoder(feats, caps[:, :-1])
            loss = criterion(logits.view(-1, logits.size(-1)), caps[:, 1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        train_avg = total_loss / max(1, steps)
        val_loss = validate(encoder, decoder, val_loader, device, criterion)
        scheduler.step()

        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_avg:.4f}  val_loss={val_loss:.4f}")
        last_epoch = epoch + 1

        # save best (by val loss)
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
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--ckpt_dir", default="checkpoints/baseline.pt")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args)
