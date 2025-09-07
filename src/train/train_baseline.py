#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path
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

    def build(self, captions_list):
        freq = defaultdict(int)
        for caps in captions_list:
            for c in caps:
                for w in c.lower().split():
                    freq[w]+=1
        idx = max(self.word2idx.values())+1
        for w,cnt in freq.items():
            if w not in self.word2idx:
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                idx+=1

    def encode(self, s, max_len=20):
        toks = s.lower().split()[: max_len-2]
        ids = [self.word2idx.get(w, self.word2idx["<unk>"]) for w in toks]
        return [self.word2idx["<s>"]] + ids + [self.word2idx["</s>"]]

# --- dataset
class CaptionDataset(Dataset):
    def __init__(self, manifest, tokenizer, transform=None, max_len=20):
        self.items = manifest["items"]
        self.tok = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["image_path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # pick first caption
        cap = random.choice(it["captions"])
        # cap = it["captions"][0] # always first caption
        ids = self.tok.encode(cap, max_len=self.max_len)
        if len(ids) < self.max_len:
            ids = ids + [0]*(self.max_len - len(ids))
        return img, torch.tensor(ids[:self.max_len], dtype=torch.long)

def collate_fn(batch):
    imgs, caps = zip(*batch)
    imgs = torch.stack(imgs)
    caps = torch.stack(caps)
    return imgs, caps

# --- model
class ImageEncoder(nn.Module):
    # Encoder: ResNet18 pretrained from torchvision with the final classification head removed.
    # The code uses the convolutional backbone and outputs a 512-dim vector per image (after global pooling).
    # The encoder parameters are frozen (p.requires_grad = False) to make training fast and cheap.
    def __init__(self, feat_dim=512):
        super().__init__()
        res = models.resnet18(pretrained=True)
        modules = list(res.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        for p in self.backbone.parameters():
            p.requires_grad = False
    def forward(self,x):
        f = self.backbone(x)
        return f.view(f.size(0), -1)

class DecoderLSTM(nn.Module):
    # Embedding layer: maps token ids → embeddings (size 128 in the script).
    # LSTM: input at each time step is concatenation of the image feature (512-d) and token embedding (128-d) → LSTM hidden size 256.
    # Final linear layer maps LSTM outputs back to vocabulary logits.
    def __init__(self, vocab_size, embed_dim=256, hidden=512, img_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + img_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
    def forward(self, img_feats, captions):
        emb = self.embed(captions)
        img_ext = img_feats.unsqueeze(1).expand(-1, emb.size(1), -1)
        inp = torch.cat([img_ext, emb], dim=2)
        out,_ = self.lstm(inp)
        logits = self.fc(out)
        return logits

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.load(open(args.manifest, "r", encoding="utf8"))
    # build tokenizer
    all_caps = [it["captions"] for it in manifest["items"]]
    tok = SimpleTokenizer()
    tok.build(all_caps)
    print(tok)
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    ds = CaptionDataset(manifest, tok, transform=transform, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    encoder = ImageEncoder().to(device) # this encoder outputs 512-dim features
    decoder = DecoderLSTM(vocab_size=len(tok.word2idx), embed_dim=128, hidden=256, img_dim=512).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3, weight_decay=1e-4)
    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # load tokenizer
        tok.word2idx = ckpt["vocab"]
        tok.idx2word = {v:k for k,v in tok.word2idx.items()}
        # load decoder weights
        decoder.load_state_dict(ckpt["decoder"])
        # load optimizer
        optimizer.load_state_dict(ckpt["optimizer"])
        # continue from last epoch
        start_epoch = ckpt.get("epoch", 0)

    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        total_loss = 0.0
        for imgs, caps in loader:
            imgs = imgs.to(device)
            caps = caps.to(device)
            feats = encoder(imgs)
            logits = decoder(feats, caps[:, :-1])
            loss = criterion(logits.view(-1, logits.size(-1)), caps[:,1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} avg_loss={(total_loss/len(loader)):.4f}")
    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt = {
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "vocab": tok.word2idx,
        "epoch": epoch + 1
    }
    torch.save(ckpt, os.path.join(args.ckpt_dir, "baseline_1.pt"))
    print("Saved checkpoint to", os.path.join(args.ckpt_dir, "baseline_1.pt"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/manifests/flickr8k.json")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args)
