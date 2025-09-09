# In this training i trained the model 10 times, each with 10 epochs, and saved the model after each training session.
# The checkpoints are named baseline_1.pt, baseline_2.pt,..., and baseline_10.pt respectively.
# I trained the model for a total of 100 epochs, all the epochs were run on the different dataset.
# Each with the manifest flickr8k1F800I.json then flickr8k2F800I.json and so on until flickr8k10F8090I.json

# Then I trained the model 3 times, one with 100 epoches, then 300 epochs, and finally 100 epochs, to reach a total of 500 epochs.
# The checkpoints are named baseline_trial.pt, baseline_trial_1.pt, and baseline_trial_2.pt respectively.
# I trained the model on a small manefist of 32 samples to check the code and training loop, the manifest file located at data/manifests/flickr8k1F32I.json.
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
        # pick ONE caption (prep provides cleaned multi-captions)
        cap = random.choice(it["captions"])
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
    def forward(self,x):
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
        out,_ = self.lstm(inp)                                         # [B,T,H]
        out = self.dropout(out)
        logits = self.fc(out)                                          # [B,T,V]
        return logits

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = json.load(open(args.manifest, "r", encoding="utf8"))

    # build tokenizer from all captions in manifest
    all_caps = [it["captions"] for it in manifest["items"]]
    tok = SimpleTokenizer()
    tok.build(all_caps)
    print(f"Vocab size: {len(tok.word2idx)}")

    # transforms: ImageNet normalization + light augmentation
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        mean, std = weights.meta["mean"], weights.meta["std"]
    except Exception:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # used to normalize your input images so they look statistically similar to what the pretrained ResNet saw during training.

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ds = CaptionDataset(manifest, tok, transform=transform, max_len=args.max_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    encoder = ImageEncoder().to(device)
    decoder = DecoderLSTM(vocab_size=len(tok.word2idx), embed_dim=128, hidden=256, img_dim=512).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=3e-3, weight_decay=1e-4)

    start_epoch = 0
    if args.resume is not None and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # load tokenizer
        tok.word2idx = ckpt.get("vocab", tok.word2idx)
        tok.idx2word = {v:k for k,v in tok.word2idx.items()}
        # load decoder weights
        decoder.load_state_dict(ckpt["decoder"])
        # load optimizer
        optimizer.load_state_dict(ckpt["optimizer"])
        # continue from last epoch
        start_epoch = ckpt.get("epoch", 0)
    last_epoch = start_epoch
    
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
        last_epoch = epoch + 1

    ckpt_dir = os.path.dirname(args.ckpt_dir)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
        
    ckpt = {
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "vocab": tok.word2idx,
        "epoch": last_epoch
    }
    
    torch.save(ckpt, args.ckpt_dir)
    print("Saved checkpoint to", args.ckpt_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # default name matches your preparation pattern (change as needed)
    p.add_argument("--manifest", default="data/manifests/flickr8k1F8091I.json")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_len", type=int, default=20)
    p.add_argument("--ckpt_dir", default="checkpoints/baseline.pt")
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()
    train(args)
