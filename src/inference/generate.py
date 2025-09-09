import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# simple model defs must match train file
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        res = models.resnet18(pretrained=True)
        modules = list(res.children())[:-1]
        self.backbone = nn.Sequential(*modules)
    def forward(self,x):
        return self.backbone(x).view(x.size(0), -1)

class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=256, img_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim + img_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab_size)
    def forward_step(self, img_feat, prev_token, hidden):
        emb = self.embed(prev_token).unsqueeze(1)  # B x 1 x E
        img_ext = img_feat.unsqueeze(1)
        inp = torch.cat([img_ext, emb], dim=2)
        out, hidden = self.lstm(inp, hidden)
        logits = self.fc(out.squeeze(1))
        return logits, hidden

def greedy_decode(encoder, decoder, imgs, vocab, max_len=20, device="cpu"):
    inv = {v:k for k,v in vocab.items()}
    encoder.eval(); decoder.eval()
    with torch.no_grad():
        feats = encoder(imgs.to(device))
        B = feats.size(0)
        prev = torch.tensor([vocab.get("<s>",1)]*B, dtype=torch.long, device=device)
        hidden = None
        results = [[] for _ in range(B)]
        for _ in range(max_len):
            logits, hidden = decoder.forward_step(feats, prev, hidden)
            _, nxt = torch.max(logits, dim=1)
            for i in range(B):
                results[i].append(int(nxt[i].item()))
            prev = nxt
    out_caps = []
    for seq in results:
        words = []
        for tid in seq:
            w = inv.get(tid, "<unk>")
            if w == "</s>":
                break
            if w not in ("<s>", "<pad>"):
                words.append(w)
        out_caps.append(" ".join(words))
    return out_caps

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/baseline.pt")
    p.add_argument("--manifest", default="data/manifests/flickr8k1F8091I.json")
    p.add_argument("--out", default="outputs/predictions_old.json")
    p.add_argument("--num", type=int, default=200)
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    vocab = ck["vocab"]
    encoder = ImageEncoder()
    decoder = DecoderLSTM(vocab_size=len(vocab))
    decoder.load_state_dict(ck["decoder"])
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

    manifest = json.load(open(args.manifest, "r", encoding="utf8"))
    items = manifest["items"][: args.num]
    imgs = []
    ids = []
    for it in items:
        img = Image.open(it["image_path"]).convert("RGB")
        imgs.append(transform(img))
        ids.append(it["image_id"])
    imgs = torch.stack(imgs)
    caps = greedy_decode(encoder, decoder, imgs, vocab, max_len=20, device="cpu")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf8") as f:
        for iid, cap in zip(ids, caps):
            f.write(json.dumps({"image_id": iid, "caption": cap}) + "\n")
    print("Wrote", args.out)
