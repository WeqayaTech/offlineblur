#!/usr/bin/env python3
"""Multi-task fine-tuning of the cross-attention demographic head on the frozen giant backbone.

    Loss = CE(gender) + KL(DLDL age distribution) + L1(expected age) + BCE(quality → "this sample was answered right")

The backbone is frozen (its dense features are the shared representation of the whole pipeline); only the
head trains. Every image is a "frame" whose ROI box is the whole image (or the face box if the manifest
has one), rendered at a random scale between --min-side and --max-side pixels so the head sees the same
token densities it will meet on 60 px background people and on the reporter filling the frame.

    python3 train_demographics.py --manifest manifests/all.csv --out weights/demo_head.pt [--backbone dinov2-giant] [--epochs 8]
"""
import argparse
import csv
import math
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from backbone import GiantBackbone
from roi_attention import AGE_BINS, DemographicHead, dldl_target, range_target


def load_manifest(path):
    rows = []
    with open(path) as fh:
        for r in csv.reader(fh):
            if len(r) < 5 or not r[0]:
                continue
            rows.append({"path": r[0], "age": float(r[1]), "lo": float(r[2]), "hi": float(r[3]), "gender": int(r[4])})
    return rows


class Sampler:
    def __init__(self, rows, patch, min_side, max_side, train=True, workers=12):
        self.rows, self.patch, self.min_side, self.max_side, self.train = rows, patch, min_side, max_side, train
        self.pool = ThreadPoolExecutor(workers)   # cv2 releases the GIL: decode/resize/augment run in parallel

    def render(self, r, side):
        im = cv2.imread(r["path"])
        if im is None:
            return None
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        if self.train:
            if random.random() < 0.5:
                im = im[:, ::-1]
            # random context margin: the tracker's box is never tight
            m = random.uniform(0.0, 0.25)
            pad = int(m * max(h, w))
            im = cv2.copyMakeBorder(np.ascontiguousarray(im), pad, pad, pad, pad, cv2.BORDER_REFLECT)
            h, w = im.shape[:2]
        s = side / min(h, w)
        tw, th = max(self.patch, int(round(w * s / self.patch)) * self.patch), max(self.patch, int(round(h * s / self.patch)) * self.patch)
        im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        if self.train and random.random() < 0.3:  # motion blur / defocus
            k = random.choice([3, 5, 7])
            im = cv2.GaussianBlur(im, (k, k), 0)
        if self.train and random.random() < 0.3:  # jpeg
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 80)])
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else im
        return np.ascontiguousarray(im)

    def batch(self, idx):
        # one scale per batch (log-uniform in train, the geometric mean in eval): images of one batch are padded to
        # the largest, so mixing 56 px and 448 px renders would waste most of the backbone pass
        side = math.exp(random.uniform(math.log(self.min_side), math.log(self.max_side))) if self.train else math.sqrt(self.min_side * self.max_side)
        ims, rows = [], []
        for i, im in zip(idx, self.pool.map(lambda j: self.render(self.rows[j], side), idx)):
            if im is not None:
                ims.append(im)
                rows.append(self.rows[i])
        return ims, rows


def prefetch(sampler, batches, depth=6):
    """Background thread renders the next batches while the GPU works on the current one."""
    q = queue.Queue(depth)

    def worker():
        for idx in batches:
            q.put(sampler.batch(idx))
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def losses(head, out, rows, device):
    g = torch.tensor([r["gender"] for r in rows], device=device)
    age = torch.tensor([r["age"] for r in rows], device=device)
    lo = torch.tensor([r["lo"] for r in rows], device=device)
    hi = torch.tensor([r["hi"] for r in rows], device=device)
    has_g, has_age, has_rng = g >= 0, age >= 0, (lo >= 0) & (hi >= lo) & (age < 0)
    tot, logs = 0.0, {}
    if has_g.any():
        lg = F.cross_entropy(out["gender_logits"][has_g], g[has_g])
        tot = tot + lg
        logs["gender"] = lg.item()
        logs["gender_acc"] = (out["gender_logits"][has_g].argmax(-1) == g[has_g]).float().mean().item()
    logp = out["age_logits"].log_softmax(-1)
    ev = (logp.exp() * torch.arange(AGE_BINS, device=device).float()).sum(-1)
    if has_age.any():
        t = dldl_target(age[has_age])
        kl = F.kl_div(logp[has_age], t, reduction="batchmean")
        l1 = (ev[has_age] - age[has_age]).abs().mean()
        tot = tot + kl + 0.05 * l1
        logs["age_kl"], logs["age_mae"] = kl.item(), l1.item()
    if has_rng.any():
        t = range_target(lo[has_rng], hi[has_rng])
        kl = F.kl_div(logp[has_rng], t, reduction="batchmean")
        tot = tot + 0.5 * kl
        logs["age_range_kl"] = kl.item()
    # quality target: was this sample answered right? (gender right when known, age within 6 y when known)
    with torch.no_grad():
        ok = torch.ones(len(rows), device=device, dtype=torch.bool)
        ok[has_g] &= out["gender_logits"][has_g].argmax(-1) == g[has_g]
        ok[has_age] &= (ev[has_age] - age[has_age]).abs() <= 6
        ok[has_rng] &= (ev[has_rng] >= lo[has_rng] - 3) & (ev[has_rng] <= hi[has_rng] + 3)
    lq = F.binary_cross_entropy_with_logits(out["quality_logit"], ok.float())
    tot = tot + lq
    logs["quality"] = lq.item()
    return tot, logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="dinov2-giant")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--n-queries", type=int, default=8)
    ap.add_argument("--n-layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-keys", type=int, default=4096)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--min-side", type=float, default=56, help="smallest rendered short side (px) — 4 tokens")
    ap.add_argument("--max-side", type=float, default=448)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    rows = load_manifest(a.manifest)
    random.shuffle(rows)
    if a.max_rows:
        rows = rows[:a.max_rows]
    n_val = max(1, int(len(rows) * a.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    print(f"[train] {len(train)} train / {len(val)} val rows; gender known {sum(r['gender'] >= 0 for r in rows)}, "
          f"exact age {sum(r['age'] >= 0 for r in rows)}, age range {sum(r['age'] < 0 and r['lo'] >= 0 for r in rows)}")

    bb = GiantBackbone(a.backbone, a.device, feat_stride=8)  # feat_stride irrelevant here: images are fed at native token size
    head = DemographicHead(bb.dim, a.dim, a.n_queries, a.n_layers, a.heads, a.max_keys, a.margin, dropout=0.1).to(a.device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * math.ceil(len(train) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    tr, va = Sampler(train, bb.patch, a.min_side, a.max_side, True), Sampler(val, bb.patch, a.min_side, a.max_side, False)
    cfg = {"backbone": a.backbone, "in_dim": bb.dim, "dim": a.dim, "n_queries": a.n_queries, "n_layers": a.n_layers, "heads": a.heads,
           "max_keys": a.max_keys, "margin": a.margin}

    def run_batch(ims, rws, train_mode):
        if not ims:
            return (None, None), None
        feats = bb.batch_feature_maps(ims)
        boxes = [torch.tensor([[0.0, 0.0, f.shape[2], f.shape[1]]], device=a.device) for f in feats]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=train_mode):
            out = head(feats, boxes)
        out = {k: v.float() for k, v in out.items()}
        return losses(head, out, rws, a.device), out

    step, t0, best = 0, time.time(), None
    for ep in range(a.epochs):
        head.train()
        order = list(range(len(train)))
        random.shuffle(order)
        agg = {}
        for ims, rws in prefetch(tr, [order[i:i + a.batch] for i in range(0, len(order), a.batch)]):
            (loss, logs), _ = run_batch(ims, rws, True)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) * 0.98 + v * 0.02 if k in agg else v
            if step % 50 == 0:
                print(f"[train] ep {ep} step {step}/{steps} loss {loss.item():.3f} " + " ".join(f"{k} {v:.3f}" for k, v in agg.items()) +
                      f" lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s")
        # validation
        head.eval()
        gc, gn, mae, man = 0, 0, 0.0, 0
        with torch.no_grad():
            for ims, rws in prefetch(va, [list(range(i, min(len(val), i + a.batch))) for i in range(0, len(val), a.batch)]):
                (_, _), out = run_batch(ims, rws, False)
                if out is None:
                    continue
                d = head.decode(out)
                for k, r in enumerate(rws):
                    if r["gender"] >= 0:
                        gc += int((d["p_female"][k] >= 0.5) == (r["gender"] == 1)); gn += 1
                    if r["age"] >= 0:
                        mae += abs(float(d["age_mean"][k]) - r["age"]); man += 1
        acc = gc / max(1, gn)
        mae_v = mae / max(1, man)
        print(f"[val] epoch {ep}: gender acc {acc:.4f} ({gn}), age MAE {mae_v:.2f} ({man})")
        score = acc - mae_v / 100.0
        if best is None or score > best:
            best = score
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": head.state_dict(), "cfg": cfg, "val": {"gender_acc": acc, "age_mae": mae_v, "epoch": ep}}, a.out)
            print(f"[train] saved {a.out}")
    print("[train] done")


if __name__ == "__main__":
    main()
