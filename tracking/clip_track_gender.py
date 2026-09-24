#!/usr/bin/env python3
"""Per-TRACK zero-shot gender with OpenCLIP: best-K crops per identity, batched, probability-averaged.

Stage 3 of the RF-DETR -> McByte -> CLIP pipeline (`run_rfdetr_mcbyte_clip.sh`). The tracker has already
said who is who; this decides each identity's gender once, from its K best views, instead of per frame —
so a back view or an occluded frame is outvoted by the good ones, and there is nothing to flicker.

Crop choice per track: quality = detection score * sqrt(box area), halved when the box touches the frame
edge (cut-off people), picked greedily with at least `--min-gap` frames between picks so the K views are
spread over time rather than K near-identical neighbours. Each crop is the box plus `--margin`, optionally
with everything outside the track's own (dilated) mask greyed out (`--masked`: in a crowd the box also
contains other people), then padded to a SQUARE before CLIP's preprocess — open_clip's preprocess
center-crops, which would cut the head off a tall person crop.

Classes are woman / man / girl / boy, each a prompt-template ensemble, but only the GENDER axis is used
for the decision: P(female) = P(woman) + P(girl). Measured on the trial clip, CLIP cannot separate the age
axis at street-crop resolution ("boy" took ~30% of men's probability, "girl" ~25% of women's), while
female vs male separates well. A track is labelled `woman` when mean P(female) >= `--blur-min` (default
0.4: escapes ship unblurred, over-blur does not), else `man`; P(child) = P(girl) + P(boy) is recorded
but never decides anything.

Outputs:
  track_labels.json   {tid: {label, p{woman,man,girl,boy}, n_crops, crop_frames}}
  masks.jsonl         the tracker's masks with prompt = track label (render_gender.py / report ready)
  clip_meta.json      model, crops, timing (crop extraction vs model), memory
  crops/<tid>.jpg     the K crops actually classified, side by side (--save-crops)

    python3 clip_track_gender.py --frames-meta <seq>/frames_meta.json --tracks <mcb>/tracks.jsonl \
        --masks <mcb>/masks.jsonl --out <dir> --model ViT-SO400M-14-SigLIP-384 --pretrained webli
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import frame_path, iter_jsonl, read_json, write_json

GB = 1024 ** 3
CLASSES = ["woman", "man", "girl", "boy"]
TEMPLATES = ["a photo of a {}.", "a photo of a {} walking on the street.", "a cropped photo of a {}.",
             "a low resolution photo of a {}.", "a photo of a {} seen from behind.",
             "a blurry photo of a {}.", "a photo of a {} in a crowd."]


def pick_views(rows, k, min_gap, W, H):
    def quality(r):
        x1, y1, x2, y2 = r["box"]
        q = r["score"] * max(1.0, (x2 - x1) * (y2 - y1)) ** 0.5
        if x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2:
            q *= 0.5
        return q
    chosen = []
    for r in sorted(rows, key=quality, reverse=True):
        if all(abs(r["f"] - c["f"]) >= min_gap for c in chosen):
            chosen.append(r)
            if len(chosen) == k:
                break
    return chosen


def crop(img, box, margin, mask=None):
    import cv2
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(W, x2 + mx)), int(min(H, y2 + my))
    c = img[y1:y2, x1:x2].copy()
    if mask is not None:
        m = cv2.dilate(mask[y1:y2, x1:x2].astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        c[~m] = 127
    h, w = c.shape[:2]
    s = max(h, w)
    sq = np.full((s, s, 3), 127, np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    return sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--tracks", required=True, help="tracks.jsonl from sam31_mcbyte.py {f, tid, box, score}")
    ap.add_argument("--masks", required=True, help="masks.jsonl from the same run {f, tid, rle}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ViT-SO400M-14-SigLIP-384")
    ap.add_argument("--pretrained", default="webli")
    ap.add_argument("--k", type=int, default=10, help="views per track")
    ap.add_argument("--min-gap", type=int, default=5, help="frames between chosen views")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--masked", action="store_true", help="grey out everything outside the track's mask")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--blur-min", type=float, default=0.4, help="label woman when mean P(woman)+P(girl) >= this")
    ap.add_argument("--save-crops", action="store_true")
    a = ap.parse_args()

    import cv2
    import open_clip
    import psutil
    import torch
    from PIL import Image
    from pycocotools import mask as mu

    fm = read_json(a.frames_meta)
    W, H = int(fm["width"]), int(fm["height"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()

    t = time.perf_counter()
    model, _, preprocess = open_clip.create_model_and_transforms(a.model, pretrained=a.pretrained, device="cuda")
    model.eval()
    tok = open_clip.get_tokenizer(a.model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        txt = []
        for c in CLASSES:
            e = model.encode_text(tok([t.format(c) for t in TEMPLATES]).cuda()).float()
            e = e / e.norm(dim=-1, keepdim=True)
            e = e.mean(0)
            txt.append(e / e.norm())
        txt = torch.stack(txt)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t
    torch.cuda.reset_peak_memory_stats()

    # ---- choose views
    per_track = defaultdict(list)
    for r in iter_jsonl(a.tracks):
        if r.get("src", "det") == "det":
            per_track[r["tid"]].append(r)
    views = {tid: pick_views(rows, a.k, a.min_gap, W, H) for tid, rows in per_track.items()}
    need = defaultdict(list)                     # frame -> [(tid, row)]
    for tid, vs in views.items():
        for r in vs:
            need[r["f"]].append((tid, r))
    masks = {}
    if a.masked:
        wanted = {(f, tid) for f, lst in need.items() for tid, _ in lst}
        for r in iter_jsonl(a.masks):
            if (r["f"], r["tid"]) in wanted:
                masks[(r["f"], r["tid"])] = r["rle"]

    # ---- extract crops (each frame read once)
    t = time.perf_counter()
    crops, owners = [], []
    for f in sorted(need):
        img = cv2.cvtColor(cv2.imread(str(frame_path(fm["img_dir"], f))), cv2.COLOR_BGR2RGB)
        for tid, r in need[f]:
            m = None
            if a.masked and (f, tid) in masks:
                rle = masks[(f, tid)]
                m = mu.decode({"size": rle["size"], "counts": rle["counts"].encode()}).astype(bool)
            crops.append(crop(img, r["box"], a.margin, m))
            owners.append((tid, f))
    crop_s = time.perf_counter() - t

    # ---- classify in batches
    t = time.perf_counter()
    probs = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(crops), a.batch):
            x = torch.stack([preprocess(Image.fromarray(c)) for c in crops[i:i + a.batch]]).cuda()
            e = model.encode_image(x).float()
            e = e / e.norm(dim=-1, keepdim=True)
            logits = e @ txt.T * model.logit_scale.exp().float()
            probs.append(torch.softmax(logits, dim=-1).cpu())
    torch.cuda.synchronize()
    model_s = time.perf_counter() - t
    probs = torch.cat(probs).numpy() if probs else np.zeros((0, len(CLASSES)))

    # ---- per-track vote
    acc = defaultdict(list)
    for (tid, f), p in zip(owners, probs):
        acc[tid].append(p)
    labels = {}
    for tid, ps in acc.items():
        p = np.mean(ps, axis=0)
        pd = {c: round(float(v), 3) for c, v in zip(CLASSES, p)}
        p_female = pd["woman"] + pd["girl"]
        label = "woman" if p_female >= a.blur_min else "man"
        labels[tid] = {"label": label, "p_female": round(p_female, 3), "p_child": round(pd["girl"] + pd["boy"], 3),
                       "p": pd, "n_crops": len(ps),
                       "crop_frames": [f for (t2, f) in owners if t2 == tid]}
    write_json(out / "track_labels.json", {str(k): v for k, v in sorted(labels.items())})

    n_rows = 0
    with open(out / "masks.jsonl", "w") as fh:
        for r in iter_jsonl(a.masks):
            lab = labels.get(r["tid"])
            if lab is None:
                continue
            r["prompt"] = lab["label"]
            fh.write(json.dumps(r) + "\n")
            n_rows += 1

    if a.save_crops:
        (out / "crops").mkdir(exist_ok=True)
        by_tid = defaultdict(list)
        for c, (tid, f) in zip(crops, owners):
            by_tid[tid].append(cv2.resize(c, (160, 160)))
        for tid, cs in by_tid.items():
            strip = cv2.cvtColor(np.hstack(cs), cv2.COLOR_RGB2BGR)
            cv2.putText(strip, f"{tid} {labels[tid]['label']} f={labels[tid]['p_female']:.2f}", (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(out / "crops" / f"{tid}.jpg"), strip)

    per_label = defaultdict(int)
    for v in labels.values():
        per_label[v["label"]] += 1
    meta = {
        "classifier": f"open_clip {a.model} ({a.pretrained})", "k": a.k, "min_gap": a.min_gap, "margin": a.margin,
        "masked": a.masked, "batch": a.batch, "blur_min": a.blur_min, "classes": CLASSES, "templates": TEMPLATES,
        "tracks": len(labels), "labels": dict(per_label), "crops": len(crops), "mask_rows": n_rows,
        "load_s": round(load_s, 1), "crop_extract_s": round(crop_s, 2), "model_s": round(model_s, 2),
        "crops_per_s": round(len(crops) / model_s, 1) if model_s else None,
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / GB, 3),
        "host_rss_gb": round(proc.memory_info().rss / GB, 3), "gpu": torch.cuda.get_device_name(0),
    }
    write_json(out / "clip_meta.json", meta)
    print(f"[clip] {len(labels)} tracks {dict(per_label)}  {len(crops)} crops  model {model_s:.1f}s "
          f"({meta['crops_per_s']} crops/s) + crops {crop_s:.1f}s  peak {meta['gpu_peak_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
