#!/usr/bin/env python3
"""Phase 1 + 3 runner — per frame: giant backbone on the frame (only tiles that contain a live track),
cross-attention ROI head on every tracked box → one observation per (frame, track).

    python3 demographics.py --out out/clip --head weights/demo_head.pt [--backbone dinov2-giant] [--every 1]

Writes  out/attrs.jsonl   {"f","tid","p_female","age_mean","age_std","quality","entropy","size_px","occ","sharp"}
        out/track_feats.npz  quality-weighted mean ROI embedding per track (identity-lock helper for phase 4)
        out/attrs_meta.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from backbone import GiantBackbone
from common import box_coverage, iter_jsonl, read_json, sharpness, write_json
from frames import frame_path
from roi_attention import DemographicHead


def load_head(path, in_dim, device):
    ck = torch.load(path, map_location="cpu")
    cfg = ck.get("cfg", {})
    if cfg.get("in_dim", in_dim) != in_dim:
        raise SystemExit(f"head was trained on in_dim {cfg.get('in_dim')} but backbone emits {in_dim}: use --backbone {cfg.get('backbone')}")
    head = DemographicHead(in_dim=in_dim, dim=cfg.get("dim", 512), n_queries=cfg.get("n_queries", 8),
                           n_layers=cfg.get("n_layers", 3), heads=cfg.get("heads", 8), max_keys=cfg.get("max_keys", 4096),
                           margin=cfg.get("margin", 0.15))
    head.load_state_dict(ck["state_dict"])
    return head.eval().to(device), cfg


def occlusion(box, others):
    """Largest fraction of this box covered by another tracked box that is *in front* of it (its bottom
    edge lower in the image = closer to the camera). 0 = free-standing."""
    return max([box_coverage(box, o) for o in others if o is not box and o[3] > box[3]] + [0.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", required=True, help="demo_head.pt from train_demographics.py")
    ap.add_argument("--backbone", default="dinov2-giant")
    ap.add_argument("--feat-stride", type=float, default=8, help="source pixels per token (smaller = denser, slower)")
    ap.add_argument("--tile-tokens", type=int, default=64)
    ap.add_argument("--every", type=int, default=1, help="run the demographic branch every N frames")
    ap.add_argument("--min-side", type=float, default=24, help="skip boxes shorter than this many pixels (still tracked)")
    ap.add_argument("--max-seconds", type=float, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out)
    frames_meta = read_json(next((out / "frames").glob("*/frames_meta.json")))
    fps, W, H = frames_meta["fps"], frames_meta["width"], frames_meta["height"]
    img_dir = frames_meta["img_dir"]
    per_frame = defaultdict(list)
    for r in iter_jsonl(out / "tracks.jsonl"):
        per_frame[r["f"]].append(r)
    frames = sorted(per_frame)
    if a.max_seconds > 0:
        frames = [f for f in frames if f < a.max_seconds * fps]
    frames = [f for f in frames if f % a.every == 0]

    bb = GiantBackbone(a.backbone, a.device, feat_stride=a.feat_stride, tile_tokens=a.tile_tokens)
    head, cfg = load_head(a.head, bb.dim, a.device)
    print(f"[demo] head cfg {cfg}")

    feat_sum, feat_w = defaultdict(lambda: np.zeros(head.dim, np.float32)), defaultdict(float)
    n_obs, t0 = 0, time.time()
    with open(out / "attrs.jsonl", "w") as fh:
        for i, f in enumerate(frames):
            rows = per_frame[f]
            boxes = [r["box"] for r in rows]
            keep = [j for j, b in enumerate(boxes) if min(b[2] - b[0], b[3] - b[1]) >= a.min_side]
            if not keep:
                continue
            bgr = cv2.imread(str(frame_path(img_dir, f)))
            if bgr is None:
                continue
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            kb = [boxes[j] for j in keep]
            feat = bb.feature_map(bgr, kb)
            tok_boxes = torch.tensor([bb.to_tokens(b, W, H) for b in kb], device=a.device, dtype=torch.float32)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                o = head([feat], [tok_boxes])
                d = head.decode({k: v.float() for k, v in o.items()})
            pooled = torch.nn.functional.normalize(o["pooled"].float(), dim=-1).cpu().numpy()
            for k, j in enumerate(keep):
                r, b = rows[j], boxes[j]
                x0, y0, x1, y1 = [int(round(v)) for v in b]
                sh = sharpness(gray[max(0, y0):y1, max(0, x0):x1]) if x1 > x0 and y1 > y0 else 0.0
                rec = {"f": f, "tid": r["tid"], "p_female": round(float(d["p_female"][k]), 4),
                       "age_mean": round(float(d["age_mean"][k]), 2), "age_std": round(float(d["age_std"][k]), 2),
                       "quality": round(float(d["quality"][k]), 4), "entropy": round(float(d["entropy"][k]), 4),
                       "size_px": round(float(b[3] - b[1]), 1), "occ": round(occlusion(b, boxes), 3), "sharp": round(sh, 3)}
                fh.write(json.dumps(rec) + "\n")
                w = float(d["quality"][k])
                feat_sum[r["tid"]] += w * pooled[k]
                feat_w[r["tid"]] += w
                n_obs += 1
            if i % 100 == 0:
                el = time.time() - t0
                print(f"[demo] frame {f} ({i + 1}/{len(frames)}) {n_obs} obs, {(i + 1) / max(el, 1e-6):.2f} fps")
    feats = {str(t): feat_sum[t] / max(feat_w[t], 1e-6) for t in feat_sum}
    np.savez(out / "track_feats.npz", **feats)
    write_json(out / "attrs_meta.json", {"backbone": a.backbone, "feat_stride": a.feat_stride, "every": a.every, "n_obs": n_obs,
                                         "n_frames": len(frames), "seconds": round(time.time() - t0, 1), "head": str(a.head), "head_cfg": cfg})
    print(f"[demo] {n_obs} observations in {time.time() - t0:.0f} s → {out / 'attrs.jsonl'}")


if __name__ == "__main__":
    main()
