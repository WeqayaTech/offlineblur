#!/usr/bin/env python3
"""Bake-off adapter B — tracking-by-detection: YOLO11x + boxmot BoT-SORT with a strong ReID.

The objective-analysis favourite for offline street tracking: a high-recall detector, a strong
appearance model (CLIP-ReID by default, far above OSNet on night crowds), and BoT-SORT's
camera-motion compensation (GMC) for the handheld pans. Writes the same tracks.jsonl as the MOTIP
adapter so the two are directly comparable.

    python3 botsort_track.py --frames-meta <seq>/frames_meta.json --out <out_dir> \
        --yolo yolo11x.pt --reid clip_market1501.pt --imgsz 1280 --conf 0.25 [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import frame_path, read_json, write_json


def make_tracker(reid_weights, device, half):
    import boxmot
    kw = dict(reid_weights=Path(reid_weights), device=device, half=half)
    for name in ("BotSort", "BoTSORT", "BoTSort"):
        cls = getattr(boxmot, name, None)
        if cls is not None:
            try:
                return cls(**kw)
            except TypeError:
                return cls(reid_weights=Path(reid_weights), device=device, half=half, with_reid=True)
    raise SystemExit("boxmot has no BotSort/BoTSORT class; check the boxmot version")


def run(frames_meta, out_dir, yolo="yolo11x.pt", reid="clip_market1501.pt", imgsz=1280, conf=0.25, gpu="0", half=True):
    out_dir = Path(out_dir)
    img_dir, W, H = frames_meta["img_dir"], frames_meta["width"], frames_meta["height"]
    n_frames = frames_meta["n_frames"]
    device = f"cuda:{gpu}"
    from ultralytics import YOLO
    det = YOLO(str(yolo))
    tracker = make_tracker(reid, device, half)

    import cv2
    tracks = out_dir / "tracks.jsonl"
    n, tids = 0, set()
    with open(tracks, "w") as out:
        for f in range(n_frames):
            im = cv2.imread(str(frame_path(img_dir, f)))
            if im is None:
                continue
            r = det.predict(im, imgsz=imgsz, conf=conf, classes=[0], device=int(gpu), verbose=False, half=half)[0]
            if r.boxes is not None and len(r.boxes):
                dets = np.concatenate([r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy()[:, None],
                                       np.zeros((len(r.boxes), 1))], axis=1).astype(np.float32)
            else:
                dets = np.empty((0, 6), np.float32)
            res = tracker.update(dets, im)  # -> [x1,y1,x2,y2,id,conf,cls,det_ind]
            for row in res:
                x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])
                box = [max(0.0, float(x1)), max(0.0, float(y1)), min(float(W), float(x2)), min(float(H), float(y2))]
                if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                    continue
                out.write(json.dumps({"f": f, "tid": tid, "box": [round(v, 1) for v in box],
                                      "score": round(float(row[5]), 3)}) + "\n")
                tids.add(tid)
                n += 1
            if f % 100 == 0:
                print(f"[botsort] frame {f}/{n_frames}, {n} boxes, {len(tids)} ids")
    write_json(out_dir / "tracks_meta.json", {"tracker": "botsort", "n_obs": n, "n_tracks": len(tids),
               "yolo": str(yolo), "reid": str(reid), "imgsz": imgsz, "conf": conf})
    print(f"[botsort] {n} boxes, {len(tids)} ids → {tracks}")
    return tracks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yolo", default="yolo11x.pt")
    ap.add_argument("--reid", default="clip_market1501.pt", help="boxmot ReID weight name (auto-downloads); clip_* is strongest")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--no-half", action="store_true")
    a = ap.parse_args()
    run(read_json(a.frames_meta), a.out, a.yolo, a.reid, a.imgsz, a.conf, a.gpu, not a.no_half)
