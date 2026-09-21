#!/usr/bin/env python3
"""Detector-only bake-off — recall check for the person detector both trackers depend on.

Both SAMURAI (seeded once from frame 0) and BoT-SORT (detected fresh every frame) lean on the same
YOLO11x person detector. If it misses someone, no tracker downstream can ever find them — the gaps
you saw in the tracker comparison video are very likely detector recall, not a tracking failure. This
compares detector configs with NO tracking/ids involved, so recall is visible on its own.

Configs:
  baseline  single pass, imgsz=1280, conf=0.25 (what both tracker adapters use today)
  tiled     full-frame pass (catches large/near people) + overlapping 2x2 tile passes at each tile's
            native resolution (catches small/distant people the single downsampled pass misses),
            conf=0.10, merged with class-agnostic NMS

    python3 detect_bakeoff.py --frames-meta <seq>/frames_meta.json --out <out_dir> [--yolo yolo11x.pt]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import box_iou, frame_path, read_json, write_json


def nms(boxes, scores, iou_thresh=0.5):
    order = np.argsort(-np.asarray(scores))
    keep = []
    boxes = list(boxes)
    while len(order):
        i = order[0]
        keep.append(i)
        rest = order[1:]
        order = np.array([j for j in rest if box_iou(boxes[i], boxes[j]) < iou_thresh])
    return keep


def detect_baseline(det, im, imgsz, conf, gpu):
    r = det.predict(im, imgsz=imgsz, conf=conf, classes=[0], device=int(gpu), verbose=False)[0]
    if r.boxes is None or not len(r.boxes):
        return [], []
    return r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()


def tile_grid(W, H, nx=2, ny=2, overlap=0.2):
    tw, th = W / nx, H / ny
    ox, oy = tw * overlap, th * overlap
    tiles = []
    for j in range(ny):
        for i in range(nx):
            x0 = max(0, i * tw - ox); y0 = max(0, j * th - oy)
            x1 = min(W, (i + 1) * tw + ox); y1 = min(H, (j + 1) * th + oy)
            tiles.append((int(x0), int(y0), int(x1), int(y1)))
    return tiles


def detect_tiled(det, im, imgsz, conf, gpu, tiles):
    H, W = im.shape[:2]
    boxes, scores = detect_baseline(det, im, imgsz, conf, gpu)  # full-frame pass, large/near people
    for x0, y0, x1, y1 in tiles:
        crop = im[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        tb, ts = detect_baseline(det, crop, imgsz, conf, gpu)
        for b, s in zip(tb, ts):
            boxes.append([b[0] + x0, b[1] + y0, b[2] + x0, b[3] + y0])
            scores.append(s)
    if not boxes:
        return [], []
    keep = nms(boxes, scores, iou_thresh=0.5)
    return [boxes[i] for i in keep], [scores[i] for i in keep]


def run(frames_meta, out_dir, yolo="yolo11x.pt", imgsz=1280, conf_baseline=0.25, conf_tiled=0.10, gpu="0"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir, W, H, n_frames, fps = frames_meta["img_dir"], frames_meta["width"], frames_meta["height"], \
        frames_meta["n_frames"], frames_meta["fps"]

    from ultralytics import YOLO
    det = YOLO(yolo)
    tiles = tile_grid(W, H, nx=2, ny=2, overlap=0.2)

    fb = open(out_dir / "baseline.jsonl", "w")
    ft = open(out_dir / "tiled.jsonl", "w")
    nb = nt = 0
    counts = []
    for f in range(n_frames):
        im = cv2.imread(str(frame_path(img_dir, f)))
        if im is None:
            break
        bb, bs = detect_baseline(det, im, imgsz, conf_baseline, gpu)
        for b, s in zip(bb, bs):
            fb.write(json.dumps({"f": f, "box": [round(v, 1) for v in b], "score": round(float(s), 3)}) + "\n")
        nb += len(bb)
        tb, ts = detect_tiled(det, im, imgsz, conf_tiled, gpu, tiles)
        for b, s in zip(tb, ts):
            ft.write(json.dumps({"f": f, "box": [round(v, 1) for v in b], "score": round(float(s), 3)}) + "\n")
        nt += len(tb)
        counts.append((len(bb), len(tb)))
        if f % 50 == 0:
            print(f"[detect] frame {f}/{n_frames}: baseline {len(bb)}, tiled {len(tb)}")
    fb.close(); ft.close()

    arr = np.array(counts)
    summary = {"n_frames": n_frames, "baseline_total": nb, "tiled_total": nt,
               "baseline_mean_per_frame": round(float(arr[:, 0].mean()), 2),
               "tiled_mean_per_frame": round(float(arr[:, 1].mean()), 2),
               "baseline_max_per_frame": int(arr[:, 0].max()), "tiled_max_per_frame": int(arr[:, 1].max())}
    write_json(out_dir / "detect_summary.json", summary)
    print(json.dumps(summary, indent=1))

    # side-by-side video
    import subprocess
    panel_w = 800
    ph = int(H * panel_w / W)
    cw = panel_w * 2 + 8
    per_frame_b, per_frame_t = {}, {}
    for r in (json.loads(l) for l in open(out_dir / "baseline.jsonl")):
        per_frame_b.setdefault(r["f"], []).append(r["box"])
    for r in (json.loads(l) for l in open(out_dir / "tiled.jsonl")):
        per_frame_t.setdefault(r["f"], []).append(r["box"])

    def draw(im, boxes, name, n_total):
        th = max(1, im.shape[0] // 360)
        for b in boxes:
            x1, y1, x2, y2 = [int(v) for v in b]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 120), th)
        cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(im, f"{name} ({n_total})", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return im

    ff = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ph}",
          "-r", f"{fps:.6f}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
          "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(out_dir / "detect_compare.mp4")]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE)
    for f in range(n_frames):
        im = cv2.imread(str(frame_path(img_dir, f)))
        if im is None:
            break
        bb = per_frame_b.get(f, [])
        tb = per_frame_t.get(f, [])
        sx, sy = panel_w / W, ph / H
        la = draw(cv2.resize(im.copy(), (panel_w, ph)), [[v * (sx if i % 2 == 0 else sy) for i, v in enumerate(b)] for b in bb], "baseline", len(bb))
        lb = draw(cv2.resize(im.copy(), (panel_w, ph)), [[v * (sx if i % 2 == 0 else sy) for i, v in enumerate(b)] for b in tb], "tiled", len(tb))
        canvas = np.zeros((ph, cw, 3), np.uint8)
        canvas[:, :panel_w] = la
        canvas[:, panel_w + 8:] = lb
        proc.stdin.write(np.ascontiguousarray(canvas).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"[detect] video -> {out_dir / 'detect_compare.mp4'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yolo", default="yolo11x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf-baseline", type=float, default=0.25)
    ap.add_argument("--conf-tiled", type=float, default=0.10)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    run(read_json(a.frames_meta), a.out, a.yolo, a.imgsz, a.conf_baseline, a.conf_tiled, a.gpu)
