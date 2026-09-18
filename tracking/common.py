#!/usr/bin/env python3
"""Tracking core — shared helpers (self-contained; the tracking module does not depend on v1/v2)."""
from __future__ import annotations

import configparser
import json
from pathlib import Path

import numpy as np


def video_meta(path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 <= fps <= 240.0):
        fps = 30.0
    m = {"video": Path(path).name, "path": str(Path(path).resolve()), "fps": float(fps),
         "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
         "n_frames_container": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return m


def iter_frames(path, max_frames=None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    f = -1
    while True:
        f += 1
        if max_frames is not None and f >= max_frames:
            break
        ok, im = cap.read()
        if not ok:
            break
        yield f, im
    cap.release()


def extract_frames(video, seq_dir, max_seconds=0.0, jpg_quality=95) -> dict:
    """Video → <seq_dir>/img1/%08d.jpg + seqinfo.ini (DanceTrack/MOT layout). Frame file k = source frame k-1."""
    import cv2
    meta = video_meta(video)
    fps = meta["fps"]
    max_frames = int(round(max_seconds * fps)) if max_seconds and max_seconds > 0 else None
    seq_dir = Path(seq_dir)
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)
    done = seq_dir / "frames_meta.json"
    if done.exists():
        m = json.loads(done.read_text())
        if m.get("max_frames") == max_frames and (img_dir / f"{m['n_frames']:08d}.jpg").exists():
            print(f"[frames] reuse {m['n_frames']} in {img_dir}")
            return m
    n = 0
    for f, im in iter_frames(video, max_frames):
        cv2.imwrite(str(img_dir / f"{f + 1:08d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        n += 1
    ini = configparser.ConfigParser()
    ini["Sequence"] = {"name": seq_dir.name, "imDir": "img1", "frameRate": f"{fps:.3f}", "seqLength": str(n),
                       "imWidth": str(meta["width"]), "imHeight": str(meta["height"]), "imExt": ".jpg"}
    with open(seq_dir / "seqinfo.ini", "w") as fh:
        ini.write(fh)
    m = dict(meta, seq=seq_dir.name, seq_dir=str(seq_dir), img_dir=str(img_dir), n_frames=n, max_frames=max_frames)
    write_json(done, m)
    print(f"[frames] {n} frames ({meta['width']}x{meta['height']} @ {fps:.2f}) → {img_dir}")
    return m


def frame_path(img_dir, f: int) -> Path:
    return Path(img_dir) / f"{f + 1:08d}.jpg"


def mot_txt_to_jsonl(txt_path, out_path, w=None, h=None) -> int:
    """MOT16 text (frame,id,x,y,w,h,...) 1-based frames → tracks.jsonl {f,tid,box[x1,y1,x2,y2],score}."""
    n = 0
    with open(txt_path) as fh, open(out_path, "w") as out:
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            f = int(float(p[0])) - 1
            tid = int(float(p[1]))
            x, y, bw, bh = map(float, p[2:6])
            box = [x, y, x + bw, y + bh]
            if w and h:
                box = [max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3])]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            out.write(json.dumps({"f": f, "tid": tid, "box": [round(v, 1) for v in box], "score": 1.0}) + "\n")
            n += 1
    return n


def iter_jsonl(path):
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
