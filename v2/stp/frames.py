#!/usr/bin/env python3
"""Video → MOT-style image sequence (<seq>/img1/%08d.jpg + seqinfo.ini).

Both transformer trackers (MeMOTR, MOTRv2) read DanceTrack/MOT17-layout jpg sequences; the demographic
phase re-reads the same jpgs so every phase sees identical pixels. Frame k (1-based file name) is
source frame k-1.

    python3 frames.py --video clip.mp4 --seq-root out/clip/frames --seq clip [--max-seconds 120]
"""
import argparse
import configparser
from pathlib import Path

import cv2

from common import iter_frames, video_meta, write_json


def extract(video, seq_root, seq_name, max_seconds=0.0, jpg_quality=95, resume=True) -> dict:
    meta = video_meta(video)
    fps = meta["fps"]
    max_frames = int(round(max_seconds * fps)) if max_seconds and max_seconds > 0 else None
    seq_dir = Path(seq_root) / seq_name
    img_dir = seq_dir / "img1"
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_path = seq_dir / "frames_meta.json"
    if resume and meta_path.exists():
        import json
        m = json.loads(meta_path.read_text())
        if m.get("max_frames") == max_frames and (img_dir / f"{m['n_frames']:08d}.jpg").exists():
            print(f"[frames] reuse {m['n_frames']} frames in {img_dir}")
            return m
    n = 0
    for f, im in iter_frames(video, max_frames):
        cv2.imwrite(str(img_dir / f"{f + 1:08d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        n += 1
        if n % 500 == 0:
            print(f"[frames] {n}")
    ini = configparser.ConfigParser()
    ini["Sequence"] = {"name": seq_name, "imDir": "img1", "frameRate": f"{fps:.3f}", "seqLength": str(n),
                       "imWidth": str(meta["width"]), "imHeight": str(meta["height"]), "imExt": ".jpg"}
    with open(seq_dir / "seqinfo.ini", "w") as fh:
        ini.write(fh)
    m = dict(meta, seq_name=seq_name, seq_dir=str(seq_dir), img_dir=str(img_dir), n_frames=n, max_frames=max_frames)
    write_json(meta_path, m)
    print(f"[frames] {n} frames ({meta['width']}x{meta['height']} @ {fps:.2f} fps) → {img_dir}")
    return m


def frame_path(img_dir, f: int) -> Path:
    return Path(img_dir) / f"{f + 1:08d}.jpg"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--seq-root", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--max-seconds", type=float, default=0.0)
    a = ap.parse_args()
    extract(a.video, a.seq_root, a.seq, a.max_seconds)
