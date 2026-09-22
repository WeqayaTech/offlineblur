#!/usr/bin/env python3
"""Render a tracker's own per-pixel masks — the differentiator over a box-only tracker.

Tracker-agnostic: works off any adapter's masks.jsonl (SAM2/SAMURAI, SAM3, or anything else that saves
RLE per-pixel masks in this schema), not just SAMURAI. Draws those masks directly: one flat colored
overlay per identity, no box-to-instance matching, no fallback rounded-box mask — what you see is
exactly what the tracker predicted. Pass --label to name the tracker in the on-screen overlay text.

Two outputs from one pass over the frames:
  --out         all tracked people, each a distinct semi-transparent color, id labelled (tracking showcase)
  --blur-out    (optional) exact per-pixel pixelation of the identities selected by identities.json /
                track_to_identity.json (from tracking/../v2/stp/aggregator.py) — everyone else untouched

    python3 render_masks.py --frames-meta <seq>/frames_meta.json --masks <out>/masks.jsonl --out track.mp4 \
        --label "SAM3" [--identities <out>/identities.json --track-to-identity <out>/track_to_identity.json \
         --blur-out blur.mp4 --min-p 0.6]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import frame_path, read_json

PALETTE = [(255, 0, 255), (0, 200, 255), (0, 255, 120), (255, 160, 0), (0, 120, 255), (200, 0, 255),
           (255, 255, 0), (120, 255, 0), (0, 255, 255), (255, 80, 80), (160, 200, 255), (255, 180, 220)]


def color(tid):
    return PALETTE[tid % len(PALETTE)]


def rle_decode(rle):
    from pycocotools import mask as mu
    return mu.decode(rle).astype(bool)


def pixelate(roi, block):
    h, w = roi.shape[:2]
    block = max(2, int(block))
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def ffmpeg_writer(path, W, H, fps):
    ff = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
          "-r", f"{fps:.6f}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
          "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(path)]
    return subprocess.Popen(ff, stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--masks", required=True, help="masks.jsonl written by a mask-saving adapter (samurai_track.py, sam3_track.py, ...)")
    ap.add_argument("--out", required=True, help="tracking-showcase video path")
    ap.add_argument("--label", default="tracker", help="name shown in the on-screen overlay text, e.g. 'SAM3' or 'SAMURAI'")
    ap.add_argument("--alpha", type=float, default=0.55, help="mask overlay opacity")
    ap.add_argument("--identities", default=None, help="identities.json from aggregator.py (enables --blur-out)")
    ap.add_argument("--track-to-identity", default=None, help="track_to_identity.json from aggregator.py")
    ap.add_argument("--blur-out", default=None, help="pixel-accurate blur video path")
    ap.add_argument("--min-p", type=float, default=0.6, help="blur an identity when P(female) >= min-p or locked female")
    a = ap.parse_args()

    fm = read_json(a.frames_meta)
    fps, W, H, img_dir, n = fm["fps"], fm["width"], fm["height"], fm["img_dir"], fm["n_frames"]

    per_frame = defaultdict(list)
    for line in open(a.masks):
        r = json.loads(line)
        per_frame[r["f"]].append((r["tid"], r["rle"]))
    n_masks = sum(len(v) for v in per_frame.values())
    print(f"[render-masks] {n_masks} masks across {len(per_frame)} frames")

    blur_ids, t2i = set(), {}
    if a.identities and a.track_to_identity and Path(a.identities).exists():
        ids = read_json(a.identities)
        t2i = read_json(a.track_to_identity)
        for k, d in ids.items():
            if d.get("class") in ("child", None):
                continue
            if d["p_female"] >= a.min_p or (d.get("locked") == "female"):
                blur_ids.add(int(k))
        print(f"[render-masks] blurring {len(blur_ids)} of {len(ids)} identities")

    def is_target(tid):
        k = t2i.get(str(tid))
        return k is not None and int(k) in blur_ids

    wt = ffmpeg_writer(a.out, W, H, fps)
    wb = ffmpeg_writer(a.blur_out, W, H, fps) if a.blur_out else None

    for f in range(n):
        im = cv2.imread(str(frame_path(img_dir, f)))
        if im is None:
            break
        entries = per_frame.get(f, [])

        overlay = im.copy()
        for tid, rle in entries:
            m = rle_decode(rle)
            if not m.any():
                continue
            c = np.array(color(tid), dtype=np.float32)
            blend = (im.astype(np.float32) * (1 - a.alpha) + c * a.alpha).astype(np.uint8)
            overlay[m] = blend[m]
            ys, xs = np.where(m)
            cx, cy = int(xs.mean()), int(ys.min())
            cv2.putText(overlay, str(tid), (max(0, cx - 8), max(12, cy - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color(tid), 2, cv2.LINE_AA)
        cv2.rectangle(overlay, (0, 0), (W, 34), (0, 0, 0), -1)
        cv2.putText(overlay, f"{a.label} per-pixel masks - {len(entries)} people, frame {f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        wt.stdin.write(np.ascontiguousarray(overlay).tobytes())

        if wb is not None:
            blurred = im.copy()
            for tid, rle in entries:
                if not is_target(tid):
                    continue
                m = rle_decode(rle)
                ys, xs = np.where(m)
                if len(xs) == 0:
                    continue
                y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                bh = y2 - y1
                roi = im[y1:y2, x1:x2]
                pix = pixelate(roi, 0.045 * bh)
                mm = m[y1:y2, x1:x2]
                region = blurred[y1:y2, x1:x2]
                region[mm] = pix[mm]
                blurred[y1:y2, x1:x2] = region
            wb.stdin.write(np.ascontiguousarray(blurred).tobytes())

        if f % 100 == 0:
            print(f"[render-masks] frame {f}/{n}")

    wt.stdin.close(); wt.wait()
    print(f"[render-masks] tracking video -> {a.out}")
    if wb is not None:
        wb.stdin.close(); wb.wait()
        print(f"[render-masks] blur video -> {a.blur_out}")


if __name__ == "__main__":
    main()
