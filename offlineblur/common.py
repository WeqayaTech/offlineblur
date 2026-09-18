#!/usr/bin/env python3
"""OfflineBlur — shared helpers: video metadata, sequential frame access, RLE masks, boxes, JSON."""
from __future__ import annotations

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


def iter_frames(path, max_frames=None, wanted=None):
    """Sequential decode, yielding (frame_index, bgr). With `wanted` (a set of frame indices) only
    those frames are decoded and yielded; the others are grabbed without conversion (fast) and
    iteration stops after the last wanted frame. Sequential access is the only reliable way to
    hit exact frame indices across codecs (seeking is inexact for many long-GOP streams)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    last = max(wanted) if wanted else None
    f = -1
    while True:
        f += 1
        if max_frames is not None and f >= max_frames:
            break
        if last is not None and f > last:
            break
        if wanted is not None and f not in wanted:
            if not cap.grab():
                break
            continue
        ok, im = cap.read()
        if not ok:
            break
        yield f, im
    cap.release()


def mask_to_rle(mask: np.ndarray) -> dict:
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def rle_to_mask(rle: dict) -> np.ndarray:
    from pycocotools import mask as mu
    return mu.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)


def mask_box(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


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


def pick_spread(items, k, key):
    """Up to k items spread evenly over an ordered list, taking the best `key` in each bucket —
    so the picks cover the whole life of a track instead of clustering on one moment."""
    items = list(items)
    if k <= 0 or len(items) <= k:
        return items
    n = len(items)
    out = []
    for b in range(k):
        lo = int(b * n / k)
        hi = max(lo + 1, int((b + 1) * n / k))
        out.append(max(items[lo:hi], key=key))
    return out


def runs(frames):
    """Sorted frame indices -> list of inclusive [start, end] runs of consecutive frames."""
    out = []
    for f in frames:
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def runs_overlap(a, b) -> bool:
    """True if two run lists share any frame (both sorted)."""
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] <= b[j][1] and b[j][0] <= a[i][1]:
            return True
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return False


def runs_overlap_frames(a, b) -> int:
    """Number of frames shared by two run lists (both sorted)."""
    i = j = n = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi >= lo:
            n += hi - lo + 1
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return n


def box_coverage(a, b):
    """Fraction of box a's area that lies inside box b."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area
