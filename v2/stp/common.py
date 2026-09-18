#!/usr/bin/env python3
"""Shared helpers (kept standalone so v2 does not import v1)."""
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


def iter_frames(path, max_frames=None):
    """Sequential decode → (frame_index, bgr). Sequential is the only exact way across long-GOP codecs."""
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


def box_coverage(a, b):
    """Fraction of box a's area inside box b."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


def sharpness(gray_crop: np.ndarray) -> float:
    """Variance of the Laplacian, normalised to a 0..1 gate at ~100 (motion blur / defocus → small)."""
    import cv2
    if gray_crop.size < 64:
        return 0.0
    v = float(cv2.Laplacian(gray_crop, cv2.CV_64F).var())
    return float(min(1.0, v / 100.0))
