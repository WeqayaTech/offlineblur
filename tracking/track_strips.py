#!/usr/bin/env python3
"""Contact sheets for hand-labelling tracks: one row per track id, N box crops spread over its life, taken from the
original video (no mask, no blur, NO classifier output, so the labeller is not anchored by any model).

    python3 track_strips.py --video in.mp4 --tracks run.tracks.jsonl --out sheets/ [--per-row 6 --rows 8]

Write the labels as {"woman": [tid], "man": [tid], "unlabelled": [tid]} (confident labels only) and score with
gender_gt_eval.py (--gt-masks = --reference = that run's --dump-masks, --reference-prompt any).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tracks", required=True, help="fast_blur.py --dump-tracks jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-row", type=int, default=6)
    ap.add_argument("--rows", type=int, default=8, help="tracks per sheet")
    ap.add_argument("--h", type=int, default=200, help="crop height in the sheet")
    a = ap.parse_args()

    tr = defaultdict(dict)
    for line in open(a.tracks):
        r = json.loads(line)
        if not r.get("filled"):
            tr[r["tid"]][r["f"]] = r["box"]
    want = defaultdict(list)                                        # frame -> [(tid, slot, box)]
    for tid, fr in tr.items():
        fs = sorted(fr, key=lambda f: -((fr[f][2] - fr[f][0]) * (fr[f][3] - fr[f][1])))[:max(40, a.per_row)]
        fs = sorted(fs)                                             # biggest boxes, then spread over time
        pick = [fs[int(i * (len(fs) - 1) / max(1, a.per_row - 1))] for i in range(min(a.per_row, len(fs)))]
        for s, f in enumerate(dict.fromkeys(pick)):
            want[f].append((tid, s, fr[f]))
    crops = defaultdict(dict)
    cap = cv2.VideoCapture(a.video)
    f = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        H, W = im.shape[:2]
        for tid, s, (x1, y1, x2, y2) in want.get(f, []):
            mx, my = 0.15 * (x2 - x1), 0.08 * (y2 - y1)
            c = im[int(max(0, y1 - my)):int(min(H, y2 + my)), int(max(0, x1 - mx)):int(min(W, x2 + mx))]
            if c.size:
                c = cv2.resize(c, (max(1, int(c.shape[1] * a.h / c.shape[0])), a.h), interpolation=cv2.INTER_CUBIC)
                cv2.putText(c, str(f), (3, a.h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                crops[tid][s] = c
        f += 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tids = sorted(crops)
    for k in range(0, len(tids), a.rows):
        rows = []
        for tid in tids[k:k + a.rows]:
            tag = np.zeros((a.h, 110, 3), np.uint8)
            fr = tr[tid]
            cv2.putText(tag, f"#{tid}", (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(tag, f"{len(fr)} f", (6, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            rows.append(np.hstack([tag] + [crops[tid][s] for s in sorted(crops[tid])]))
        wmax = max(r.shape[1] for r in rows)
        sheet = np.vstack([np.pad(r, ((0, 4), (0, wmax - r.shape[1]), (0, 0))) for r in rows])
        cv2.imwrite(str(out / f"sheet_{k // a.rows:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"[strips] {len(tids)} tracks -> {out}")


if __name__ == "__main__":
    main()
