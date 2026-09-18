#!/usr/bin/env python3
"""Contact sheet of every identity: its highest-quality crop with id, gender, P(female), age, observations, lock.
    python3 sheet.py --out out/clip [--png out/clip/review_sheet.jpg]"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import iter_jsonl, read_json
from frames import frame_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--png", default=None)
    ap.add_argument("--tile-h", type=int, default=260)
    ap.add_argument("--width", type=int, default=1600)
    a = ap.parse_args()
    out = Path(a.out)
    meta = read_json(next((out / "frames").glob("*/frames_meta.json")))
    ids, t2i = read_json(out / "identities.json"), read_json(out / "track_to_identity.json")
    best = {}
    for o in iter_jsonl(out / "attrs.jsonl"):
        k = t2i[str(o["tid"])]
        if k not in best or o["quality"] > best[k][0]:
            best[k] = (o["quality"], o["f"], o["tid"])
    boxes = {(r["f"], r["tid"]): r["box"] for r in iter_jsonl(out / "tracks.jsonl")}
    tiles = []
    for k in sorted(ids, key=int):
        d = ids[k]
        if int(k) not in best:
            continue
        q, f, tid = best[int(k)]
        x1, y1, x2, y2 = [int(v) for v in boxes[(f, tid)]]
        im = cv2.imread(str(frame_path(meta["img_dir"], f)))
        crop = im[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (max(48, int(crop.shape[1] * a.tile_h / crop.shape[0])), a.tile_h))
        crop = cv2.copyMakeBorder(crop, 40, 0, 0, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        col = (255, 0, 255) if d["gender"] == "female" else (255, 160, 0)
        cv2.putText(crop, "#%s %s %.2f" % (k, d["gender"][0].upper(), d["p_female"]), (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.putText(crop, "%sy n%d q%.2f %s" % (d["age_mean"], d["n_obs"], q, "L" if d["locked"] else ""), (2, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        tiles.append(crop)
    rows, row, w = [], [], 0
    for t in tiles:
        if w + t.shape[1] > a.width and row:
            rows.append(row)
            row, w = [], 0
        row.append(t)
        w += t.shape[1]
    rows.append(row)
    rh = a.tile_h + 40
    sheet = np.zeros((len(rows) * rh, a.width, 3), np.uint8)
    for i, r in enumerate(rows):
        x = 0
        for t in r:
            sheet[i * rh:i * rh + t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1]
    png = a.png or str(out / "review_sheet.jpg")
    cv2.imwrite(png, sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[sheet] {len(tiles)} identities → {png}")


if __name__ == "__main__":
    main()
