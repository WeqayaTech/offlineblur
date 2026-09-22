#!/usr/bin/env python3
"""Run a YOLO person detector over already-extracted frames and write a MOT-format detections file
for trackers that consume external detections (McByte's --det_path, this repo's common.mot_txt_to_jsonl).

    python3 gen_yolo_dets.py --frames-meta <seq>/frames_meta.json --out dets.txt \
        --yolo yolo26x.pt --imgsz 1280 --conf 0.25 [--gpu 0]

Line format (1-indexed frames, MOT16 convention): frame_id,-1,left,top,width,height,conf,-1,-1,-1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import frame_path, read_json


def run(frames_meta, out_path, yolo="yolo26x.pt", imgsz=1280, conf=0.25, gpu="0"):
    import cv2
    from ultralytics import YOLO

    img_dir, n_frames = frames_meta["img_dir"], frames_meta["n_frames"]
    det = YOLO(str(yolo))
    n = 0
    with open(out_path, "w") as out:
        for f in range(n_frames):
            im = cv2.imread(str(frame_path(img_dir, f)))
            if im is None:
                continue
            r = det.predict(im, imgsz=imgsz, conf=conf, classes=[0], device=int(gpu), verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = box.tolist()
                    w, h = x2 - x1, y2 - y1
                    out.write(f"{f + 1},-1,{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{c:.3f},-1,-1,-1\n")
                    n += 1
            if f % 100 == 0:
                print(f"[gen-dets] frame {f}/{n_frames}, {n} dets so far")
    print(f"[gen-dets] {n} dets ({yolo}, imgsz={imgsz}, conf={conf}) → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yolo", default="yolo26x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    run(read_json(a.frames_meta), a.out, a.yolo, a.imgsz, a.conf, a.gpu)
