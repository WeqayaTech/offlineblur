#!/usr/bin/env python3
"""OfflineBlur stage 1 — detect, segment and track every person in every frame.

One model does all three: a YOLO instance-segmentation checkpoint (default yolo11x-seg, COCO
class 0 = person) on every frame at --imgsz, with BoT-SORT + ReID as the tracker so a person keeps
the same track id through short occlusions and crossings. Nothing is decided here: every tracked
box >= --conf is logged with its full-resolution pixel mask (log-raw rule) so stages 2-4 can be
re-run with different policies without touching the GPU again.

    python3 s1_detect_track.py --video clip.mp4 --out out/clip [--imgsz 1280] [--max-seconds 120]

Writes  out/tracks.jsonl   one line per frame: {"f", "t", "objs": [{"tid", "box", "conf", "area", "rle"}]}
        out/s1_meta.json   video metadata, real frame count, settings, timing
        out/tracker.yaml   the BoT-SORT config actually used
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from common import mask_to_rle, video_meta, write_json

WEIGHTS_DIR = os.environ.get("OFFLINEBLUR_WEIGHTS", "/workspace/offlineblur/weights")

# BoT-SORT with appearance re-identification. track_buffer (frames a lost track is kept alive) is
# filled in from --track-buffer-s so it scales with the video's fps.
TRACKER_YAML = """tracker_type: botsort
track_high_thresh: {high}
track_low_thresh: {low}
new_track_thresh: {new}
track_buffer: {buffer}
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
proximity_thresh: 0.5
appearance_thresh: 0.8
with_reid: True
model: {reid}
"""


def resolve_weights(name: str) -> str:
    p = Path(name)
    if p.exists():
        return str(p)
    cand = Path(WEIGHTS_DIR) / p.name
    return str(cand) if cand.exists() else name      # ultralytics auto-downloads a bare name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det-model", default="yolo11x-seg.pt", help="any ultralytics *-seg checkpoint")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size (long side); raise for 4K with tiny people")
    ap.add_argument("--conf", type=float, default=0.10, help="floor passed to the detector (the tracker's 2nd stage uses low boxes)")
    ap.add_argument("--iou", type=float, default=0.60, help="NMS IoU; high so overlapping people in a crowd survive")
    ap.add_argument("--track-high", type=float, default=0.50)
    ap.add_argument("--track-low", type=float, default=0.10)
    ap.add_argument("--new-track", type=float, default=0.60, help="conf needed to START a track (junk suppression)")
    ap.add_argument("--track-buffer-s", type=float, default=5.0, help="seconds a lost track is kept for re-association")
    ap.add_argument("--reid", default="auto", help="'auto' = detector features; or a *-cls.pt checkpoint")
    ap.add_argument("--max-seconds", type=float, default=0, help="0 = whole video")
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-half", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = video_meta(a.video)
    fps, W, H = meta["fps"], meta["width"], meta["height"]
    max_frames = None if a.max_seconds <= 0 else int(a.max_seconds * fps)
    det_model = resolve_weights(a.det_model)
    cfg = out / "tracker.yaml"
    cfg.write_text(TRACKER_YAML.format(high=a.track_high, low=a.track_low, new=a.new_track,
                                       buffer=max(1, int(a.track_buffer_s * fps)), reid=a.reid))

    import torch
    torch.backends.cudnn.benchmark = True
    from ultralytics import YOLO
    model = YOLO(det_model)
    print(f"[s1] {meta['video']}: {W}x{H} @ {fps:.3f} fps, container says {meta['n_frames_container']} frames; "
          f"model {det_model} imgsz {a.imgsz}; tracker {cfg}", flush=True)

    fh = open(out / "tracks.jsonl", "w")
    t0 = time.time()
    n = n_objs = n_box_masks = 0
    tids = set()
    stream = model.track(source=a.video, stream=True, persist=True, tracker=str(cfg), classes=[0],
                         conf=a.conf, iou=a.iou, imgsz=a.imgsz, half=not a.no_half, device=a.device,
                         retina_masks=True, verbose=False)
    for r in stream:
        f = n
        n += 1
        objs = []
        b = r.boxes
        if b is not None and len(b) and b.id is not None:
            ids = b.id.int().tolist()
            xyxy = b.xyxy.cpu().numpy()
            confs = b.conf.cpu().numpy()
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None
            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                m = None
                if masks is not None and i < len(masks) and masks[i].shape == (H, W):
                    m = masks[i] > 0.5
                if m is None or not m.any():
                    # no usable mask (should be rare with retina_masks): fall back to the box
                    m = np.zeros((H, W), dtype=bool)
                    m[max(0, int(y1)):min(H, int(np.ceil(y2))), max(0, int(x1)):min(W, int(np.ceil(x2)))] = True
                    n_box_masks += 1
                objs.append({"tid": int(tid), "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                             "conf": round(float(confs[i]), 4), "area": int(m.sum()), "rle": mask_to_rle(m)})
                tids.add(int(tid))
        fh.write(json.dumps({"f": f, "t": round(f / fps, 3), "objs": objs}) + "\n")
        n_objs += len(objs)
        if n % 500 == 0:
            el = time.time() - t0
            print(f"  [s1] frame {n} · {n/el:.1f} fps · {len(tids)} tracks so far · eta "
                  f"{((meta['n_frames_container'] if max_frames is None else max_frames) - n)/max(n/el,1e-6)/60:.0f} min", flush=True)
        if max_frames is not None and n >= max_frames:
            break
    fh.close()
    el = time.time() - t0
    write_json(out / "s1_meta.json", {**meta, "n_frames": n, "n_tracks": len(tids), "n_objs": n_objs,
                                      "n_box_fallback_masks": n_box_masks, "seconds": round(el, 1),
                                      "fps_processing": round(n / max(el, 1e-6), 2),
                                      "settings": {**vars(a), "det_model": det_model},
                                      "tracker_cfg": cfg.read_text()})
    print(f"[s1] done {meta['video']}: {n} frames, {len(tids)} raw tracks, {n_objs} person-frames in "
          f"{el/60:.1f} min ({n/max(el,1e-6):.1f} fps)", flush=True)


if __name__ == "__main__":
    main()
