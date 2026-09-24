#!/usr/bin/env python3
"""Replay one clip's saved person detections through McByte (box mode) under many association settings and score
each on identity stability — seconds per setting instead of a full pipeline run.

Detections come from `fast_blur.py --dump-dets` ({f, boxes, scores}); every setting sees exactly the same boxes, so
differences are the tracker's alone. Scores come from track_quality.py: `likely_fragments` (one person, several
ids — what looser matching should reduce) and `suspect_swaps` (one id jumping to another person — what looser
matching can cause).

    python3 mcbyte_sweep.py --dets dets.jsonl --fps 25 --out sweep/ [--lost-seconds 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from track_quality import quality


def run_one(frames, fps, cfg, lost_buffer, high_conf, out_path):
    import supervision as sv
    from trackers import McByteTracker
    from trackers.utils.iou import BIoU, DIoU, GIoU, IoU
    iou_obj = {"iou": IoU(), "giou": GIoU(), "diou": DIoU()}.get(cfg["iou"]) or BIoU(buffer_ratio=cfg["buffer"])
    tr = McByteTracker(lost_track_buffer=lost_buffer, frame_rate=fps, track_activation_threshold=high_conf,
                       high_conf_det_threshold=high_conf, enable_mask_manager=False, iou=iou_obj,
                       minimum_iou_threshold_first_assoc=cfg["a1"], minimum_iou_threshold_second_assoc=cfg["a2"],
                       minimum_iou_threshold_unconfirmed_assoc=cfg["au"])
    rows = 0
    with open(out_path, "w") as fh:
        for r in frames:
            b = np.array(r["boxes"], np.float32).reshape(-1, 4)
            s = np.array(r["scores"], np.float32)
            res = tr.update(sv.Detections(xyxy=b, confidence=s))
            for i in range(len(res)):
                t = int(res.tracker_id[i])
                if t >= 0:
                    fh.write(json.dumps({"f": r["f"], "tid": t, "box": [round(float(v), 1) for v in res.xyxy[i]]}) + "\n")
                    rows += 1
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", required=True)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lost-seconds", type=float, default=4.0)
    ap.add_argument("--high-conf", type=float, default=0.5)
    ap.add_argument("--window", type=int, default=100, help="frames back a re-birth counts as a fragment")
    ap.add_argument("--buffers", default="0.1,0.2,0.3,0.5", help="BIoU buffer ratios to try")
    ap.add_argument("--a1s", default="0.1", help="1st-association thresholds to try")
    ap.add_argument("--a2s", default="0.5,0.3,0.2", help="2nd-association thresholds to try")
    ap.add_argument("--aus", default="0.3,0.2", help="unconfirmed-association thresholds to try")
    ap.add_argument("--iou-a2s", default="0.3,0.2,0.1", help="plain-IoU 2nd-association thresholds to try")
    a = ap.parse_args()

    frames = [json.loads(l) for l in open(a.dets)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lost_buffer = int(round(a.lost_seconds * 30))
    fl = lambda v: [float(x) for x in v.split(",")]
    grid = [{"iou": "iou", "buffer": 0.0, "a1": 0.1, "a2": 0.5, "au": 0.3}]                 # McByte defaults
    for a2 in fl(a.iou_a2s):
        grid.append({"iou": "iou", "buffer": 0.0, "a1": 0.1, "a2": a2, "au": 0.3})
    for buf, a1, a2, au in itertools.product(fl(a.buffers), fl(a.a1s), fl(a.a2s), fl(a.aus)):
        grid.append({"iou": "biou", "buffer": buf, "a1": a1, "a2": a2, "au": au})
    results = []
    for cfg in grid:
        name = (f"{cfg['iou']}" + (f"{cfg['buffer']}" if cfg["iou"] == "biou" else "")
                + f"_a1{cfg['a1']}_a2{cfg['a2']}_au{cfg['au']}")
        p = out / f"{name}.tracks.jsonl"
        try:
            rows = run_one(frames, a.fps, cfg, lost_buffer, a.high_conf, p)
        except Exception as e:                                                               # e.g. rejected threshold
            print(f"{name}: FAILED {e}")
            continue
        q = quality(p, a.window)
        q.update(cfg=name, tracked_per_frame=round(rows / len(frames), 1))
        results.append(q)
    results.sort(key=lambda q: (q["likely_fragments"] + 2 * q["suspect_swaps"], q["tracks"]))
    print(f"{'setting':40s} tracks frags swaps tracked/f median_life short")
    for q in results:
        print(f"{q['cfg']:40s} {q['tracks']:6d} {q['likely_fragments']:5d} {q['suspect_swaps']:5d} "
              f"{q['tracked_per_frame']:9.1f} {q['median_life']:11} {q['short_tracks_<25f']:5d}")
    (out / "sweep.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
