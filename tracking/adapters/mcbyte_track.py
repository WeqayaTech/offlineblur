#!/usr/bin/env python3
"""Bake-off adapter E — McByte: training-free tracking-by-detection with SAM+Cutie mask propagation
as an association cue, via Roboflow's `trackers` library (pip install "trackers[mask]"; the [mask]
extra pulls torch/torchvision + rf-segment-anything + rf-cutie, all modern, no separate env needed —
unlike the original tstanczyk95/McByte research repo, which pins torch 1.12.1+cu116 and fails outright
on newer GPU architectures the old CUDA toolkit's nvrtc doesn't know how to JIT-compile for, e.g. this
pod's RTX 2000 Ada: "nvrtc: error: invalid value for --gpu-architecture (-arch)").

Architecturally, McByte is a ByteTrack derivative: any external detector's boxes drive track
birth/association as usual (like BoT-SORT), and a SAM-seeded, Cutie-propagated per-pixel mask is
layered on top purely as an extra association cue (mask-box overlap breaks ambiguous ID matches) —
masks never gate whether a track exists, only which detection it links to. This is unlike SAMURAI (a
single-object VOT tracker retrofitted for multi-person use) or SAM 3 (native multi-object, but its own
detector); McByte is the closest of the four to a drop-in mask-level upgrade over BoT-SORT.

Reuses this repo's own YOLO detections (gen_yolo_dets.py, e.g. yolo26x/yolo11x) via McByteTracker's
plain `sv.Detections` input — exactly the "combine with any detection model you already use" design
the library advertises. Like BoT-SORT (see botsort_track.py) and the original McByte, the birth/assoc
thresholds default high (track_activation_threshold=0.7, high_conf_det_threshold=0.6) relative to a
0.25-conf detector — lower them to match or real, visible detections are silently never tracked.

Writes tracks.jsonl (McByte's own boxes/ids) and masks.jsonl (RLE per id per frame, pulled from the
tracker's internal `_last_mask_output` — a label map + {tracker_id: label} dict, not part of the public
`update()` return value). No mask exists for frame 1: SAM only seeds after a tracklet has survived
`--min-mask-creation-frames` consecutive frames (default 3), and Cutie's first propagated frame lands
one frame after that.

    python3 mcbyte_track.py --frames-meta <seq>/frames_meta.json --out <out_dir> \
        --det-txt <out_dir>/dets.txt --fps 25 --track-activation-threshold 0.25 \
        --high-conf-det-threshold 0.25 [--gpu 0] [--no-masks]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import frame_path, read_json, write_json


def mask_to_rle(mask: np.ndarray):
    from pycocotools import mask as mu
    mask = np.squeeze(mask)  # mask_output.masks may carry a singleton leading/trailing axis
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(r, list):
        r = r[0]
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def load_dets_by_frame(det_txt):
    """MOT16 text (1-indexed frame,-1,x,y,w,h,conf,-1,-1,-1) -> {f (0-indexed): (boxes[N,4] xyxy, conf[N])}."""
    by_frame = defaultdict(list)
    with open(det_txt) as fh:
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 7:
                continue
            f = int(float(p[0])) - 1
            x, y, w, h, conf = map(float, p[2:7])
            by_frame[f].append((x, y, x + w, y + h, conf))
    return by_frame


def run(frames_meta, out_dir, det_txt, fps=25.0, track_activation_threshold=0.25, high_conf_det_threshold=0.25,
        lost_track_buffer=30, gpu="0", save_masks=True, min_mask_creation_frames=3, max_frames=None):
    import cv2
    import supervision as sv
    from trackers import McByteMaskConfig, McByteTracker

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir, n_frames = frames_meta["img_dir"], frames_meta["n_frames"]
    if max_frames:
        n_frames = min(n_frames, max_frames)

    dets_by_frame = load_dets_by_frame(det_txt)

    mask_config = McByteMaskConfig(device=f"cuda:{gpu}") if save_masks else None
    tracker = McByteTracker(
        lost_track_buffer=lost_track_buffer, frame_rate=fps,
        track_activation_threshold=track_activation_threshold, high_conf_det_threshold=high_conf_det_threshold,
        enable_mask_manager=save_masks, mask_config=mask_config, minimum_mask_creation_frames=min_mask_creation_frames)

    tracks_path = out_dir / "tracks.jsonl"
    masks_path = out_dir / "masks.jsonl"
    mask_out = open(masks_path, "w") if save_masks else None
    n_boxes, n_masks, tids = 0, 0, set()

    with open(tracks_path, "w") as out:
        for f in range(n_frames):
            im_bgr = cv2.imread(str(frame_path(img_dir, f)))
            rows = dets_by_frame.get(f, [])
            xyxy = np.array([r[:4] for r in rows], dtype=np.float32) if rows else np.empty((0, 4), np.float32)
            conf = np.array([r[4] for r in rows], dtype=np.float32) if rows else np.empty((0,), np.float32)
            detections = sv.Detections(xyxy=xyxy, confidence=conf)

            # McByte's SAM/Cutie backends consume the frame as RGB with no internal conversion (see
            # McByteTracker.update docstring) - a BGR frame silently degrades mask quality.
            frame_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB) if save_masks else None
            result = tracker.update(detections, frame=frame_rgb)

            mask_output = tracker._last_mask_output if save_masks else None
            for i in range(len(result)):
                tid = int(result.tracker_id[i])
                if tid < 0:
                    continue
                x1, y1, x2, y2 = result.xyxy[i].tolist()
                score = float(result.confidence[i]) if result.confidence is not None else 1.0
                out.write(json.dumps({"f": f, "tid": tid, "box": [round(v, 1) for v in (x1, y1, x2, y2)],
                                      "score": round(score, 3)}) + "\n")
                tids.add(tid)
                n_boxes += 1

                if mask_out is not None and mask_output is not None and mask_output.masks is not None:
                    # mask_output.masks is a one-hot stack [N_objects, H, W]; tracklet_mask_dict maps
                    # tracker_id -> index into that stack (NOT a label value to compare against).
                    idx = mask_output.tracklet_mask_dict.get(tid)
                    if idx is not None and idx < len(mask_output.masks):
                        m = mask_output.masks[idx]
                        if m.any():
                            mask_out.write(json.dumps({"f": f, "tid": tid, "rle": mask_to_rle(m)}) + "\n")
                            n_masks += 1

            if f % 50 == 0 or f == n_frames - 1:
                print(f"[mcbyte] frame {f + 1}/{n_frames}, {n_boxes} boxes, {n_masks} masks, {len(tids)} ids so far")

    if mask_out is not None:
        mask_out.close()
    write_json(out_dir / "tracks_meta.json", {"tracker": "mcbyte", "n_obs": n_boxes, "n_tracks": len(tids),
               "n_masks": n_masks, "det_txt": str(det_txt), "fps": fps,
               "track_activation_threshold": track_activation_threshold, "high_conf_det_threshold": high_conf_det_threshold,
               "lost_track_buffer": lost_track_buffer, "min_mask_creation_frames": min_mask_creation_frames,
               "masks": str(masks_path) if save_masks else None})
    print(f"[mcbyte] {n_boxes} boxes, {n_masks} masks, {len(tids)} ids → {tracks_path}")
    return tracks_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--det-txt", required=True, help="MOT-format detections file (frame,-1,x,y,w,h,conf,-1,-1,-1), e.g. from gen_yolo_dets.py")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--track-activation-threshold", type=float, default=0.25, help="McByte default is 0.7 - align with the detector's --conf, same class of birth-threshold trap as BoT-SORT (see botsort_track.py)")
    ap.add_argument("--high-conf-det-threshold", type=float, default=0.25, help="McByte default is 0.6 - align with the detector's --conf")
    ap.add_argument("--lost-track-buffer", type=int, default=30)
    ap.add_argument("--min-mask-creation-frames", type=int, default=3, help="consecutive visible frames before a tracklet gets a SAM/Cutie mask (1 = immediate)")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None, help="process only the first N frames (smoke-testing)")
    a = ap.parse_args()
    run(read_json(a.frames_meta), a.out, a.det_txt, a.fps, a.track_activation_threshold, a.high_conf_det_threshold,
        a.lost_track_buffer, a.gpu, not a.no_masks, a.min_mask_creation_frames, a.max_frames)
