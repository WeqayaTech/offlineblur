#!/usr/bin/env python3
"""Bake-off adapter C — SAMURAI: SAM2.1 video predictor with motion-aware memory (yangchris11/samurai).

SAMURAI is a zero-shot VOT tracker, not an open-set multi-object tracker: every identity needs an
initial box prompt, there is no built-in detector, and (see below) its motion-aware Kalman filter is
singleton state on the model — one SAM2 object per predictor pass, no true batching. To cover a whole
clip we periodically re-run YOLO (every --rescan-every frames), match its detections against the most
recent box of every object already being tracked, and give any unmatched detection its own fresh
single-object SAM2 pass. Without this, SAMURAI can only ever track people already visible in the first
scan — anyone entering later is invisible to it (measured: 43% of all detections across a 12s clip
belonged to people never tracked, seeding frame 0 only).

A late-seeded object cannot just be added at `frame_idx=scan_frame` on the full video: SAMURAI's
motion-aware memory-selection code (`_prepare_memory_conditioned_features` in sam2_base.py) scores
recent frames with a hardcoded `range(frame_idx - 1, 1, -1)` over `non_cond_frame_outputs`, which
assumes tracking always started at absolute frame 0 — seeding at frame 25 makes it look up frame 25's
entry there, but frame 25 is the *conditioning* frame (stored in `cond_frame_outputs`), so it
KeyErrors. Fix: give each new entrant its own frame sequence that starts at local index 0 — a symlinked
slice of the frames from its entry point onward — then map the returned local indices back to absolute
frame numbers when writing tracks.jsonl.

Writes `tracks.jsonl` (boxes, same schema as the other adapters, for compare.py) and, since SAM2's mask
is already computed every frame and otherwise thrown away, `masks.jsonl` (RLE-encoded per-pixel mask per
frame per id) — the actual differentiator over a box-only tracker, for pixel-accurate rendering/blur.

    python3 samurai_track.py --frames-meta <seq>/frames_meta.json --out <out_dir> \
        --samurai-repo /root/tracking/samurai --checkpoint <ckpt>.pt --model-size large \
        --yolo yolo11x.pt --conf 0.4 --rescan-every 25 [--gpu 0] [--no-masks]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import box_iou, frame_path, read_json, write_json


def make_frame_slice(img_dir, start_frame, n_frames, slice_dir):
    """Symlink frames [start_frame, n_frames) into slice_dir, renamed 1-based from 1 (local frame 0)."""
    d = Path(slice_dir)
    if d.exists():
        return d
    d.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(range(start_frame, n_frames)):
        src = Path(frame_path(img_dir, f)).resolve()
        os.symlink(src, d / f"{i + 1:08d}.jpg")
    return d

MODEL_CFG = {
    "large": "configs/samurai/sam2.1_hiera_l.yaml",
    "base_plus": "configs/samurai/sam2.1_hiera_b+.yaml",
    "small": "configs/samurai/sam2.1_hiera_s.yaml",
    "tiny": "configs/samurai/sam2.1_hiera_t.yaml",
}


def mask_to_box(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0]


def mask_to_rle(mask: np.ndarray):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def run(frames_meta, out_dir, samurai_repo, checkpoint, model_size="large",
        yolo="yolo11x.pt", conf=0.4, rescan_every=25, iou_new=0.3, lookback=10, gpu="0", save_masks=True):
    import torch
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir, W, H = frames_meta["img_dir"], frames_meta["width"], frames_meta["height"]
    n_frames = frames_meta["n_frames"]
    device = f"cuda:{gpu}"

    sys.path.insert(0, str(Path(samurai_repo) / "sam2"))
    from sam2.build_sam import build_sam2_video_predictor

    from ultralytics import YOLO
    det = YOLO(yolo)
    predictor = build_sam2_video_predictor(MODEL_CFG[model_size], checkpoint, device=device)

    # SAMURAI's motion-aware Kalman filter lives as singleton state on the model itself
    # (self.kf_mean / self.kf_covariance / self.stable_frames / self.frame_cnt, set once in
    # __init__ and never reset). It has no notion of a batch dimension, so tracking >1 object on
    # one inference_state crashes inside _forward_sam_heads — one full single-object pass per person,
    # resetting that state in between.
    #
    # SAMURAI also has no detector of its own, so a single frame-0 seed only ever covers people already
    # visible at the start. To cover the whole clip, re-run YOLO every `rescan_every` frames; any
    # detection that doesn't match (by IoU) the most recent box of an already-tracked object gets its
    # own fresh SAM2 pass seeded at that frame and propagated to the end of the clip.
    tracks = out_dir / "tracks.jsonl"
    masks_path = out_dir / "masks.jsonl"
    boxes_by_oid: dict[int, dict[int, list]] = {}
    n, next_id, n_scans_with_new = 0, 0, 0
    mask_out = open(masks_path, "w") if save_masks else None
    with open(tracks, "w") as out, torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for scan_frame in range(0, n_frames, rescan_every):
            im = cv2.imread(str(frame_path(img_dir, scan_frame)))
            r = det.predict(im, imgsz=1280, conf=conf, classes=[0], device=int(gpu), verbose=False)[0]
            dets = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None and len(r.boxes) else []

            new_boxes = []
            for db in dets:
                matched = False
                for fbmap in boxes_by_oid.values():
                    ref_box = None
                    for lb in range(lookback + 1):
                        ref_box = fbmap.get(scan_frame - lb)
                        if ref_box is not None:
                            break
                    if ref_box is not None and box_iou(db, ref_box) > iou_new:
                        matched = True
                        break
                if not matched:
                    new_boxes.append(db)

            if new_boxes:
                slice_dir = make_frame_slice(img_dir, scan_frame, n_frames, out_dir / "_slices" / f"s{scan_frame}")

            for db in new_boxes:
                oid = next_id
                next_id += 1
                predictor.kf_mean = None
                predictor.kf_covariance = None
                predictor.stable_frames = 0
                predictor.frame_cnt = 0
                state = predictor.init_state(str(slice_dir), offload_video_to_cpu=True)
                predictor.add_new_points_or_box(state, frame_idx=0, obj_id=oid, box=db)

                fbmap, n_obj = {}, 0
                for local_idx, obj_ids, masks in predictor.propagate_in_video(state):
                    frame_idx = scan_frame + local_idx
                    m = (masks[0][0].cpu().numpy() > 0.0)
                    bbox = mask_to_box(m)
                    if bbox is None:
                        continue
                    bbox = [max(0.0, bbox[0]), max(0.0, bbox[1]), min(float(W), bbox[2]), min(float(H), bbox[3])]
                    if bbox[2] - bbox[0] < 1 or bbox[3] - bbox[1] < 1:
                        continue
                    out.write(json.dumps({"f": frame_idx, "tid": int(oid), "box": [round(v, 1) for v in bbox],
                                          "score": 1.0}) + "\n")
                    if mask_out is not None:
                        mask_out.write(json.dumps({"f": frame_idx, "tid": int(oid), "rle": mask_to_rle(m)}) + "\n")
                    fbmap[frame_idx] = bbox
                    n_obj += 1
                boxes_by_oid[oid] = fbmap
                n += n_obj
                predictor.reset_state(state)
                del state
                torch.cuda.empty_cache()

            if new_boxes:
                n_scans_with_new += 1
            print(f"[samurai] scan@{scan_frame}/{n_frames}: {len(dets)} detections, "
                  f"{len(new_boxes)} new people, {len(boxes_by_oid)} objects total, {n} boxes so far")

    if mask_out is not None:
        mask_out.close()
    write_json(out_dir / "tracks_meta.json", {"tracker": "samurai", "n_obs": n, "n_tracks": len(boxes_by_oid),
               "model_size": model_size, "checkpoint": str(checkpoint), "yolo": str(yolo), "conf": conf,
               "rescan_every": rescan_every, "iou_new": iou_new, "lookback": lookback,
               "masks": str(masks_path) if save_masks else None})
    print(f"[samurai] {n} boxes, {len(boxes_by_oid)} ids (rescanned every {rescan_every} frames) → {tracks}")
    return tracks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samurai-repo", required=True, help="path to the yangchris11/samurai checkout (its sam2/ submodule is used)")
    ap.add_argument("--checkpoint", required=True, help="SAM2.1 checkpoint .pt matching --model-size")
    ap.add_argument("--model-size", default="large", choices=list(MODEL_CFG))
    ap.add_argument("--yolo", default="yolo11x.pt")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--rescan-every", type=int, default=25, help="re-run YOLO every N frames to seed new entrants")
    ap.add_argument("--iou-new", type=float, default=0.3, help="IoU below which a detection is treated as a new person")
    ap.add_argument("--lookback", type=int, default=10, help="frames to look back for an existing object's last box when matching")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--no-masks", action="store_true", help="skip writing masks.jsonl (RLE per-pixel masks); tracks.jsonl boxes only")
    a = ap.parse_args()
    run(read_json(a.frames_meta), a.out, a.samurai_repo, a.checkpoint, a.model_size, a.yolo, a.conf,
        a.rescan_every, a.iou_new, a.lookback, a.gpu, not a.no_masks)
