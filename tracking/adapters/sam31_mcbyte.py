#!/usr/bin/env python3
"""SAM 3.1 per-frame gender detections + McByte identities + per-track gender vote.

Stage 2 of the two-stage pipeline (stage 1: sam31_detect.py --text woman,man,child). Runs in its own env
(`trackers[mask]` needs numpy>=2, the sam3 repo pins numpy<2), reading stage 1's masks.jsonl from disk.

Per frame:
  1. merge  — the prompts are not exclusive: the same person is often both a `woman` and a `man`
     detection (SAM 3's own conflict rate on this clip is ~45%). Detections from all prompts are
     clustered by mask IoU (>= --merge-iou) into ONE person detection carrying every prompt's score;
     its box/mask are those of the highest-scoring member.
  2. track  — McByte (Roboflow `trackers`, a ByteTrack derivative that adds SAM-seeded, Cutie-propagated
     masks as an association cue) assigns identities class-agnostically. Tracking per gender would split
     a person every time the detector flips its mind.
  3. vote   — after the clip, each track's gender is the prompt with the largest summed score over the
     frames it was detected, so single-frame flips and false positives are outvoted. With
     `--min-share S`, a track whose `--blur-label` share of the score sum is >= S gets that label even
     when another prompt wins: an escape ships unblurred and an over-blur does not, so ties go to blur.

Masks written per track per frame: the SAM 3.1 detection mask when the track was matched this frame
(`src: det`); when McByte kept the track alive without a detection and its Cutie mask exists, that mask
(`src: cutie`) — this is what bridges the frames where the detector's score dips.

Outputs:
  tracks.jsonl         {f, tid, box, score, scores{prompt: s}, src}
  masks.jsonl          {f, tid, prompt=<track label>, score, src, rle}   render_gender.py-ready
  track_labels.json    {tid: {label, frames, det_frames, score_sum{..}, frac_detected{..}}}
  mcbyte_meta.json     config, counts, timing, memory

    python3 sam31_mcbyte.py --frames-meta <seq>/frames_meta.json --dets <det>/masks.jsonl --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import frame_path, iter_jsonl, read_json, write_json

GB = 1024 ** 3


def rle_obj(r):
    return {"size": r["size"], "counts": r["counts"].encode()}


def mask_to_rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(np.squeeze(mask).astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def merge_frame(rows, merge_iou):
    """Cluster one frame's detections across prompts. Returns [{rep: row, scores: {prompt: max}}]."""
    from pycocotools import mask as mu
    clusters = []
    for r in sorted(rows, key=lambda r: -r["score"]):
        best, best_iou = None, merge_iou
        if clusters:
            ious = np.asarray(mu.iou([rle_obj(r["rle"])], [rle_obj(c["rep"]["rle"]) for c in clusters],
                                     [0] * len(clusters)))[0]
            j = int(ious.argmax())
            if ious[j] >= best_iou:
                best = clusters[j]
        if best is None:
            clusters.append({"rep": r, "scores": {r["prompt"]: r["score"]}})
        else:
            best["scores"][r["prompt"]] = max(best["scores"].get(r["prompt"], 0.0), r["score"])
    return clusters


def vote(score_sum, blur_label="woman", min_share=None):
    """Track label from its per-prompt score sums (argmax, or blur_label once its share >= min_share)."""
    total = sum(score_sum.values())
    if min_share is not None and total > 0 and score_sum.get(blur_label, 0.0) / total >= min_share:
        return blur_label
    return max(score_sum, key=score_sum.get)


def run(frames_meta, dets_path, out_dir, merge_iou=0.5, high_conf=0.4, activation=0.4, lost_buffer=30,
        mask_manager=True, min_mask_frames=3, max_frames=None, blur_label="woman", min_share=None):
    import cv2
    import psutil
    import supervision as sv
    import torch
    from trackers import McByteMaskConfig, McByteTracker

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir, n_frames = frames_meta["img_dir"], int(frames_meta["n_frames"])
    if max_frames:
        n_frames = min(n_frames, max_frames)
    fps = float(frames_meta.get("fps", 25.0))

    dets = defaultdict(list)
    prompts = []
    for r in iter_jsonl(dets_path):
        dets[r["f"]].append(r)
        if r["prompt"] not in prompts:
            prompts.append(r["prompt"])

    t = time.perf_counter()
    tracker = McByteTracker(
        lost_track_buffer=lost_buffer, frame_rate=fps, track_activation_threshold=activation,
        high_conf_det_threshold=high_conf, enable_mask_manager=mask_manager,
        mask_config=McByteMaskConfig(device="cuda") if mask_manager else None,
        minimum_mask_creation_frames=min_mask_frames)
    init_s = time.perf_counter() - t
    proc = psutil.Process()

    votes = defaultdict(lambda: defaultdict(float))
    hits = defaultdict(lambda: defaultdict(int))
    det_frames, all_frames = defaultdict(int), defaultdict(int)
    mask_rows = []                      # (f, tid, score, src, rle) — labelled after the vote
    merge_s = track_s = io_s = read_s = 0.0
    n_clusters = n_cutie = 0
    host_peak = 0.0
    tracks_out = open(out_dir / "tracks.jsonl", "w")
    t_all = time.perf_counter()
    for f in range(n_frames):
        t0 = time.perf_counter()
        clusters = merge_frame(dets.get(f, []), merge_iou)
        t1 = time.perf_counter()
        im = cv2.cvtColor(cv2.imread(str(frame_path(img_dir, f))), cv2.COLOR_BGR2RGB) if mask_manager else None
        t2 = time.perf_counter()
        xyxy = (np.array([c["rep"]["box"] for c in clusters], dtype=np.float32) if clusters
                else np.empty((0, 4), np.float32))
        conf = np.array([max(c["scores"].values()) for c in clusters], dtype=np.float32)
        d = sv.Detections(xyxy=xyxy, confidence=conf, data={"cid": np.arange(len(clusters))})
        res = tracker.update(d, frame=im)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        seen = set()
        for i in range(len(res)):
            tid = int(res.tracker_id[i])
            if tid < 0:
                continue
            c = clusters[int(res.data["cid"][i])]
            seen.add(tid)
            det_frames[tid] += 1
            all_frames[tid] += 1
            for p, s in c["scores"].items():
                votes[tid][p] += s
                hits[tid][p] += 1
            tracks_out.write(json.dumps({"f": f, "tid": tid, "box": [round(float(v), 1) for v in res.xyxy[i]],
                                         "score": round(float(res.confidence[i]), 3),
                                         "scores": {p: round(s, 3) for p, s in c["scores"].items()},
                                         "src": "det"}) + "\n")
            mask_rows.append((f, tid, float(res.confidence[i]), "det", c["rep"]["rle"]))
        mo = tracker._last_mask_output if mask_manager else None
        if mo is not None and mo.masks is not None:
            # one-hot stack [N, H, W]; tracklet_mask_dict: {tracker_id: index into the stack}
            for tid, idx in mo.tracklet_mask_dict.items():
                tid = int(tid)
                if tid in seen or tid < 0 or idx >= len(mo.masks) or not mo.masks[idx].any():
                    continue
                all_frames[tid] += 1
                mask_rows.append((f, tid, 0.0, "cutie", mask_to_rle(mo.masks[idx])))
                n_cutie += 1
        t4 = time.perf_counter()
        merge_s += t1 - t0
        read_s += t2 - t1
        track_s += t3 - t2
        io_s += t4 - t3
        n_clusters += len(clusters)
        host_peak = max(host_peak, proc.memory_info().rss / GB)
        if f % 50 == 0:
            print(f"[sam31-mcbyte] frame {f}/{n_frames}  {len(clusters)} people-dets  {len(seen)} tracked  "
                  f"{len(votes)} ids  track {track_s / (f + 1):.3f} s/frame", flush=True)
    wall = time.perf_counter() - t_all
    tracks_out.close()

    labels = {}
    for tid, v in votes.items():
        label = vote(v, blur_label, min_share)
        total = sum(v.values())
        labels[tid] = {"label": label, "share": {p: round(s / total, 3) for p, s in sorted(v.items())}, "frames": all_frames[tid], "det_frames": det_frames[tid],
                       "score_sum": {p: round(s, 2) for p, s in sorted(v.items())},
                       "frac_detected": {p: round(hits[tid][p] / det_frames[tid], 3) for p in sorted(v)}}
    with open(out_dir / "masks.jsonl", "w") as fh:
        for f, tid, score, src, rle in mask_rows:
            if tid in labels:
                fh.write(json.dumps({"f": f, "tid": tid, "prompt": labels[tid]["label"], "score": round(score, 3),
                                     "src": src, "rle": rle}) + "\n")
    write_json(out_dir / "track_labels.json", {str(k): v for k, v in sorted(labels.items())})
    per_label = defaultdict(int)
    for v in labels.values():
        per_label[v["label"]] += 1
    meta = {
        "tracker": "mcbyte", "detector": "sam3.1", "dets": str(dets_path), "prompts": prompts,
        "merge_iou": merge_iou, "high_conf_det_threshold": high_conf, "track_activation_threshold": activation,
        "lost_track_buffer": lost_buffer, "mask_manager": mask_manager, "blur_label": blur_label,
        "min_share": min_share, "min_mask_creation_frames": min_mask_frames,
        "n_frames": n_frames, "people_dets": n_clusters, "ids": len(labels), "ids_per_label": dict(per_label),
        "mask_rows_det": len(mask_rows) - n_cutie, "mask_rows_cutie": n_cutie,
        "init_s": round(init_s, 1), "merge_s": round(merge_s, 2), "frame_read_s": round(read_s, 2),
        "tracker_s": round(track_s, 2), "io_s": round(io_s, 2), "wall_s": round(wall, 2),
        "tracker_fps": round(n_frames / track_s, 2), "total_fps": round(n_frames / wall, 2),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / GB, 3) if torch.cuda.is_available() else None,
        "host_rss_peak_gb": round(host_peak, 3),
    }
    write_json(out_dir / "mcbyte_meta.json", meta)
    print(f"[sam31-mcbyte] done: {len(labels)} ids {dict(per_label)}  McByte {meta['tracker_fps']} fps "
          f"(total {meta['total_fps']} fps)  cutie-bridged masks {n_cutie}  peak {meta['gpu_peak_gb']} GB",
          flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--dets", required=True, help="masks.jsonl from sam31_detect.py (multi-prompt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge-iou", type=float, default=0.5, help="cross-prompt mask IoU that means same person")
    ap.add_argument("--high-conf-det-threshold", type=float, default=0.4,
                    help="McByte default 0.6; ByteTrack's first-association cut. Detections below it (down to the "
                         "detector's own threshold) only extend existing tracks")
    ap.add_argument("--track-activation-threshold", type=float, default=0.4, help="McByte default 0.7")
    ap.add_argument("--lost-track-buffer", type=int, default=30)
    ap.add_argument("--no-mask-manager", action="store_true", help="IoU-only association (no SAM/Cutie)")
    ap.add_argument("--min-mask-creation-frames", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--blur-label", default="woman")
    ap.add_argument("--min-share", type=float, default=None,
                    help="label a track --blur-label when its share of the vote is >= this (default: argmax)")
    a = ap.parse_args()
    run(read_json(a.frames_meta), a.dets, a.out, a.merge_iou, a.high_conf_det_threshold,
        a.track_activation_threshold, a.lost_track_buffer, not a.no_mask_manager, a.min_mask_creation_frames,
        a.max_frames, a.blur_label, a.min_share)
