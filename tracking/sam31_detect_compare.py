#!/usr/bin/env python3
"""How much of a tracked run does per-frame detection alone recover? GT-free, same clip, same model.

Compares a detector-only masks.jsonl (sam31_detect.py) against tracked runs by per-frame mask IoU:
  - vs the tracked blur prompt (`--tracked`, e.g. sam31_gender woman): recall = tracked observations
    matched by a detection in the same frame; extra = detections no tracked observation explains.
  - vs the tracked person control (`--control`): per person identity, the fraction of its frames that
    a detection covers. This is what a downstream tracker has to work with — a person covered in 60% of
    frames is a flicker a tracker must bridge; one covered in 0% is an escape no tracker can fix.

    python3 sam31_detect_compare.py --detect <det>/masks.jsonl --tracked <trk>/masks.jsonl \
        --control <person>/masks.jsonl --out <det>/compare.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import iter_jsonl, write_json


def by_frame(path, prompt=None):
    out = defaultdict(list)
    for r in iter_jsonl(path):
        if prompt is None or r.get("prompt") == prompt:
            out[r["f"]].append(r)
    return out


def ious(a, b):
    from pycocotools import mask as mu
    if not a or not b:
        return np.zeros((len(a), len(b)))
    enc = lambda rs: [{"size": r["rle"]["size"], "counts": r["rle"]["counts"].encode()} for r in rs]
    return np.asarray(mu.iou(enc(a), enc(b), [0] * len(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", required=True)
    ap.add_argument("--tracked", required=True, help="tracked masks.jsonl containing the blur prompt")
    ap.add_argument("--control", required=True, help="tracked masks.jsonl containing the person prompt")
    ap.add_argument("--prompt", default="woman")
    ap.add_argument("--control-prompt", default="person")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--covered", type=float, default=0.5, help="identity counts as covered at this frame fraction")
    ap.add_argument("--verdicts", default=None,
                    help="person_identities.json (sam3_gender_report.py) for --control: splits coverage by "
                         "verdict, so coverage on women reads as recall/flicker and on men as false positives")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    det = by_frame(a.detect)
    trk = by_frame(a.tracked, a.prompt)
    ctl = by_frame(a.control, a.control_prompt)
    frames = sorted(set(det) | set(trk) | set(ctl))

    n_trk = n_trk_hit = n_det = n_det_explained = 0
    ctl_frames = defaultdict(int)
    ctl_hit = defaultdict(int)
    trk_ids_hit = defaultdict(lambda: [0, 0])
    for f in frames:
        d, t, c = det.get(f, []), trk.get(f, []), ctl.get(f, [])
        m = ious(t, d)
        n_trk += len(t)
        n_det += len(d)
        if len(t) and len(d):
            hit = m.max(1) >= a.iou
            n_trk_hit += int(hit.sum())
            n_det_explained += int((m.max(0) >= a.iou).sum())
            for r, h in zip(t, hit):
                trk_ids_hit[r["tid"]][0] += int(h)
        for r in t:
            trk_ids_hit[r["tid"]][1] += 1
        mc = ious(c, d)
        for i, r in enumerate(c):
            ctl_frames[r["tid"]] += 1
            if len(d) and mc[i].max() >= a.iou:
                ctl_hit[r["tid"]] += 1

    ctl_frac = {tid: ctl_hit[tid] / n for tid, n in ctl_frames.items()}
    trk_frac = {tid: h / n for tid, (h, n) in trk_ids_hit.items()}
    hist = lambda v: {"0": sum(x == 0 for x in v), "(0,0.25)": sum(0 < x < .25 for x in v),
                      "[0.25,0.5)": sum(.25 <= x < .5 for x in v), "[0.5,0.75)": sum(.5 <= x < .75 for x in v),
                      "[0.75,1]": sum(x >= .75 for x in v)}
    res = {
        "iou": a.iou,
        "detections": n_det, "detections_per_frame": round(n_det / max(1, len(frames)), 2),
        "tracked_obs": n_trk,
        "tracked_obs_recalled": round(n_trk_hit / max(1, n_trk), 3),
        "detections_explained_by_tracked": round(n_det_explained / max(1, n_det), 3),
        "tracked_ids": len(trk_frac),
        "tracked_ids_frame_coverage_hist": hist(list(trk_frac.values())),
        "control_ids": len(ctl_frac),
        "control_ids_frame_coverage_hist": hist(list(ctl_frac.values())),
        "control_ids_covered": sum(v >= a.covered for v in ctl_frac.values()),
        "control_ids_never_detected": sum(v == 0 for v in ctl_frac.values()),
    }
    if a.verdicts:
        v = json.loads(Path(a.verdicts).read_text())
        by = defaultdict(list)
        for tid, frac in ctl_frac.items():
            by[v.get(str(tid), {}).get("verdict", "unknown")].append(frac)
        res["control_coverage_by_verdict"] = {
            k: {"ids": len(x), "mean_frame_coverage": round(float(np.mean(x)), 3), "hist": hist(x)}
            for k, x in sorted(by.items())}
    write_json(a.out, res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
