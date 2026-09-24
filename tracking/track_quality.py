#!/usr/bin/env python3
"""GT-free identity-stability numbers for a tracks.jsonl ({f, tid, box[, filled]}): how often a person's track
breaks into several, and how often a tracked person has no box on a frame inside its own life.

  tracks            identities produced
  median_life       frames from first to last appearance
  short_tracks      tracks living < 25 frames (1 s at 25 fps): usually fragments or passers-by
  holes             frames inside a track's life with no box (the mask blinks out there)
  likely_fragments  a track that starts within `--rebirth-window` frames after another ended, at the same place
                    (IoU >= 0.3 or centres closer than half the box height): one person, two ids

    python3 track_quality.py run_a/tracks.jsonl run_b/tracks.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def near(a, b):
    ca = ((a[0] + a[2]) / 2, (a[1] + a[3]) / 2)
    cb = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    h = max(a[3] - a[1], b[3] - b[1])
    return iou(a, b) >= 0.3 or ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5 < 0.5 * h


def quality(path, window):
    tr = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        tr[r["tid"]][r["f"]] = (r["box"], r.get("filled", False))
    lives, holes, det_frames, filled = [], 0, 0, 0
    ends, starts = [], []
    for tid, fr in tr.items():
        fs = sorted(fr)
        life = fs[-1] - fs[0] + 1
        lives.append(life)
        holes += life - len(fs)
        det_frames += sum(1 for f in fs if not fr[f][1])
        filled += sum(1 for f in fs if fr[f][1])
        ends.append((fs[-1], fr[fs[-1]][0], tid))
        starts.append((fs[0], fr[fs[0]][0], tid))
    first = min(s[0] for s in starts)
    frags = 0
    for s, box, tid in starts:
        if s == first:
            continue
        if any(0 < s - e <= window and t != tid and near(b, box) for e, b, t in ends):
            frags += 1
    covered = sum(lives)
    return {"tracks": len(tr), "median_life": statistics.median(lives), "short_tracks_<25f": sum(l < 25 for l in lives),
            "holes": holes, "hole_share": round(holes / covered, 3), "gap_filled_frames": filled,
            "likely_fragments": frags}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="+")
    ap.add_argument("--rebirth-window", type=int, default=60)
    a = ap.parse_args()
    for p in a.tracks:
        print(f"{p}: {json.dumps(quality(p, a.rebirth_window))}")
