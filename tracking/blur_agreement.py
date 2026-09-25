#!/usr/bin/env python3
"""How closely two runs blur the same pixels: reads the per-frame union of blurred pixels written with --dump-blur
(fast_blur_edgetam.py / dist_edgetam.py) and reports, over all frames, the IoU of the blurred areas, the share of the
reference's blurred pixels the other run also blurs (recall) and the share of the other run's that the reference
blurs (precision), plus the frames where only one of them blurs anything.

    python3 blur_agreement.py reference_blur.jsonl other_blur.jsonl [other2_blur.jsonl ...]
"""
from __future__ import annotations

import json
import sys

from pycocotools import mask as mu


def _enc(r):
    return {"size": r["size"], "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]}


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["f"]] = _enc(r["rle"])
    return out


def compare_unions(ref, oth):
    """Same as compare() for in-memory {frame: rle} dicts (counts as str or bytes)."""
    return compare({f: _enc(r) for f, r in ref.items()}, {f: _enc(r) for f, r in oth.items()})


def compare(ref, oth):
    inter = a_ref = a_oth = 0.0
    only_ref = only_oth = 0
    worst = []
    for f in set(ref) | set(oth):
        r, o = ref.get(f), oth.get(f)
        ar = float(mu.area(r)) if r else 0.0
        ao = float(mu.area(o)) if o else 0.0
        i = float(mu.area(mu.merge([r, o], intersect=True))) if r and o else 0.0
        inter, a_ref, a_oth = inter + i, a_ref + ar, a_oth + ao
        only_ref += bool(ar) and not ao
        only_oth += bool(ao) and not ar
        u = ar + ao - i
        if u:
            worst.append((round(i / u, 3), f))
    worst.sort()
    return {"pixel_iou": round(inter / max(1.0, a_ref + a_oth - inter), 4), "recall": round(inter / max(1.0, a_ref), 4),
            "precision": round(inter / max(1.0, a_oth), 4), "frames_blurred_ref": len(ref), "frames_blurred_other": len(oth),
            "frames_only_ref": only_ref, "frames_only_other": only_oth, "worst_frames(iou,f)": worst[:8]}


if __name__ == "__main__":
    ref = load(sys.argv[1])
    for p in sys.argv[2:]:
        print(p, json.dumps(compare(ref, load(p))))
