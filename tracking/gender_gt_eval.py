#!/usr/bin/env python3
"""Score any pipeline's blur against hand-labelled people: blurred share of each woman's frames, false-blur
share of each man's frames. Pipeline-agnostic: it only reads the masks each pipeline would pixelate.

Ground truth is labelled on one run's tracks (`--gt`: a hand-label JSON {woman: [tid], man: [tid]}, kept locally
and not versioned, e.g. labelled on RF-DETR+McByte tracks) and carried onto a reference person layer (`--reference`, a masks.jsonl
with `--reference-prompt` person, e.g. the SAM 3.1 tracked run) by per-frame mask IoU majority. The
reference layer defines WHICH frames count for a person, so a pipeline is also charged for frames its own
detector missed. A frame is "blurred" when the union of the pipeline's blur masks covers >= `--cover` of
the reference person's pixels.

    python3 gender_gt_eval.py --gt gt.json --gt-masks <run>/masks.jsonl --reference <sam31_gender>/masks.jsonl \
        --run "SAM 3=<sam3_gender>/masks.jsonl" --run "RF-DETR+CLIP=<rfclip>/masks.jsonl" --out eval.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import iter_jsonl, write_json


def enc(r):
    return {"size": r["size"], "counts": r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]}


def by_frame(path, prompt=None):
    out = defaultdict(list)
    for r in iter_jsonl(path):
        if prompt is None or r.get("prompt") == prompt:
            out[r["f"]].append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--gt-masks", required=True, help="masks.jsonl of the run the GT track ids belong to")
    ap.add_argument("--reference", required=True, help="masks.jsonl holding the reference person layer")
    ap.add_argument("--reference-prompt", default="person",
                    help="prompt of the reference person layer; 'any' = every row (e.g. a fast_blur.py --dump-masks "
                         "run, whose rows carry the gender label as prompt)")
    ap.add_argument("--run", action="append", required=True, help="NAME=masks.jsonl (blur masks = --blur-prompt)")
    ap.add_argument("--blur-prompt", default="woman")
    ap.add_argument("--cover", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    gt = json.loads(Path(a.gt).read_text())
    gt_label = {t: "woman" for t in gt["woman"]} | {t: "man" for t in gt["man"]}
    ref = by_frame(a.reference, None if a.reference_prompt == "any" else a.reference_prompt)
    src = by_frame(a.gt_masks)

    # carry GT onto reference identities: majority reference id per GT track, over frames with IoU >= 0.5
    votes = defaultdict(Counter)
    for f, rows in src.items():
        rows = [r for r in rows if r["tid"] in gt_label]
        if not rows or not ref.get(f):
            continue
        m = np.asarray(mu.iou([enc(r["rle"]) for r in rows], [enc(r["rle"]) for r in ref[f]], [0] * len(ref[f])))
        for i, r in enumerate(rows):
            j = int(m[i].argmax())
            if m[i, j] >= 0.5:
                votes[ref[f][j]["tid"]][gt_label[r["tid"]]] += 1
    ref_label = {}
    dropped = []
    for rid, c in votes.items():
        if len(c) == 1:
            ref_label[rid] = next(iter(c))
        else:
            dropped.append(rid)                    # GT tracks of both genders landed on one reference id
    print(f"[gt-eval] {len(gt_label)} labelled tracks -> {len(ref_label)} reference people "
          f"({sum(v == 'woman' for v in ref_label.values())} women, {sum(v == 'man' for v in ref_label.values())} men), "
          f"{len(dropped)} dropped as ambiguous", flush=True)

    frames_of = defaultdict(list)                  # rid -> [(f, rle)]
    for f, rows in ref.items():
        for r in rows:
            if r["tid"] in ref_label:
                frames_of[r["tid"]].append((f, r["rle"]))

    results = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        blur = by_frame(path, a.blur_prompt)
        union = {f: mu.merge([enc(r["rle"]) for r in rows]) for f, rows in blur.items() if rows}
        share = {}
        for rid, fr in frames_of.items():
            hit = 0
            for f, rle in fr:
                u = union.get(f)
                if u is None:
                    continue
                p = enc(rle)
                area = mu.area(p)
                if area and mu.area(mu.merge([p, u], intersect=True)) / area >= a.cover:
                    hit += 1
            share[rid] = hit / len(fr)
        w = [share[r] for r in share if ref_label[r] == "woman"]
        m = [share[r] for r in share if ref_label[r] == "man"]
        results[name] = {
            "women": len(w), "women_blurred_share_mean": round(statistics.mean(w), 3),
            "women_below_half": sum(x < 0.5 for x in w), "women_never": sum(x == 0 for x in w),
            "men": len(m), "men_false_blur_share_mean": round(statistics.mean(m), 3),
            "men_above_half": sum(x >= 0.5 for x in m),
            "per_person": {str(r): round(share[r], 3) for r in sorted(share)},
        }
        r = results[name]
        print(f"  {name:34s} women blurred {r['women_blurred_share_mean']:.3f} (<50%: {r['women_below_half']}/{r['women']}, "
              f"never: {r['women_never']})   men false-blur {r['men_false_blur_share_mean']:.3f} "
              f"(>=50%: {r['men_above_half']}/{r['men']})", flush=True)
    write_json(a.out, {"gt": a.gt, "reference": a.reference, "cover": a.cover,
                       "reference_labels": {str(k): v for k, v in sorted(ref_label.items())},
                       "dropped_reference_ids": dropped, "runs": results})


if __name__ == "__main__":
    main()
