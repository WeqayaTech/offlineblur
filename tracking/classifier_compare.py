#!/usr/bin/env python3
"""Compare gender classifiers that ran on the SAME tracks (fast_blur.py with only --clip-model changed).

Because detector and tracker are identical, every difference is the classifier's. Per run it reports, on the
hand-labelled people (carried onto these tracks by gender_gt_eval.py, `reference_labels` of its --out json):
  errors@tau    women tracks below the run's threshold (escapes) / men tracks at or above it (false blurs)
  auc           probability a random labelled woman scores higher P(female) than a random labelled man
  margin        lowest woman minus highest man: > 0 means one threshold separates them all
  best_tau      the threshold range with the fewest errors (tuned on the same labels: optimistic)
and frame-weighted versions (a track counts by its number of frames). Pairwise: label agreement on ALL tracks and
the flipped tracks with both probabilities.

    python3 classifier_compare.py --gt-eval eval.json --tracks run_a.tracks.jsonl \
        --run "CLIP ViT-L=run_a.json" --run "PE-Core-L=run_b.json" [--tracks-b run_b.tracks.jsonl] --out cmp.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


def load_tracks(path):
    n = Counter()
    boxes = {}
    for line in open(path):
        r = json.loads(line)
        n[r["tid"]] += 1
        boxes[(r["f"], r["tid"])] = tuple(r["box"])
    return n, boxes


def auc(pos, neg):
    if not pos or not neg:
        return None
    s = sum(1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg)
    return s / (len(pos) * len(neg))


def errors(pf, gt, tau, w=None):
    w = w or {}
    esc = [t for t, g in gt.items() if g == "woman" and pf[t] < tau]
    fb = [t for t, g in gt.items() if g == "man" and pf[t] >= tau]
    fw = sum(w.get(t, 1) for t in esc)
    fm = sum(w.get(t, 1) for t in fb)
    return esc, fb, fw, fm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-eval", required=True, help="gender_gt_eval.py --out json (reference = these tracks)")
    ap.add_argument("--tracks", required=True, help="fast_blur.py --dump-tracks of the first run")
    ap.add_argument("--tracks-b", action="append", default=[], help="dump-tracks of other runs: must be identical")
    ap.add_argument("--run", action="append", required=True, help="NAME=fast_blur output .json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ev = json.loads(Path(a.gt_eval).read_text())
    gt = {int(k): v for k, v in ev["reference_labels"].items()}
    frames, boxes = load_tracks(a.tracks)
    for p in a.tracks_b:
        _, b2 = load_tracks(p)
        same = b2 == boxes
        print(f"[cmp] tracks identical to {a.tracks}: {same}  ({p})")
        if not same:
            raise SystemExit("tracks differ: the comparison would mix tracker and classifier effects")

    runs = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        d = json.loads(Path(path).read_text())
        runs[name] = {"meta": d["meta"], "pf": {int(k): v["p_female"] for k, v in d["labels"].items()},
                      "label": {int(k): v["label"] for k, v in d["labels"].items()}}

    report = {"gt_people": len(gt), "gt_women": sum(v == "woman" for v in gt.values()),
              "gt_men": sum(v == "man" for v in gt.values()), "runs": {}, "pairs": {}}
    for name, r in runs.items():
        pf, m = r["pf"], r["meta"]
        g = {t: v for t, v in gt.items() if t in pf}
        tau = m.get("blur_min", 0.25)
        pos = [pf[t] for t, v in g.items() if v == "woman"]
        neg = [pf[t] for t, v in g.items() if v == "man"]
        esc, fb, fw, fm = errors(pf, g, tau, frames)
        cands = sorted({round(x, 3) for x in pf.values()} | {tau})
        best = min(cands, key=lambda t: (len(errors(pf, g, t)[0]) + len(errors(pf, g, t)[1]), abs(t - tau)))
        nbest = sum(len(x) for x in errors(pf, g, best)[:2])
        lo = [t for t in cands if sum(len(x) for x in errors(pf, g, t)[:2]) == nbest]
        wf = sum(frames[t] for t, v in g.items() if v == "woman")
        mf = sum(frames[t] for t, v in g.items() if v == "man")
        res = {
            "classifier": m.get("clip"), "tau": tau, "tracks": len(pf), "tracks_labelled_woman": sum(
                v == "woman" for v in r["label"].values()),
            "crops": m.get("crops"), "clip_ms_per_frame": m.get("stage_ms_per_frame", {}).get("clip"),
            "ms_per_crop": round(1000 * m["stage_ms_per_frame"]["clip"] * m["frames"] / 1000 / max(1, m["crops"]), 2)
            if m.get("crops") else None,
            "end_to_end_hz": m.get("end_to_end_hz"), "gpu_peak_gb": m.get("gpu_peak_gb"), "load_s": m.get("load_s"),
            "gt_escapes@tau": [(t, pf[t]) for t in esc], "gt_false_blurs@tau": [(t, pf[t]) for t in fb],
            "gt_women_frames_escaped@tau": round(fw / wf, 3) if wf else None,
            "gt_men_frames_blurred@tau": round(fm / mf, 3) if mf else None,
            "auc": round(auc(pos, neg), 3) if auc(pos, neg) is not None else None,
            "min_woman": round(min(pos), 3) if pos else None, "max_man": round(max(neg), 3) if neg else None,
            "margin": round(min(pos) - max(neg), 3) if pos and neg else None,
            "best_tau_errors": nbest, "best_tau_range": [min(lo), max(lo)],
            "gt_women_p": sorted(round(x, 3) for x in pos), "gt_men_p": sorted(round(x, 3) for x in neg),
        }
        report["runs"][name] = res
        print(f"[cmp] {name:22s} tau {tau:.2f}  AUC {res['auc']}  margin {res['margin']}  "
              f"escapes {len(esc)}/{len(pos)}  false blurs {len(fb)}/{len(neg)}  "
              f"best tau {res['best_tau_range']} -> {nbest} errors  women tracks {res['tracks_labelled_woman']}/"
              f"{len(pf)}  clip {res['clip_ms_per_frame']} ms/frame", flush=True)
    for (na, ra), (nb, rb) in combinations(runs.items(), 2):
        common = sorted(set(ra["label"]) & set(rb["label"]))
        flips = [(t, ra["label"][t], ra["pf"][t], rb["label"][t], rb["pf"][t], gt.get(t)) for t in common
                 if ra["label"][t] != rb["label"][t]]
        report["pairs"][f"{na} vs {nb}"] = {
            "tracks": len(common), "agree": len(common) - len(flips),
            "flipped_frames_share": round(sum(frames[f[0]] for f in flips) / max(1, sum(frames[t] for t in common)), 3),
            "flips": [{"tid": t, na: [la, round(pa, 3)], nb: [lb, round(pb, 3)], "gt": g} for t, la, pa, lb, pb, g in flips],
        }
        print(f"[cmp] {na} vs {nb}: agree on {len(common) - len(flips)}/{len(common)} tracks")
        for t, la, pa, lb, pb, g in flips:
            print(f"      #{t:<3d} {na}: {la} {pa:.3f}   {nb}: {lb} {pb:.3f}   GT: {g or '-'}   {frames[t]} frames")
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
