#!/usr/bin/env python3
"""Score a SAM 3 multi-prompt run: does the "woman" prompt work as the blur selector?

Reads one `masks.jsonl` written by `adapters/sam3_track.py` with prompts `woman,man,person` and answers
the only two questions that decide whether prompt-as-selector is viable in production:

  ESCAPES   a `person` the `woman` and `man` prompts both ignore. SAM 3 sees a human, no gender concept
            fires. If that person is a woman she ships unblurred — the failure that actually costs us.
  CONFLICTS the same physical person claimed by both `woman` and `man` (overlapping masks). SAM 3 is
            internally undecided, so any blur/no-blur call there is a coin flip.

Matching is per-pixel mask IoU between prompt outputs on the same frame (`pycocotools`), never box IoU:
SAM 3 gives real masks and the whole point of this pipeline is per-pixel work. Everything is rolled up
to the `person` identity, because a selector that flickers frame to frame on one person is not usable —
`--min-frac` is the share of an identity's frames a concept must cover to own it.

No ground truth is involved: these are GT-free agreement metrics between SAM 3's own concepts. They
say where the model is inconsistent, not who is actually a woman. Eyeball the render for that.

Writes metrics.json (summary), person_identities.json (per-identity verdicts) and prints a table.

    python3 sam3_gender_report.py --masks <out>/masks.jsonl --frames-meta <seq>/frames_meta.json \
        --out <out>/gender [--blur-prompt woman --other-prompt man --control-prompt person]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json, write_json


def _rle(r):
    """pycocotools wants `counts` as bytes; jsonl stores it as an ascii str."""
    c = r["counts"]
    return {"size": [int(r["size"][0]), int(r["size"][1])],
            "counts": c.encode("ascii") if isinstance(c, str) else c}


def iou_matrix(dt, gt):
    from pycocotools import mask as mu
    if not dt or not gt:
        return [[0.0] * len(gt) for _ in dt]
    return mu.iou([_rle(r) for r in dt], [_rle(r) for r in gt], [0] * len(gt)).tolist()


def load(masks_path):
    """masks.jsonl -> {frame: {prompt: [(tid, rle), ...]}} and the set of tids per prompt."""
    per_frame = defaultdict(lambda: defaultdict(list))
    tids = defaultdict(set)
    n = 0
    for line in open(masks_path):
        if not line.strip():
            continue
        r = json.loads(line)
        p = r.get("prompt", "?")
        per_frame[r["f"]][p].append((r["tid"], r["rle"]))
        tids[p].add(r["tid"])
        n += 1
    return per_frame, tids, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", required=True, help="masks.jsonl from adapters/sam3_track.py")
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True, help="output directory for metrics.json / person_identities.json")
    ap.add_argument("--blur-prompt", default="woman")
    ap.add_argument("--other-prompt", default="man")
    ap.add_argument("--control-prompt", default="person")
    ap.add_argument("--extra-prompts", default="",
                    help="comma-separated concepts reported as coverage columns only, never changing a "
                         "verdict (e.g. 'child' — useful to see how many escapes are actually children)")
    ap.add_argument("--cover-iou", type=float, default=0.5,
                    help="mask IoU at which a gender mask is judged to be the same physical person as a control mask")
    ap.add_argument("--conflict-iou", type=float, default=0.5,
                    help="mask IoU at which a woman mask and a man mask are judged to be the same person")
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="share of an identity's frames a concept must cover to own that identity")
    a = ap.parse_args()

    fm = read_json(a.frames_meta)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_frame, tids, n_masks = load(a.masks)
    BL, OT, CT = a.blur_prompt, a.other_prompt, a.control_prompt
    EXTRA = [p for p in (x.strip() for x in a.extra_prompts.split(",")) if p]
    # verdicts come from BL/OT alone; EXTRA concepts are measured and reported, never decisive
    MEASURED = [BL, OT] + EXTRA
    print(f"[report] {n_masks} masks over {len(per_frame)} frames; "
          f"ids: " + ", ".join(f"{p}={len(t)}" for p, t in sorted(tids.items())))
    if CT not in tids:
        print(f"[report] warning: no '{CT}' masks — escape rate cannot be computed, "
              f"rerun the adapter with '{CT}' in --text")

    # ---- per-identity coverage: for each control (person) identity, which concept covers its frames
    seen = defaultdict(int)                      # control tid -> frames it appears in
    cov = defaultdict(lambda: defaultdict(int))  # control tid -> prompt -> frames covered
    spans = defaultdict(lambda: [10**9, -1])     # any tid -> [first frame, last frame]
    obs = defaultdict(int)                       # prompt -> observations
    conflict_frames, blur_frames_total = 0, 0

    for f in sorted(per_frame):
        byp = per_frame[f]
        for p, items in byp.items():
            obs[p] += len(items)
            for tid, _ in items:
                s = spans[tid]
                s[0], s[1] = min(s[0], f), max(s[1], f)

        ctrl = byp.get(CT, [])
        for tid, _ in ctrl:
            seen[tid] += 1
        for p in MEASURED:
            gm = byp.get(p, [])
            if not gm or not ctrl:
                continue
            m = iou_matrix([r for _, r in ctrl], [r for _, r in gm])
            for ci, (ctid, _) in enumerate(ctrl):
                if m[ci] and max(m[ci]) >= a.cover_iou:
                    cov[ctid][p] += 1

        # conflicts: a woman mask and a man mask on the same pixels this frame
        wm, mm = byp.get(BL, []), byp.get(OT, [])
        blur_frames_total += len(wm)
        if wm and mm:
            m = iou_matrix([r for _, r in wm], [r for _, r in mm])
            conflict_frames += sum(1 for row in m if row and max(row) >= a.conflict_iou)

    # ---- verdict per control identity
    verdicts, counts = {}, defaultdict(int)
    for tid, nf in sorted(seen.items()):
        fb, fo = cov[tid].get(BL, 0) / nf, cov[tid].get(OT, 0) / nf
        if fb >= a.min_frac and fo >= a.min_frac:
            v = "conflict"
        elif fb >= a.min_frac:
            v = BL
        elif fo >= a.min_frac:
            v = OT
        elif fb > 0 or fo > 0:
            v = "flicker"      # some gender coverage, but never for a majority of the identity's frames
        else:
            v = "ungendered"   # no gender concept ever fired on this person -> ships unblurred
        counts[v] += 1
        rec = {"frames": nf, "first": spans[tid][0], "last": spans[tid][1]}
        for p in MEASURED:
            rec[f"frac_{p}"] = round(cov[tid].get(p, 0) / nf, 3)
        rec["verdict"] = v
        if v in ("ungendered", "flicker"):
            # name any extra concept that does cover this escape — usually explains it
            expl = [p for p in EXTRA if cov[tid].get(p, 0) / nf >= a.min_frac]
            if expl:
                rec["explained_by"] = expl
                for p in expl:
                    counts[f"escape_is_{p}"] += 1
        verdicts[str(tid)] = rec

    n_ctrl = len(seen)
    escapes = counts["ungendered"] + counts["flicker"]
    spans_len = {p: [spans[t][1] - spans[t][0] + 1 for t in ts if t in spans] for p, ts in tids.items()}
    metrics = {
        "clip": fm.get("video"), "n_frames": fm.get("n_frames"), "n_masks": n_masks,
        "prompts": {p: {"ids": len(t), "obs": obs.get(p, 0),
                        "obs_per_frame": round(obs.get(p, 0) / max(1, fm.get("n_frames", 1)), 2),
                        "median_span_frames": (statistics.median(spans_len[p]) if spans_len.get(p) else 0)}
                    for p, t in sorted(tids.items())},
        "control_prompt": CT, "blur_prompt": BL, "other_prompt": OT,
        "thresholds": {"cover_iou": a.cover_iou, "conflict_iou": a.conflict_iou, "min_frac": a.min_frac},
        "identities": {"control_total": n_ctrl, **{k: counts[k] for k in
                       (BL, OT, "conflict", "flicker", "ungendered")},
                       **{f"escape_is_{p}": counts[f"escape_is_{p}"] for p in EXTRA}},
        "escape_rate_identities": round(escapes / n_ctrl, 3) if n_ctrl else None,
        "conflict_rate_blur_obs": round(conflict_frames / blur_frames_total, 3) if blur_frames_total else None,
        "conflict_obs": conflict_frames, "blur_obs": blur_frames_total,
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "person_identities.json", verdicts)

    print(f"\n  prompt      ids     obs   obs/frame   median span")
    for p, d in metrics["prompts"].items():
        print(f"  {p:<10}{d['ids']:>5}{d['obs']:>8}{d['obs_per_frame']:>12}{d['median_span_frames']:>14}")
    if n_ctrl:
        print(f"\n  '{CT}' identities: {n_ctrl}")
        for k in (BL, OT, "conflict", "flicker", "ungendered"):
            print(f"    {k:<12}{counts[k]:>4}  ({counts[k] / n_ctrl:.0%})")
        for p in EXTRA:
            if counts[f"escape_is_{p}"]:
                print(f"    of which '{p}':{counts[f'escape_is_{p}']:>3}")
        print(f"\n  escape rate (no confident gender, ships unblurred): {escapes}/{n_ctrl} = {escapes / n_ctrl:.0%}")
    if blur_frames_total:
        print(f"  conflict rate ('{BL}' masks also claimed by '{OT}'): "
              f"{conflict_frames}/{blur_frames_total} = {conflict_frames / blur_frames_total:.1%}")
    print(f"\n[report] -> {out_dir}/metrics.json, {out_dir}/person_identities.json")


if __name__ == "__main__":
    main()
