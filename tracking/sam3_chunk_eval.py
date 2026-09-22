#!/usr/bin/env python3
"""Did chunking cost us any identity? Compare a chunked run against a single-session baseline.

Chunking bounds VRAM by tearing the tracker down every N frames and re-linking identities across the
reset (see adapters/sam3_track.py). That is only worth doing if it is close to free in tracking
quality, and "close to free" has to be measured against the un-chunked run on the same clip rather
than asserted from the stitch counts, which only say how many objects matched, not whether they
matched *correctly*.

This aligns the two runs frame by frame using per-pixel mask IoU, then reports what changed in terms
that matter for a blur pipeline:

  SPLIT   one baseline identity is covered by two or more chunked identities. The person kept being
          tracked but their id changed partway through — exactly the flicker chunking might introduce,
          and the thing to check against the chunk boundaries.
  MERGE   one chunked identity covers two or more baseline identities: two different people were
          welded into one id. Worse than a split, because a selector decision then leaks between them.
  ORPHAN  a mask on one side with no counterpart on the other. Detection differences, not id bookkeeping.

Attribution needs care. SAM 3 is deterministic, so the two runs are bit-identical until the first
reset — measured on the trial clip: zero switches before frame 100, seven within four frames of it.
Once a reset perturbs the trajectory the runs drift, so a switch deep inside a later chunk is still
*downstream of* a reset, not independent evidence that the tracker was unstable there. This tool
therefore reports switches before the first boundary separately: that count is a determinism check
and should be zero. Everything after the first boundary is attributed to chunking, and the
distance-to-preceding-boundary histogram shows whether the damage is immediate or accumulated.

    python3 sam3_chunk_eval.py --a <single>/masks.jsonl --b <chunked>/masks.jsonl \
        --frames-meta <seq>/frames_meta.json --b-meta <chunked>/tracks_meta.json --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json, write_json


def _b(r):
    c = r["counts"]
    return {"size": [int(r["size"][0]), int(r["size"][1])],
            "counts": c.encode("ascii") if isinstance(c, str) else c}


def iou_matrix(dt, gt):
    from pycocotools import mask as mu
    if not dt or not gt:
        return [[0.0] * len(gt) for _ in dt]
    return mu.iou([_b(r) for r in dt], [_b(r) for r in gt], [0] * len(gt)).tolist()


def load(path, prompt=None):
    per_frame = defaultdict(list)
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if prompt and r.get("prompt") != prompt:
            continue
        per_frame[r["f"]].append((r["tid"], r["rle"]))
    return per_frame


def boundaries(b_meta):
    """Absolute frames at which a new chunk starts emitting, from the chunked run's own metadata."""
    if not b_meta:
        return []
    chunk, ov, n = b_meta.get("chunk_frames") or 0, b_meta.get("chunk_overlap") or 0, b_meta.get("n_frames") or 0
    if not chunk or chunk >= n:
        return []
    step = chunk - ov
    return [ci * step + ov for ci in range(1, max(1, -(-(n - ov) // step)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline masks.jsonl (single session)")
    ap.add_argument("--b", required=True, help="chunked masks.jsonl")
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--b-meta", default=None, help="chunked run's tracks_meta.json, for boundary attribution")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=None, help="only compare this prompt's masks")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--min-frames", type=int, default=3,
                    help="ignore a pairing supported by fewer frames than this (transient mismatches)")
    ap.add_argument("--boundary-slack", type=int, default=2,
                    help="a switch this many frames from a chunk boundary is attributed to chunking")
    a = ap.parse_args()

    fm = read_json(a.frames_meta)
    A, B = load(a.a, a.prompt), load(a.b, a.prompt)
    bmeta = read_json(a.b_meta) if a.b_meta and Path(a.b_meta).exists() else None
    bounds = boundaries(bmeta)
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    a_ids = {t for v in A.values() for t, _ in v}
    b_ids = {t for v in B.values() for t, _ in v}
    print(f"[chunk-eval] baseline {len(a_ids)} ids / {sum(len(v) for v in A.values())} masks; "
          f"chunked {len(b_ids)} ids / {sum(len(v) for v in B.values())} masks")
    print(f"[chunk-eval] chunk boundaries at frames {bounds or '(none — b is not chunked)'}")

    pair = defaultdict(int)              # (a_tid, b_tid) -> frames matched
    a_seen, b_seen = defaultdict(int), defaultdict(int)
    a_orphan, b_orphan = 0, 0
    timeline = defaultdict(dict)         # a_tid -> {frame: b_tid}

    for f in range(fm["n_frames"]):
        av, bv = A.get(f, []), B.get(f, [])
        for t, _ in av:
            a_seen[t] += 1
        for t, _ in bv:
            b_seen[t] += 1
        if not av or not bv:
            a_orphan += len(av); b_orphan += len(bv)
            continue
        m = iou_matrix([r for _, r in av], [r for _, r in bv])
        cand = sorted(((m[i][j], i, j) for i in range(len(av)) for j in range(len(bv))
                       if m[i][j] >= a.match_iou), reverse=True)
        ua, ub = set(), set()
        for _, i, j in cand:
            if i in ua or j in ub:
                continue
            ua.add(i); ub.add(j)
            pair[(av[i][0], bv[j][0])] += 1
            timeline[av[i][0]][f] = bv[j][0]
        a_orphan += len(av) - len(ua)
        b_orphan += len(bv) - len(ub)

    a2b, b2a = defaultdict(set), defaultdict(set)
    for (at, bt), n in pair.items():
        if n >= a.min_frames:
            a2b[at].add(bt); b2a[bt].add(at)

    splits = {at: sorted(bs) for at, bs in a2b.items() if len(bs) > 1}
    merges = {bt: sorted(a_) for bt, a_ in b2a.items() if len(a_) > 1}

    # when does each split switch, and how far after a reset?
    switch_frames, all_sw = {}, []
    for at in splits:
        fr = sorted(timeline[at])
        sw = [f for prev, f in zip(fr, fr[1:]) if timeline[at][prev] != timeline[at][f]]
        switch_frames[str(at)] = sw
        all_sw += sw
    first_b = bounds[0] if bounds else None
    before_first = [f for f in all_sw if first_b is not None and f < first_b]
    immediate = sum(1 for f in all_sw if any(0 <= f - b <= a.boundary_slack for b in bounds))
    lag = {}
    for f in all_sw:
        prior = [b for b in bounds if b <= f]
        if prior:
            lag[f - max(prior)] = lag.get(f - max(prior), 0) + 1

    res = {
        "clip": fm.get("video"), "n_frames": fm.get("n_frames"), "prompt": a.prompt,
        "match_iou": a.match_iou, "min_frames": a.min_frames, "boundary_slack": a.boundary_slack,
        "chunk_boundaries": bounds,
        "baseline": {"ids": len(a_ids), "masks": sum(len(v) for v in A.values()), "unmatched_masks": a_orphan},
        "chunked": {"ids": len(b_ids), "masks": sum(len(v) for v in B.values()), "unmatched_masks": b_orphan},
        "splits": len(splits), "merges": len(merges),
        "switches_total": len(all_sw),
        "switches_before_first_boundary": len(before_first),
        "switches_within_slack_of_a_boundary": immediate,
        "switches_by_frames_after_preceding_boundary": dict(sorted(lag.items())),
        "split_detail": {str(k): v for k, v in splits.items()},
        "merge_detail": {str(k): v for k, v in merges.items()},
        "split_switch_frames": switch_frames,
    }
    write_json(out_dir / "chunk_eval.json", res)

    ab = sum(len(v) for v in A.values())
    print(f"\n  baseline ids      {len(a_ids)}")
    print(f"  chunked  ids      {len(b_ids)}")
    print(f"  splits            {len(splits)}   (one baseline person, >1 chunked id)")
    print(f"  id switches       {len(all_sw)}")
    print(f"     before 1st reset {len(before_first)}   <- must be 0; nonzero means SAM 3 is not deterministic")
    print(f"     within {a.boundary_slack} of a reset {immediate}   <- immediate damage")
    print(f"     later            {len(all_sw) - len(before_first) - immediate}   <- drift downstream of a reset")
    print(f"  merges            {len(merges)}   (two baseline people welded into one chunked id)")
    print(f"  unmatched masks   baseline {a_orphan}/{ab} ({a_orphan / max(1, ab):.1%}), "
          f"chunked {b_orphan}/{sum(len(v) for v in B.values())}")
    print(f"\n[chunk-eval] -> {out_dir}/chunk_eval.json")


if __name__ == "__main__":
    main()
