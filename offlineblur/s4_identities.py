#!/usr/bin/env python3
"""OfflineBlur stage 4 (v1.2) — decide each segment's class from its judged crops, then link segments
into one identity per person, then decide each identity from ALL its crops.

Order matters: classes are decided before linking so a link can never merge a woman into a man
(the v1.1 failure), and the identity vote pools every crop afterwards so a person seen mostly from
behind still gets the benefit of the frames where the face was visible.

  1. VOTE-SPLIT: a segment whose judged crops flip cleanly from one gender to the other (both
     halves >= 2 votes, >= 75 % pure) is a tracker id switch the appearance check missed; it is cut
     at the worst box continuity between the two halves.
  2. SEGMENT CLASS: weighted vote (weights from stage 3: face x3, sliver 0). not_person /
     depiction majorities drop the segment (never blurred).
  3. LINK (union-find, best pair first; segments sharing a frame can never be one person):
       duplicate  a segment whose pixels lie inside another tracked person's mask joins that host
       face       ArcFace cosine >= --face-link, any gap, class ignored (faces do not lie;
                  measured: different people never exceed 0.16 on the trial clip)
       body       OSNet cosine >= --body-link (0.85: different people never reached it) AND gap
                  <= --body-max-gap-s AND the box continues the earlier motion AND classes agree
       handoff    a short segment that starts exactly where a longer identity ends (or ends where
                  one starts) on the same spot, with a compatible or undecided class
  4. IDENTITY CLASS: the weighted vote over every crop of every segment; Child only if child
     answers dominate AND the median estimated age <= 12; uncertain if the winning share is below
     --gender-margin or the total weight below --min-weight.
  5. FILL short holes (<= --fill-gap-s) by linear box interpolation.

    python3 s4_identities.py --out out/clip

Writes  out/identities.json   identities (segments, tid_segments map for the renderer, interp boxes), links
        out/classes.json      per identity: cls, uncertain, reasons, votes (what the renderer and gallery read)
"""
import argparse
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import box_center, box_coverage, box_iou, read_json, runs, runs_overlap_frames, write_json

CHILD_AGE_MAX = 12


def gender_seq(rows):
    return [(r["f"], r["v"]["gender"]) for r in rows if r["v"] and r["w"] > 0 and r["v"]["gender"] in ("man", "woman")]


def vote_split_point(rows):
    """Return (last frame of the left half, first frame of the right half) where the judged gender flips
    cleanly (>= 3 votes and >= 75 % purity on each side); else None."""
    seq = gender_seq(rows)
    if len(seq) < 6:
        return None
    best = None
    for i in range(3, len(seq) - 2):
        left, right = Counter(g for _, g in seq[:i]), Counter(g for _, g in seq[i:])
        (gl, cl), (gr, cr) = left.most_common(1)[0], right.most_common(1)[0]
        if gl != gr and cl / i >= 0.75 and cr / (len(seq) - i) >= 0.75:
            score = cl / i + cr / (len(seq) - i)
            if best is None or score > best[0]:
                best = (score, seq[i - 1][0], seq[i][0])
    return None if best is None else (best[1], best[2])


def decide(rows, a):
    """Weighted decision over judged crops. Returns a dict with cls / uncertain / dropped_reason / votes."""
    valid = [r for r in rows if r["v"]]
    rec = {"cls": None, "uncertain": False, "dropped_reason": None, "reasons": [], "n_crops_judged": len(rows),
           "n_valid": len(valid), "n_weak": sum(1 for r in valid if r["w"] == 0), "weight": 0.0}
    if not valid:
        rec.update({"dropped_reason": "no_valid_answers", "uncertain": True})
        return rec
    n = len(valid)
    vc = Counter(r["v"]["verdict"] for r in valid)
    rec["verdict_votes"] = dict(vc)
    if vc["not_person"] / n >= 0.5:
        rec["dropped_reason"] = "not_person"
        return rec
    if vc["depiction"] / n >= 0.5 and not a.blur_depictions:
        rec["dropped_reason"] = "depiction"
        return rec
    if vc["real_person"] / n < 0.75:
        rec["reasons"].append(f"person_share {vc['real_person']/n:.2f}")
    human = [r for r in valid if r["v"]["verdict"] != "not_person" and r["w"] > 0]
    gw = defaultdict(float)
    gc = Counter()
    for r in human:
        g = r["v"]["gender"]
        if g in ("man", "woman"):
            gw[g] += r["w"]
            gc[g] += 1
    rec["gender_votes"] = dict(gc)
    rec["gender_weight"] = {k: round(v, 2) for k, v in gw.items()}
    total = sum(gw.values())
    rec["weight"] = round(total, 2)
    if total <= 0:
        rec.update({"cls": None, "uncertain": True})
        rec["reasons"].append("no_gender_answers")
    else:
        # the winning share is the mean of the weighted share and the plain count share: a single face-weighted answer
        # must not outvote three plain ones (a man's head from behind was blurred that way on a 720p clip)
        g = max(gw, key=lambda k: gw[k] / total + gc[k] / sum(gc.values()))
        share = 0.5 * (gw[g] / total + gc[g] / sum(gc.values()))
        rec["cls"] = "Woman" if g == "woman" else "Man"
        rec["gender_share"] = round(share, 3)
        if share < a.gender_margin:
            rec["uncertain"] = True
            rec["reasons"].append(f"gender_margin {share:.2f}")
        if total < a.min_weight:
            rec["uncertain"] = True
            rec["reasons"].append(f"low_weight {total:.1f}")
    aw = defaultdict(float)
    for r in human:
        if r["v"]["age_group"] in ("child", "adult"):
            aw[r["v"]["age_group"]] += r["w"]
    ages = [r["v"]["estimated_age"] for r in human if r["v"]["estimated_age"] is not None]
    med = statistics.median(ages) if ages else None
    rec["age_weight"] = {k: round(v, 2) for k, v in aw.items()}
    rec["median_age"] = med
    if sum(aw.values()) > 0:
        child_share = aw["child"] / sum(aw.values())
        if child_share >= 0.5 and med is not None and med <= CHILD_AGE_MAX:
            rec["cls"] = "Child"
        elif child_share >= 0.3 or (med is not None and med <= CHILD_AGE_MAX + 3):
            rec["uncertain"] = True
            rec["reasons"].append(f"age_split child={child_share:.2f} median={med}")
    return rec


def compatible(ca, cb):
    return ca is None or cb is None or ca == cb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--face-link", type=float, default=0.45)
    ap.add_argument("--face-link-strong", type=float, default=0.60, help="face similarity needed to join two groups whose decided classes differ")
    ap.add_argument("--face-min-n", type=int, default=2, help="both segments need at least this many face crops for a face link")
    ap.add_argument("--body-link", type=float, default=0.85)
    ap.add_argument("--body-max-gap-s", type=float, default=3.0)
    ap.add_argument("--handoff-gap-s", type=float, default=0.3)
    ap.add_argument("--handoff-overlap", type=int, default=3)
    ap.add_argument("--handoff-max-s", type=float, default=2.0)
    ap.add_argument("--fill-gap-s", type=float, default=0.5)
    ap.add_argument("--gender-margin", type=float, default=0.75)
    ap.add_argument("--min-weight", type=float, default=2.0, help="total vote weight below this marks the identity uncertain")
    ap.add_argument("--blur-depictions", action="store_true")
    ap.add_argument("--vote-split-iou", type=float, default=0.6, help="a vote flip only splits a segment if the boxes are discontinuous (IoU below this) between the halves")
    a = ap.parse_args()

    out = Path(a.out)
    t0 = time.time()
    meta = read_json(out / "s1_meta.json")
    fps, W, H = meta["fps"], meta["width"], meta["height"]
    S = read_json(out / "segments.json")
    segs = {int(k): v for k, v in S["segments"].items()}
    votes = {int(k): v for k, v in read_json(out / "segment_votes.json")["votes"].items()}
    next_sid = max(segs) + 1

    # ---------------------------------------------------------------- 1. vote splits
    n_vote_splits = 0
    for sid in sorted(segs):
        rows = sorted(votes.get(sid, []), key=lambda r: r["f"])
        sp = vote_split_point(rows)
        if sp is None:
            continue
        fa, fb = sp
        seq = [x for x in segs[sid]["frames"] if fa <= x[0] <= fb]
        worst, cut = 2.0, None
        for p, q in zip(seq, seq[1:]):
            v = box_iou(p[1:], q[1:])
            if v < worst:
                worst, cut = v, q[0]
        if cut is None or worst >= a.vote_split_iou:
            continue                                   # votes flipped but the box never jumped: judge noise, not a switch
        s = segs.pop(sid)
        parts = [[x for x in s["frames"] if x[0] < cut], [x for x in s["frames"] if x[0] >= cut]]
        for fr in parts:
            if not fr:
                continue
            fs = [x[0] for x in fr]
            lo, hi = fs[0], fs[-1]
            segs[next_sid] = {**s, "sid": next_sid, "frames": fr, "first_f": lo, "last_f": hi, "n_frames": len(fs),
                              "seconds": round(len(fs) / fps, 2), "runs": runs(fs), "split_from": sid,
                              "crops": [c for c in s["crops"] if lo <= c["f"] <= hi],
                              "face_emb": None, "body_emb": None}          # the parent's embeddings describe two people
            votes[next_sid] = [r for r in rows if lo <= r["f"] <= hi]
            next_sid += 1
        votes.pop(sid, None)
        n_vote_splits += 1

    # ---------------------------------------------------------------- 2. segment classes
    seg_cls = {sid: decide(votes.get(sid, []), a) for sid in segs}
    for sid, s in segs.items():
        s["cls"] = seg_cls[sid]["cls"] if seg_cls[sid]["dropped_reason"] is None else None
        s["decided"] = seg_cls[sid]["cls"] is not None and not seg_cls[sid]["uncertain"]
        s["box_first"], s["box_last"] = s["frames"][0][1:], s["frames"][-1][1:]

    # ---------------------------------------------------------------- 3. linking
    ids = sorted(segs)
    parent = {i: i for i in ids}
    members = {i: [i] for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def can_union(ri, rj, max_overlap=0):
        return all(runs_overlap_frames(segs[mi]["runs"], segs[mj]["runs"]) <= max_overlap
                   for mi in members[ri] for mj in members[rj])

    def union(i, j, how, **extra):
        ri, rj = find(i), find(j)
        if ri == rj:
            return False
        parent[rj] = ri
        members[ri] += members.pop(rj)
        links.append({"a": i, "b": j, "how": how, **extra})
        return True

    links = []
    # duplicates -> host (overlap allowed: same pixels, same time)
    by_tid = defaultdict(list)
    for sid, s in segs.items():
        by_tid[s["tid"]].append(sid)
    for sid, s in segs.items():
        if s.get("dup_of") is None:
            continue
        hosts = [h for h in by_tid.get(s["dup_of"], []) if runs_overlap_frames(s["runs"], segs[h]["runs"]) > 0]
        if not hosts:
            continue
        host = max(hosts, key=lambda h: segs[h]["n_frames"])
        if s.get("dup_how") != "mask":
            continue                                   # box containment is NOT evidence: background people sit inside a near person's box
        union(host, sid, "duplicate", how_detected=s.get("dup_how"))

    def velocity(s, n=5):
        fr = s["frames"][-n:]
        if len(fr) < 2:
            return 0.0, 0.0
        (x0, y0), (x1, y1) = box_center(fr[0][1:]), box_center(fr[-1][1:])
        dt = max(1, fr[-1][0] - fr[0][0])
        return (x1 - x0) / dt, (y1 - y0) / dt

    pairs, near = [], []
    for n_i, i in enumerate(ids):
        A = segs[i]
        for j in ids[n_i + 1:]:
            B = segs[j]
            if runs_overlap_frames(A["runs"], B["runs"]) > 0:
                continue
            first, second = (A, B) if A["first_f"] <= B["first_f"] else (B, A)
            gap_f = max(0, second["first_f"] - first["last_f"])
            gap_s = gap_f / fps
            fs_ = float(np.dot(A["face_emb"], B["face_emb"])) if A["face_emb"] and B["face_emb"] else None
            bs_ = float(np.dot(A["body_emb"], B["body_emb"])) if A["body_emb"] and B["body_emb"] else None
            if fs_ is not None and fs_ >= a.face_link and min(A["n_face"], B["n_face"]) >= a.face_min_n:
                pairs.append((fs_ + 1.0, i, j, "face", fs_, bs_, gap_s))
                continue
            if bs_ is None or bs_ < a.body_link - 0.1 or gap_s > a.body_max_gap_s:
                continue
            vx, vy = velocity(first)
            px, py = box_center(first["box_last"])
            px, py = px + vx * gap_f, py + vy * gap_f
            bx, by = box_center(second["box_first"])
            h_last = first["box_last"][3] - first["box_last"][1]
            move_ok = float(np.hypot(px - bx, py - by)) <= 1.5 * h_last + 0.10 * W * gap_s
            ok_cls = compatible(A["cls"], B["cls"])
            if bs_ >= a.body_link and move_ok and ok_cls:
                pairs.append((bs_, i, j, "body", fs_, bs_, gap_s))
            else:
                near.append({"a": i, "b": j, "body_sim": round(bs_, 3), "gap_s": round(gap_s, 2), "move_ok": bool(move_ok),
                             "cls": [A["cls"], B["cls"]]})
    pairs.sort(reverse=True)
    for score, i, j, how, fs_, bs_, gap_s in pairs:
        ri, rj = find(i), find(j)
        if ri == rj or not can_union(ri, rj):
            continue
        # two groups with DECIDED different classes: a body link never joins them; a face link only with a strong,
        # well-supported match (tiny blurry faces produced a 0.52 bridge between a man and a woman on a 360p clip)
        ca = {segs[m]["cls"] for m in members[ri] if segs[m]["decided"]}
        cb = {segs[m]["cls"] for m in members[rj] if segs[m]["decided"]}
        if ca and cb and ca != cb:
            if how == "body" or fs_ < a.face_link_strong or min(segs[i]["n_face"], segs[j]["n_face"]) < 3:
                continue
        # even an UNCERTAIN lean counts: a "Man ?" segment must not join a woman on clothing similarity, and only a strong,
        # well-supported face match may override it
        la = {segs[m]["cls"] for m in members[ri] if segs[m]["cls"]}
        lb = {segs[m]["cls"] for m in members[rj] if segs[m]["cls"]}
        if la and lb and la != lb:
            if how == "body" or fs_ < a.face_link_strong or min(segs[i]["n_face"], segs[j]["n_face"]) < 3:
                continue
        union(i, j, how, face_sim=None if fs_ is None else round(fs_, 3),
              body_sim=None if bs_ is None else round(bs_, 3), gap_s=round(gap_s, 2))

    # handoff
    def group_info(root):
        ms = members[root]
        first = min(ms, key=lambda m: segs[m]["first_f"])
        last = max(ms, key=lambda m: segs[m]["last_f"])
        decided = {segs[m]["cls"] for m in ms if segs[m]["decided"]}
        return {"first_f": segs[first]["first_f"], "last_f": segs[last]["last_f"],
                "box_first": segs[first]["box_first"], "box_last": segs[last]["box_last"],
                "n_frames": sum(segs[m]["n_frames"] for m in ms), "decided": decided,
                "weight": sum(seg_cls[m]["weight"] for m in ms)}

    hand_gap = int(round(a.handoff_gap_s * fps))
    changed = True
    while changed:
        changed = False
        roots = sorted(set(find(i) for i in ids))
        info = {r: group_info(r) for r in roots}
        cands = []
        for g in roots:
            G = info[g]
            if G["n_frames"] / fps > a.handoff_max_s:
                continue
            for p in roots:
                if p == g:
                    continue
                P = info[p]
                if P["n_frames"] <= G["n_frames"]:
                    continue
                if P["last_f"] - a.handoff_overlap <= G["first_f"] <= P["last_f"] + hand_gap:
                    bp, bg = P["box_last"], G["box_first"]
                elif P["first_f"] - hand_gap <= G["last_f"] <= P["first_f"] + a.handoff_overlap:
                    bp, bg = P["box_first"], G["box_last"]
                else:
                    continue
                if any(segs[m]["cls"] for m in members[g]):
                    continue                            # only a segment with NO gender lean of its own is handed off
                hp, hg = bp[3] - bp[1], bg[3] - bg[1]
                if not (0.4 <= hg / max(hp, 1e-6) <= 2.5):
                    continue                            # a tiny box inside a big one is not the same person continuing
                score = box_iou(bp, bg)
                if score >= 0.3:
                    cands.append((score, g, p))
        cands.sort(reverse=True)
        for score, g, p in cands:
            if find(g) != g or find(p) != p or not can_union(p, g, a.handoff_overlap):
                continue
            union(p, g, "handoff", score=round(score, 3))
            changed = True
            break

    # ---------------------------------------------------------------- 4+5. identities, class, interpolation
    max_gap = int(a.fill_gap_s * fps)
    identities, classes, tid_segments = [], [], defaultdict(list)
    groups = sorted(members.values(), key=lambda ms: min(segs[m]["first_f"] for m in ms))
    for k, ms in enumerate(groups, 1):
        ms = sorted(ms, key=lambda m: segs[m]["first_f"])
        boxes_by_f = {}
        for sid in ms:
            for f, *box in segs[sid]["frames"]:
                if f not in boxes_by_f or (box[3] - box[1]) > (boxes_by_f[f][3] - boxes_by_f[f][1]):
                    boxes_by_f[f] = box
            tid_segments[segs[sid]["tid"]].append([segs[sid]["first_f"], segs[sid]["last_f"], k])
        fs = sorted(boxes_by_f)
        interp = []
        for p, q in zip(fs, fs[1:]):
            gap = q - p - 1
            if 0 < gap <= max_gap:
                b0, b1 = boxes_by_f[p], boxes_by_f[q]
                for g in range(1, gap + 1):
                    w = g / (gap + 1)
                    interp.append([p + g] + [round(b0[c] * (1 - w) + b1[c] * w, 1) for c in range(4)])
        hows = {l["how"] for l in links if l["a"] in ms or l["b"] in ms}
        base = {"identity": k, "segments": ms, "tracks": sorted({segs[m]["tid"] for m in ms}),
                "first_f": fs[0], "last_f": fs[-1], "n_frames": len(fs), "seconds": round(len(fs) / fps, 2),
                "n_face_crops": sum(segs[m]["n_face"] for m in ms), "handoff": "handoff" in hows,
                "links": sorted(hows)}
        identities.append({**base, "n_interp": len(interp), "interp": interp})
        rows = [r for m in ms for r in votes.get(m, [])]
        classes.append({**base, **decide(rows, a),
                        "segment_classes": {m: (seg_cls[m]["cls"], seg_cls[m]["uncertain"]) for m in ms}})
    for tid in tid_segments:
        tid_segments[tid].sort()
    kept = [r for r in classes if r["dropped_reason"] is None]
    summ = {"identities": len(classes), "kept": len(kept),
            "dropped": dict(Counter(r["dropped_reason"] for r in classes if r["dropped_reason"])),
            "by_class": dict(Counter(r["cls"] for r in kept)), "uncertain": sum(1 for r in kept if r["uncertain"]),
            "segments": len(segs), "vote_splits": n_vote_splits,
            "links": dict(Counter(l["how"] for l in links))}
    write_json(out / "identities.json", {"fps": fps, "identities": identities, "links": links,
                                         "near_misses": sorted(near, key=lambda d: -d["body_sim"])[:40],
                                         "tid_segments": tid_segments, "settings": vars(a), "seconds": round(time.time() - t0, 1)})
    write_json(out / "classes.json", {"identities": classes, "summary": summ, "settings": vars(a)})
    # final segments (after vote splits) with their crops, for the gallery / contact sheets
    write_json(out / "segments_final.json", {"segments": {sid: {"sid": sid, "tid": s["tid"], "first_f": s["first_f"], "last_f": s["last_f"],
                                                                 "n_frames": s["n_frames"], "split_from": s.get("split_from"),
                                                                 "cls": seg_cls[sid]["cls"], "uncertain": seg_cls[sid]["uncertain"],
                                                                 "crops": s["crops"]} for sid, s in segs.items()}})
    print(f"[s4] {summ}", flush=True)


if __name__ == "__main__":
    main()
