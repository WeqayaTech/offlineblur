#!/usr/bin/env python3
"""Phase 4 — Bayesian temporal pooling: many noisy per-frame observations → one stable profile per track.

Per track id, every observation (frame) carries a weight
    w = quality · g_size · g_occ · g_sharp · (1 - entropy)^k
(quality = the head's own "am I right" estimate; the gates zero-out frames where the person is tiny,
covered by someone else, or motion-blurred; flat cross-attention means the queries found nothing).
Then
  gender  Beta posterior:  alpha += w·p_female, beta += w·(1-p_female), prior Beta(1,1)
          → P(female) = alpha/(alpha+beta), evidence n = alpha+beta-2
  age     Gaussian product of experts: each frame is N(mu_t, sigma_t²) tempered by w
          → precision = Σ w/σ², mean = Σ w·μ/σ² / precision, std = 1/sqrt(precision)
  overwrite rule (blueprint step 3): an observation with w ≥ --sharp-thresh counts --sharp-boost times,
          so a single clear look dominates a long blurry history instead of averaging with it
  lock    once evidence ≥ --lock-n and P(female) ≥ --lock-p on one side, the profile is "locked";
          later low-weight frames cannot flip it (their weight is scaled by --post-lock-damp)
Identity lock across tracker id breaks (optional, off by default): tracks whose quality-weighted ROI
embeddings (track_feats.npz) have cosine ≥ --relink-sim, do not overlap in time and are ≤ --relink-gap-s
apart are merged into one identity before pooling.

    python3 aggregator.py --out out/clip [--relink-sim 0.92]

Writes  out/identities.json   one entry per identity: gender, P(female), age mean/std, evidence, locked, tracks, span
        out/track_to_identity.json
"""
import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import iter_jsonl, read_json, write_json


def gate(x, lo, hi):
    """0 below lo, 1 above hi, linear between."""
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    return float(min(1.0, max(0.0, (x - lo) / (hi - lo))))


class Profile:
    def __init__(self, args):
        self.a = args
        self.alpha, self.beta = 1.0, 1.0
        self.prec, self.prec_mean = 0.0, 0.0
        self.n_obs, self.n_used, self.w_sum = 0, 0, 0.0
        self.locked = None  # None / "female" / "male"
        self.first_f, self.last_f = None, None
        self.best = None

    def weight(self, o):
        a = self.a
        g_size = gate(o["size_px"], a.size_lo, a.size_hi)
        g_occ = 1.0 - gate(o["occ"], a.occ_lo, a.occ_hi)
        g_sharp = gate(o["sharp"], a.sharp_lo, a.sharp_hi)
        w = o["quality"] * g_size * g_occ * g_sharp * max(0.0, 1.0 - o["entropy"]) ** a.entropy_pow
        if w >= a.sharp_thresh:
            w *= a.sharp_boost
        return w

    def add(self, o):
        self.n_obs += 1
        self.first_f = o["f"] if self.first_f is None else min(self.first_f, o["f"])
        self.last_f = o["f"] if self.last_f is None else max(self.last_f, o["f"])
        w = self.weight(o)
        if w <= 1e-4:
            return
        if self.locked is not None:
            side = "female" if o["p_female"] >= 0.5 else "male"
            if side != self.locked and w < self.a.sharp_thresh:
                w *= self.a.post_lock_damp
        self.n_used += 1
        self.w_sum += w
        self.alpha += w * o["p_female"]
        self.beta += w * (1.0 - o["p_female"])
        s2 = max(1.0, o["age_std"]) ** 2
        self.prec += w / s2
        self.prec_mean += w * o["age_mean"] / s2
        if self.best is None or w > self.best[0]:
            self.best = (w, o["f"])
        if self.locked is None and self.evidence >= self.a.lock_n:
            p = self.p_female
            if p >= self.a.lock_p:
                self.locked = "female"
            elif p <= 1.0 - self.a.lock_p:
                self.locked = "male"

    @property
    def p_female(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence(self):
        return self.alpha + self.beta - 2.0

    def gender_ci(self):
        """95 % credible interval of P(female) under the Beta posterior (normal approximation)."""
        a, b = self.alpha, self.beta
        m = a / (a + b)
        v = a * b / ((a + b) ** 2 * (a + b + 1))
        return [round(max(0.0, m - 1.96 * math.sqrt(v)), 3), round(min(1.0, m + 1.96 * math.sqrt(v)), 3)]

    def age(self):
        if self.prec <= 0 or self.w_sum < 0.05:   # no usable evidence → no age (avoids 45±1200 nonsense)
            return None, None
        return self.prec_mean / self.prec, 1.0 / math.sqrt(self.prec)

    def merge(self, other):
        self.alpha += other.alpha - 1.0
        self.beta += other.beta - 1.0
        self.prec += other.prec
        self.prec_mean += other.prec_mean
        self.n_obs += other.n_obs
        self.n_used += other.n_used
        self.w_sum += other.w_sum
        self.first_f = min(x for x in (self.first_f, other.first_f) if x is not None)
        self.last_f = max(x for x in (self.last_f, other.last_f) if x is not None)
        if other.best and (self.best is None or other.best[0] > self.best[0]):
            self.best = other.best
        self.locked = None
        if self.evidence >= self.a.lock_n:
            p = self.p_female
            self.locked = "female" if p >= self.a.lock_p else "male" if p <= 1 - self.a.lock_p else None

    def to_dict(self, fps):
        p = self.p_female
        am, asd = self.age()
        gender = "female" if p >= 0.5 else "male"
        conf = max(p, 1 - p)
        cls = None
        if am is not None:
            cls = "child" if am + 1.0 * asd < 13 else ("Woman" if gender == "female" else "Man")
        return {"gender": gender, "p_female": round(p, 4), "gender_ci95": self.gender_ci(), "gender_conf": round(conf, 4),
                "age_mean": None if am is None else round(am, 1), "age_std": None if asd is None else round(asd, 1),
                "class": cls, "locked": self.locked, "evidence": round(self.evidence, 2), "n_obs": self.n_obs, "n_used": self.n_used,
                "first_f": self.first_f, "last_f": self.last_f, "span_s": None if self.first_f is None else round((self.last_f - self.first_f + 1) / fps, 2),
                "best_frame": None if self.best is None else self.best[1]}


def relink(tracks_span, feats, sim_thr, max_gap, fps):
    """Union-find over tracks: same identity if embeddings agree, time-disjoint, and the gap is short."""
    parent = {t: t for t in tracks_span}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    ids = [t for t in tracks_span if str(t) in feats]
    cands = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sa, sb = tracks_span[a], tracks_span[b]
            if sa[1] < sb[0]:
                gap = (sb[0] - sa[1]) / fps
            elif sb[1] < sa[0]:
                gap = (sa[0] - sb[1]) / fps
            else:
                continue  # overlap in time → two different people
            if gap > max_gap:
                continue
            s = float(np.dot(feats[str(a)], feats[str(b)]))
            if s >= sim_thr:
                cands.append((s, a, b))
    n = 0
    for s, a, b in sorted(cands, reverse=True):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # keep identities time-disjoint after the merge
        members = {t for t in tracks_span if find(t) in (ra, rb)}
        spans = sorted(tracks_span[t] for t in members)
        if any(spans[i][1] >= spans[i + 1][0] for i in range(len(spans) - 1)):
            continue
        parent[rb] = ra
        n += 1
    groups = defaultdict(list)
    for t in tracks_span:
        groups[find(t)].append(t)
    print(f"[aggregate] relink: {len(cands)} candidate pairs, {n} merges, {len(groups)} identities")
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size-lo", type=float, default=32, help="box height (px) below which weight is 0")
    ap.add_argument("--size-hi", type=float, default=96, help="box height (px) above which size gate is 1")
    ap.add_argument("--occ-lo", type=float, default=0.15, help="coverage by another box below which occlusion gate is 1")
    ap.add_argument("--occ-hi", type=float, default=0.6, help="coverage above which weight is 0")
    ap.add_argument("--sharp-lo", type=float, default=0.05)
    ap.add_argument("--sharp-hi", type=float, default=0.4)
    ap.add_argument("--entropy-pow", type=float, default=1.0)
    ap.add_argument("--sharp-thresh", type=float, default=0.8, help="weight at which an observation counts as a clear look")
    ap.add_argument("--sharp-boost", type=float, default=3.0)
    ap.add_argument("--lock-n", type=float, default=8.0, help="evidence (sum of weights) needed to lock gender")
    ap.add_argument("--lock-p", type=float, default=0.9)
    ap.add_argument("--post-lock-damp", type=float, default=0.2)
    ap.add_argument("--relink-sim", type=float, default=0.0, help="cosine threshold for merging broken track ids (0 = off)")
    ap.add_argument("--relink-gap-s", type=float, default=10.0)
    a = ap.parse_args()

    out = Path(a.out)
    fps = read_json(next((out / "frames").glob("*/frames_meta.json")))["fps"]
    per_track = defaultdict(list)
    for o in iter_jsonl(out / "attrs.jsonl"):
        per_track[o["tid"]].append(o)
    span = {}
    for r in iter_jsonl(out / "tracks.jsonl"):
        s = span.get(r["tid"])
        span[r["tid"]] = [r["f"], r["f"]] if s is None else [min(s[0], r["f"]), max(s[1], r["f"])]

    if a.relink_sim > 0 and (out / "track_feats.npz").exists():
        feats = dict(np.load(out / "track_feats.npz"))
        groups = relink(span, feats, a.relink_sim, a.relink_gap_s, fps)
    else:
        groups = {t: [t] for t in span}

    identities, t2i = {}, {}
    for k, (root, members) in enumerate(sorted(groups.items(), key=lambda kv: min(span[t][0] for t in kv[1]))):
        prof = Profile(a)
        for t in sorted(members, key=lambda t: span[t][0]):
            p = Profile(a)
            for o in sorted(per_track.get(t, []), key=lambda o: o["f"]):
                p.add(o)
            if p.first_f is None:
                p.first_f, p.last_f = span[t]
            prof.merge(p) if prof.first_f is not None else prof.__dict__.update(p.__dict__)
            t2i[str(t)] = k
        d = prof.to_dict(fps)
        d["tracks"] = sorted(members)
        d["first_f"], d["last_f"] = min(span[t][0] for t in members), max(span[t][1] for t in members)
        d["span_s"] = round((d["last_f"] - d["first_f"] + 1) / fps, 2)
        identities[str(k)] = d
    write_json(out / "identities.json", identities)
    write_json(out / "track_to_identity.json", t2i)
    n_lock = sum(1 for d in identities.values() if d["locked"])
    n_f = sum(1 for d in identities.values() if d["gender"] == "female")
    print(f"[aggregate] {len(identities)} identities ({len(span)} tracks): {n_f} female, {len(identities) - n_f} male, {n_lock} locked")
    for k, d in list(identities.items())[:40]:
        print(f"  id {k:>3} tracks {d['tracks']}: {d['gender']:6s} P(f)={d['p_female']:.2f} age {d['age_mean']}±{d['age_std']} "
              f"evidence {d['evidence']:.1f}/{d['n_obs']} {'LOCKED' if d['locked'] else ''} span {d['span_s']} s")


if __name__ == "__main__":
    main()
