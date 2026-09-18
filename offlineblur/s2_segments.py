#!/usr/bin/env python3
"""OfflineBlur stage 2 (v1.2) — cut raw tracks into clean single-person SEGMENTS and describe each one.

BoT-SORT keeps a person's id through short occlusions but in crossings it sometimes hands one
person's id to another, and it re-detects the same person under a new id. This stage does not
decide identities (stage 4 does, after stage 3 has judged every segment); it produces the pieces
and everything needed to judge and link them:

  1. SPLIT raw tracks where the tracker probably switched person: at an internal gap whose boxes
     do not line up (IoU < --gap-split-iou), and where the appearance of two consecutive sampled
     crops collapses (OSNet cosine < --split-sim; the cut goes at the worst box continuity between them).
  2. DESCRIBE each segment with --k crops spread over its life: an outlined crop image for the
     judge, an OSNet person re-id embedding (MSMT17 weights), a face embedding (InsightFace
     SCRFD + ArcFace on the crop upscaled to --face-crop-min-h) when a face lies inside the mask,
     the mask fill ratio and height (stage 3 discounts slivers), and how much of the mask lies
     inside another tracked person's mask in the same frame (a duplicate detection of the same body
     — stage 4 attaches those to their host instead of judging them alone).

    python3 s2_segments.py --video clip.mp4 --out out/clip

Writes  out/crops/t<tid>_f<frame>.jpg   outlined crops
        out/segments.json               per segment: tid, frames+boxes, runs, crops, embeddings, dup_of
"""
import argparse
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import box_center, box_coverage, box_iou, iter_frames, iter_jsonl, pick_spread, read_json, rle_to_mask, runs, write_json

WEIGHTS_DIR = os.environ.get("OFFLINEBLUR_WEIGHTS", "/workspace/offlineblur/weights")


class FaceModel:
    def __init__(self, det_size: int):
        try:
            import onnxruntime as ort
            ort.preload_dlls()                       # finds torch's CUDA/cuDNN pip libs for the CUDA provider
        except Exception:
            pass
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                                providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size), det_thresh=0.5)
        print(f"[s2] face model providers: {self.app.models['detection'].session.get_providers()}", flush=True)

    def detect(self, bgr):
        return self.app.get(bgr)          # .bbox, .det_score, .normed_embedding


class BodyModel:
    """OSNet person re-identification through boxmot's runtime (weights auto-download to WEIGHTS_DIR)."""

    def __init__(self, weights_name: str, device="cuda"):
        import torch
        from boxmot.reid.core.runtime import ReID
        Path(WEIGHTS_DIR).mkdir(parents=True, exist_ok=True)
        self.r = ReID(weights=Path(WEIGHTS_DIR) / weights_name, device=torch.device(device), half=False)
        self.name = weights_name

    def embed(self, bgr_frame, box):
        f = np.asarray(self.r.model.get_features(np.array([box], dtype=np.float32), bgr_frame))[0].astype(np.float32)
        return f / max(np.linalg.norm(f), 1e-9)


class TracksReader:
    """Sequential access to tracks.jsonl by frame index (frames are asked in increasing order)."""

    def __init__(self, path):
        self.it = iter_jsonl(path)
        self.cur = None

    def get(self, f):
        while self.cur is None or self.cur["f"] < f:
            try:
                self.cur = next(self.it)
            except StopIteration:
                return []
        return self.cur["objs"] if self.cur["f"] == f else []


def unit_mean(vecs):
    if not vecs:
        return None
    m = np.mean(vecs, axis=0)
    return [round(float(x), 5) for x in m / max(np.linalg.norm(m), 1e-9)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8, help="crops per segment (spread over its life)")
    ap.add_argument("--min-track-frames", type=int, default=3, help="shorter segments are ignored as noise")
    ap.add_argument("--pad", type=float, default=0.15, help="crop padding as a fraction of the box")
    ap.add_argument("--gap-split-dist", type=float, default=0.75,
                    help="split a track at an internal gap if the box after the gap is farther than this × height from where the motion predicted it")
    ap.add_argument("--split-sim", type=float, default=0.35, help="split a segment where consecutive crops' OSNet cosine falls below this")
    ap.add_argument("--reid", default="osnet_x1_0_msmt17.pt")
    ap.add_argument("--face-det-size", type=int, default=640)
    ap.add_argument("--face-crop-min-h", type=int, default=512, help="upscale a crop to at least this height before face detection")
    ap.add_argument("--face-min-px", type=int, default=24, help="min face height in ORIGINAL pixels")
    ap.add_argument("--face-min-score", type=float, default=0.6)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out)
    meta = read_json(out / "s1_meta.json")
    fps, W, H = meta["fps"], meta["width"], meta["height"]
    t0 = time.time()

    # ---------------------------------------------------------------- 1a. raw tracks -> gap-split segments
    life = defaultdict(list)                      # tid -> [(f, area, box)]
    for r in iter_jsonl(out / "tracks.jsonl"):
        for o in r["objs"]:
            life[o["tid"]].append((r["f"], o["area"], o["box"]))
    segs = {}
    next_sid = 1
    n_gap_splits = 0
    def jumped(cur, nxt):
        """True if the box after a gap is not where the recent motion says this person should be."""
        prev = cur[-1]
        gap = nxt[0] - prev[0]
        tail = cur[-6:]
        (x0, y0), (x1, y1) = box_center(tail[0][2]), box_center(prev[2])
        dt = max(1, prev[0] - tail[0][0])
        vx, vy = ((x1 - x0) / dt, (y1 - y0) / dt) if len(tail) > 1 else (0.0, 0.0)
        px, py = x1 + vx * gap, y1 + vy * gap
        bx, by = box_center(nxt[2])
        h_prev, h_next = prev[2][3] - prev[2][1], nxt[2][3] - nxt[2][1]
        tol = a.gap_split_dist * h_prev + 0.05 * W * gap / fps
        size_ok = 0.5 <= (h_next / max(h_prev, 1e-6)) <= 2.0
        return float(np.hypot(px - bx, py - by)) > tol or not size_ok

    for tid, lst in life.items():
        lst.sort(key=lambda x: x[0])
        cur = [lst[0]]
        for prev, nxt in zip(lst, lst[1:]):
            # a gap whose far side is off the predicted path, or a single-frame jump of more than the person's own height
            # (the tracker hopped onto someone else: a boy → a woman on the Speakers Corner clip)
            (px, py), (qx, qy) = box_center(prev[2]), box_center(nxt[2])
            hop = np.hypot(px - qx, py - qy) > 1.0 * max(prev[2][3] - prev[2][1], 1.0)
            if (nxt[0] - prev[0] > 1 and jumped(cur, nxt)) or (nxt[0] - prev[0] == 1 and hop):
                segs[next_sid] = {"sid": next_sid, "tid": tid, "frames": cur}
                next_sid += 1
                n_gap_splits += 1
                cur = []
            cur.append(nxt)
        segs[next_sid] = {"sid": next_sid, "tid": tid, "frames": cur}
        next_sid += 1
    n_short = sum(1 for s in segs.values() if len(s["frames"]) < a.min_track_frames)
    segs = {k: s for k, s in segs.items() if len(s["frames"]) >= a.min_track_frames}
    print(f"[s2] {len(life)} raw tracks -> {len(segs)} segments ({n_gap_splits} gap splits, {n_short} too short dropped)", flush=True)

    # ---------------------------------------------------------------- 2. crops + embeddings + duplicate overlap
    picked = {sid: pick_spread(s["frames"], a.k, key=lambda x: x[1]) for sid, s in segs.items()}
    by_f = defaultdict(list)                     # f -> [sid]
    for sid, sel in picked.items():
        for f, _, _ in sel:
            by_f[f].append(sid)
    crops_dir = out / "crops"
    crops_dir.mkdir(exist_ok=True)
    faces, body = FaceModel(a.face_det_size), BodyModel(a.reid, a.device)
    for s in segs.values():
        s.update({"crops": [], "bembs": [], "fembs": []})
    reader = TracksReader(out / "tracks.jsonl")
    n_done = 0
    for f, img in iter_frames(a.video, wanted=set(by_f)):
        objs = {o["tid"]: o for o in reader.get(f)}
        masks = {}

        def mask_of(tid):
            if tid not in masks:
                masks[tid] = rle_to_mask(objs[tid]["rle"])
            return masks[tid]

        for sid in by_f[f]:
            s = segs[sid]
            o = objs.get(s["tid"])
            if o is None:
                continue
            mask = mask_of(s["tid"])
            x1, y1, x2, y2 = o["box"]
            pw, ph = a.pad * (x2 - x1), a.pad * (y2 - y1)
            cx1, cy1 = max(0, int(x1 - pw)), max(0, int(y1 - ph))
            cx2, cy2 = min(W, int(x2 + pw) + 1), min(H, int(y2 + ph) + 1)
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            bemb = body.embed(img, o["box"])
            # face: detect on the crop upscaled so small faces are found; map back to frame coords
            ch = cy2 - cy1
            sc = min(4.0, max(1.0, a.face_crop_min_h / max(ch, 1)))
            det_img = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC) if sc > 1.0 else crop
            best, best_bbox = None, None
            for fc in faces.detect(det_img):
                fx1, fy1, fx2, fy2 = [v / sc for v in fc.bbox]
                cxm, cym = int(cx1 + (fx1 + fx2) / 2), int(cy1 + (fy1 + fy2) / 2)
                if (0 <= cym < H and 0 <= cxm < W and mask[cym, cxm] and (fy2 - fy1) >= a.face_min_px
                        and fc.det_score >= a.face_min_score):
                    if best is None or fc.det_score > best.det_score:
                        best, best_bbox = fc, [int(cx1 + fx1), int(cy1 + fy1), int(cx1 + fx2), int(cy1 + fy2)]
            face_rec = None
            if best is not None:
                s["fembs"].append(np.asarray(best.normed_embedding, dtype=np.float32))
                face_rec = {"bbox": best_bbox, "score": round(float(best.det_score), 3)}
            # duplicate check: how much of this mask lies inside another tracked person's mask
            dup, dup_frac, dup_box = None, 0.0, 0.0
            area = max(1, o["area"])
            for tid2, o2 in objs.items():
                if tid2 == s["tid"] or box_iou(o["box"], o2["box"]) <= 0.0:
                    continue
                frac = float(np.logical_and(mask, mask_of(tid2)).sum()) / area
                bfrac = box_coverage(o["box"], o2["box"])
                if max(frac, bfrac) > max(dup_frac, dup_box):
                    dup, dup_frac, dup_box = tid2, frac, bfrac
            outlined = crop.copy()
            sub = mask[cy1:cy2, cx1:cx2].astype(np.uint8)
            cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(outlined, cs, -1, (0, 255, 0), max(2, int(0.006 * max(outlined.shape[:2]))))
            name = f"t{s['tid']}_f{f:07d}.jpg"
            cv2.imwrite(str(crops_dir / name), outlined, [cv2.IMWRITE_JPEG_QUALITY, 92])
            s["bembs"].append(bemb)
            s["crops"].append({"f": f, "area": o["area"], "box": o["box"], "crop": name, "face": face_rec,
                               "fill": round(o["area"] / max(1.0, (x2 - x1) * (y2 - y1)), 3),
                               "h": round(y2 - y1, 1), "femb_idx": len(s["fembs"]) - 1 if face_rec else None,
                               "inside": None if dup is None or max(dup_frac, dup_box) < 0.2
                               else {"tid": dup, "frac": round(dup_frac, 3), "box_frac": round(dup_box, 3)}})
        n_done += 1
        if n_done % 200 == 0:
            print(f"  [s2] {n_done}/{len(by_f)} frames ({(time.time()-t0)/n_done:.2f} s/frame)", flush=True)
    print(f"[s2] embedded {len(segs)} segments in {time.time()-t0:.0f}s "
          f"({sum(1 for s in segs.values() if s['fembs'])} with a face)", flush=True)

    # ---------------------------------------------------------------- 1b. appearance splits (id switch inside a track)
    n_app_splits = 0
    consec_sims = []
    for sid in sorted(segs):
        s = segs[sid]
        order = np.argsort([c["f"] for c in s["crops"]])
        crops = [s["crops"][i] for i in order]
        bembs = [s["bembs"][i] for i in order]
        cut_frames = []
        for i in range(len(crops) - 1):
            sim = float(np.dot(bembs[i], bembs[i + 1]))
            consec_sims.append(round(sim, 3))
            if sim < a.split_sim:
                fa, fb = crops[i]["f"], crops[i + 1]["f"]
                seq = [x for x in s["frames"] if fa <= x[0] <= fb]
                worst, cut = 2.0, None
                for p, q in zip(seq, seq[1:]):
                    v = box_iou(p[2], q[2])
                    if v < worst:
                        worst, cut = v, q[0]
                if cut is not None:
                    cut_frames.append(cut)
        if not cut_frames:
            continue
        bounds = [-1] + sorted(set(cut_frames)) + [10 ** 12]
        parts = []
        for lo, hi in zip(bounds, bounds[1:]):
            fr = [x for x in s["frames"] if lo <= x[0] < hi]
            if not fr:
                continue
            idx = [i for i, c in enumerate(s["crops"]) if lo <= c["f"] < hi]
            fe_idx = [s["crops"][i]["femb_idx"] for i in idx if s["crops"][i]["femb_idx"] is not None]
            parts.append({"sid": None, "tid": s["tid"], "frames": fr, "crops": [s["crops"][i] for i in idx],
                          "bembs": [s["bembs"][i] for i in idx], "fembs": [s["fembs"][j] for j in fe_idx]})
        if len(parts) <= 1:
            continue
        del segs[sid]
        for p in parts:
            p["sid"] = next_sid
            segs[next_sid] = p
            next_sid += 1
        n_app_splits += len(parts) - 1
    segs = {k: s for k, s in segs.items() if len(s["frames"]) >= a.min_track_frames}

    outseg = {}
    for sid, s in segs.items():
        fs = [x[0] for x in s["frames"]]
        crops = sorted(s["crops"], key=lambda c: c["f"])
        for c in crops:
            c.pop("femb_idx", None)
        # duplicate-of (stage 4 applies the length rules): the other track most crops sit inside, by mask or by box
        ins_m = [c["inside"]["tid"] for c in crops if c["inside"] and c["inside"]["frac"] >= 0.5]
        ins_b = [c["inside"]["tid"] for c in crops if c["inside"] and c["inside"].get("box_frac", 0) >= 0.85]
        dup_of, dup_how = None, None
        for lst, how in ((ins_m, "mask"), (ins_b, "box")):
            if lst:
                t = max(set(lst), key=lst.count)
                if lst.count(t) * 2 > len(crops):
                    dup_of, dup_how = t, how
                    break
        outseg[sid] = {"sid": sid, "tid": s["tid"], "first_f": fs[0], "last_f": fs[-1], "n_frames": len(fs),
                       "seconds": round(len(fs) / fps, 2), "runs": runs(fs),
                       "frames": [[f, *box] for f, _, box in s["frames"]],
                       "crops": crops, "n_face": len(s["fembs"]), "dup_of": dup_of, "dup_how": dup_how,
                       "face_emb": unit_mean(s["fembs"]), "body_emb": unit_mean(s["bembs"])}
    stats = {"raw_tracks": len(life), "gap_splits": n_gap_splits, "appearance_splits": n_app_splits,
             "segments": len(outseg), "with_face": sum(1 for v in outseg.values() if v["n_face"]),
             "duplicates": sum(1 for v in outseg.values() if v["dup_of"] is not None),
             "consecutive_crop_sims_p05_p50": [float(np.percentile(consec_sims, 5)), float(np.percentile(consec_sims, 50))] if consec_sims else None}
    write_json(out / "segments.json", {"fps": fps, "segments": outseg, "reid": body.name, "stats": stats,
                                       "settings": vars(a), "seconds": round(time.time() - t0, 1)})
    print(f"[s2] {stats}", flush=True)


if __name__ == "__main__":
    main()
