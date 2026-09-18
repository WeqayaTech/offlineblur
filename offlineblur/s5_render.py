#!/usr/bin/env python3
"""OfflineBlur stage 5 — render the deliverables from the decided identities.

For every identity whose class is the target (Woman for a male viewer) and that stage 3 did not
drop: its per-frame instance mask (from stage 1), dilated by --dilate-frac of the person's height,
unioned over ±--temporal neighbouring frames (kills one-frame holes), feathered, and pixelated.
Short interpolated holes from stage 2 are rendered as boxes. Masks are per identity, so a woman's
blur never spills onto the man standing next to her.

    python3 s5_render.py --video clip.mp4 --out out/clip [--viewer male]

Writes  out/render/<stem>_blurred.mp4   pixelated women, feathered edges, original audio
        out/render/<stem>_matte.mp4     white-on-black soft matte (feathered alpha), same fps
        out/render/<stem>_debug.mp4     every tracked person outlined + identity id + class
        out/render/masks_rle.jsonl      one line per (frame, identity): COCO RLE of the mask, class,
                                        whether it was blurred — for compositing in an editor
        out/render/render_summary.json
"""
import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import iter_frames, iter_jsonl, mask_to_rle, read_json, rle_to_mask, write_json

VIEWER_TARGET = {"male": "Woman", "female": "Man"}
COLORS = {"Woman": (255, 0, 255), "Man": (255, 128, 0), "Child": (0, 165, 255), None: (160, 160, 160)}


class FfmpegWriter:
    def __init__(self, path, w, h, fps, audio_from=None, crf=18, preset="medium", gray=False):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray" if gray else "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-"]
        if audio_from:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a?", "-c:a", "aac", "-b:a", "256k", "-shortest"]
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        self.p.stdin.close()
        self.p.wait()


def pixelate(roi, block):
    h, w = roi.shape[:2]
    block = max(2, int(block))
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


class TrackWindow:
    """Sliding window over tracks.jsonl (sequential by frame), decoding masks once per frame."""

    def __init__(self, path, tid_segments, blur_ids, dilate_frac):
        self.it = iter_jsonl(path)
        self.buf = {}
        self.loaded = -1
        self.eof = False
        self.tid_segments, self.blur_ids, self.dilate_frac = tid_segments, blur_ids, dilate_frac

    def identity_of(self, tid, f):
        for first_f, last_f, k in self.tid_segments.get(tid, ()):
            if first_f <= f <= last_f:
                return k
        return None

    def _load_one(self):
        try:
            r = next(self.it)
        except StopIteration:
            self.eof = True
            return
        items = []
        for o in r["objs"]:
            k = self.identity_of(o["tid"], r["f"])
            m = rle_to_mask(o["rle"])
            md = None
            if k in self.blur_ids:
                bh = max(1.0, o["box"][3] - o["box"][1])
                ks = max(3, int(self.dilate_frac * bh)) | 1
                md = cv2.dilate(m.astype(np.uint8), np.ones((ks, ks), np.uint8))
            items.append({"k": k, "tid": o["tid"], "box": o["box"], "mask": m, "dil": md})
        self.loaded = r["f"]
        self.buf[r["f"]] = items

    def get(self, f):
        while not self.eof and self.loaded < f:
            self._load_one()
        return self.buf.get(f, [])

    def drop_before(self, f):
        for k in [k for k in self.buf if k < f]:
            del self.buf[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--viewer", choices=["male", "female"], default="male")
    ap.add_argument("--dilate-frac", type=float, default=0.02, help="mask dilation as a fraction of person height")
    ap.add_argument("--temporal", type=int, default=1, help="union the mask over ±N frames")
    ap.add_argument("--feather", type=int, default=6, help="edge feather in px")
    ap.add_argument("--pixel-frac", type=float, default=0.10, help="pixel block = this × person height")
    ap.add_argument("--pixel-min", type=int, default=8)
    ap.add_argument("--style", choices=["pixelate", "gaussian"], default="pixelate")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--no-uncertain-blur", action="store_true", help="skip every identity stage 4 marked uncertain")
    ap.add_argument("--min-uncertain-s", type=float, default=0.5,
                    help="gender-uncertain identities shorter than this are not blurred (a sub-second blip is far more often a glitch than a person)")
    ap.add_argument("--uncertain-min-share", type=float, default=0.60, help="gender-uncertain identities need at least this winning share to be blurred")
    ap.add_argument("--uncertain-min-weight", type=float, default=2.0, help="... and at least this much total vote weight")
    ap.add_argument("--no-debug", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    rd = out / "render"
    rd.mkdir(exist_ok=True)
    stem = Path(a.video).stem
    meta = read_json(out / "s1_meta.json")
    fps, W, H, n_frames = meta["fps"], meta["width"], meta["height"], meta["n_frames"]
    ident = read_json(out / "identities.json")
    classes = read_json(out / "classes.json")
    target = VIEWER_TARGET[a.viewer]
    tid_segments = {int(k): v for k, v in ident["tid_segments"].items()}
    info = {r["identity"]: r for r in classes["identities"]}
    def gender_uncertain(r):
        return any(x.startswith(("gender_margin", "low_weight", "no_gender")) for x in r.get("reasons", []))

    # an age-only doubt (adult vs child) never unblurs an adult-looking target: the adult-read-as-child escape is the
    # consequential error, over-blurring a borderline child is not. Only gender doubt on a sub-second blip is skipped.
    def blur_ok(r):
        if r["dropped_reason"] is not None or r["cls"] != target:
            return False
        if not (r["uncertain"] and gender_uncertain(r)):
            return True
        if a.no_uncertain_blur or r["seconds"] < a.min_uncertain_s:
            return False
        return r.get("gender_share", 0) >= a.uncertain_min_share and r.get("weight", 0) >= a.uncertain_min_weight

    blur_ids = {k for k, r in info.items() if blur_ok(r)}
    skipped = sorted(k for k, r in info.items() if r["dropped_reason"] is None and r["cls"] == target and k not in blur_ids)
    interp = defaultdict(list)
    for it in ident["identities"]:
        if it["identity"] in blur_ids:
            for f, x1, y1, x2, y2 in it["interp"]:
                interp[f].append((it["identity"], (x1, y1, x2, y2)))
    print(f"[s5] target {target}: blur identities {sorted(blur_ids)} "
          f"({sum(1 for k in blur_ids if info[k]['uncertain'])} uncertain) of {len(info)}; skipped uncertain {skipped}", flush=True)

    win = TrackWindow(out / "tracks.jsonl", tid_segments, blur_ids, a.dilate_frac)
    vw = FfmpegWriter(rd / f"{stem}_blurred.mp4", W, H, fps, audio_from=a.video, crf=a.crf)
    mw = FfmpegWriter(rd / f"{stem}_matte.mp4", W, H, fps, crf=12, preset="fast", gray=True)
    dw = None if a.no_debug else FfmpegWriter(rd / f"{stem}_debug.mp4", W, H, fps, crf=23, preset="fast")
    rf = open(rd / "masks_rle.jsonl", "w")
    T = a.temporal
    kf = (a.feather * 2 + 1,) * 2
    t0 = time.time()
    n_blur_frames = n_regions = 0
    id_frames = defaultdict(int)

    for fidx, frame in iter_frames(a.video, max_frames=n_frames):
        win.get(fidx + T)
        win.drop_before(fidx - T)
        per_id = {}
        for df in range(-T, T + 1):
            for it in win.get(fidx + df):
                if it["k"] in blur_ids:
                    per_id[it["k"]] = per_id[it["k"]] | it["dil"] if it["k"] in per_id else it["dil"].copy()
            for k, (x1, y1, x2, y2) in interp.get(fidx + df, []):
                if k not in per_id:
                    per_id[k] = np.zeros((H, W), dtype=np.uint8)
                per_id[k][max(0, int(y1)):min(H, int(y2) + 1), max(0, int(x1)):min(W, int(x2) + 1)] = 1
        matte = np.zeros((H, W), dtype=np.float32)
        debug = frame.copy() if dw else None
        for k, m in per_id.items():
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            roi = frame[y1:y2, x1:x2]
            if a.style == "pixelate":
                blurred = pixelate(roi, max(a.pixel_min, a.pixel_frac * (y2 - y1)))
            else:
                s = max(3, int(0.05 * (y2 - y1))) | 1
                blurred = cv2.GaussianBlur(roi, (s, s), 0)
            alpha = cv2.GaussianBlur(m[y1:y2, x1:x2].astype(np.float32), kf, a.feather / 2.0) if a.feather > 0 \
                else m[y1:y2, x1:x2].astype(np.float32)
            frame[y1:y2, x1:x2] = (roi * (1 - alpha[:, :, None]) + blurred * alpha[:, :, None]).astype(np.uint8)
            matte[y1:y2, x1:x2] = np.maximum(matte[y1:y2, x1:x2], alpha)
            n_regions += 1
            id_frames[k] += 1
            rf.write(json.dumps({"f": fidx, "identity": k, "cls": info[k]["cls"], "blurred": True,
                                 "rle": mask_to_rle(m.astype(bool))}) + "\n")
        # export the other (not blurred) human identities too, so an editor has every mask
        for it in win.get(fidx):
            k = it["k"]
            if k is None or k in per_id or k not in info or info[k]["dropped_reason"] is not None:
                continue
            rf.write(json.dumps({"f": fidx, "identity": k, "cls": info[k]["cls"], "blurred": False,
                                 "rle": mask_to_rle(it["mask"])}) + "\n")
        if per_id:
            n_blur_frames += 1
        if dw:
            for it in win.get(fidx):
                k = it["k"]
                r = info.get(k, {})
                cls = r.get("cls")
                col = COLORS.get(cls, COLORS[None])
                if r.get("dropped_reason"):
                    col = (90, 90, 90)
                cs, _ = cv2.findContours(it["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug, cs, -1, col, 2)
                x1, y1 = int(it["box"][0]), int(it["box"][1])
                tag = f"id{k if k is not None else '-'} t{it['tid']} {cls or '-'}" + (" ?" if r.get("uncertain") else "") \
                    + (f" [{r['dropped_reason']}]" if r.get("dropped_reason") else "")
                cv2.putText(debug, tag, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            for k, (x1, y1, x2, y2) in interp.get(fidx, []):
                cv2.rectangle(debug, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)
            cv2.putText(debug, f"f={fidx} t={fidx/fps:.2f}s blur={sorted(per_id)}", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            dw.write(debug)
        vw.write(frame)
        mw.write((matte * 255).astype(np.uint8))
        if (fidx + 1) % 500 == 0:
            print(f"  [s5] {fidx+1}/{n_frames} ({(fidx+1)/(time.time()-t0):.1f} fps)", flush=True)
    vw.release()
    mw.release()
    if dw:
        dw.release()
    rf.close()
    summ = {"video": meta["video"], "frames": n_frames, "target": target, "blur_identities": sorted(blur_ids),
            "identities": {k: {"cls": r["cls"], "uncertain": r["uncertain"], "dropped": r["dropped_reason"],
                               "seconds": r["seconds"], "blurred_frames": id_frames.get(k, 0)} for k, r in info.items()},
            "skipped_uncertain": skipped,
            "frames_with_blur": n_blur_frames, "regions": n_regions, "seconds": round(time.time() - t0, 1),
            "settings": vars(a)}
    write_json(rd / "render_summary.json", summ)
    print(f"[s5] {meta['video']}: blurred {sorted(blur_ids)} · {n_blur_frames}/{n_frames} frames with blur · "
          f"{(time.time()-t0)/60:.1f} min → {rd}", flush=True)


if __name__ == "__main__":
    main()
