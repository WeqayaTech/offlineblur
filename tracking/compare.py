#!/usr/bin/env python3
"""Bake-off comparison — GT-free proxy metrics + a side-by-side debug video for two trackers.

These clips have no ground truth, so true IDF1/HOTA/ID-switch counts cannot be computed here (that
needs labels; see the MOT17 leg in the README for real numbers). What we CAN measure without labels:
track continuity (how long ids persist), fragmentation (how many short-lived ids), detection density,
and simultaneous-people coverage. The decisive check remains the side-by-side debug video.

    python3 compare.py --seq <seq> --frames-meta <seq>/frames_meta.json \
        --a name=motip,tracks=out_motip/tracks.jsonl --b name=botsort,tracks=out_botsort/tracks.jsonl \
        --out compare_out
"""
from __future__ import annotations

import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import frame_path, iter_jsonl, read_json, write_json

PALETTE = [(255, 0, 255), (0, 200, 255), (0, 255, 120), (255, 160, 0), (0, 120, 255), (200, 0, 255),
           (255, 255, 0), (120, 255, 0), (0, 255, 255), (255, 80, 80), (160, 200, 255), (255, 180, 220)]


def load(tracks):
    per_frame, per_id = defaultdict(list), defaultdict(list)
    for r in iter_jsonl(tracks):
        per_frame[r["f"]].append(r)
        per_id[r["tid"]].append(r["f"])
    return per_frame, per_id


def metrics(per_frame, per_id, fps, n_frames):
    lens = {t: (max(fs) - min(fs) + 1) for t, fs in per_id.items()}  # span in frames
    spans = np.array(sorted(lens.values())) if lens else np.array([0])
    n_boxes = sum(len(v) for v in per_frame.values())
    active = [len(per_frame.get(f, [])) for f in range(n_frames)]
    short = sum(1 for v in lens.values() if v < 0.5 * fps)      # tracks living < 0.5 s
    late_births = sum(1 for t, fs in per_id.items() if min(fs) > 5 and min(fs) < n_frames - 5)
    # gaps: frames where an id is missing between its first and last appearance (occlusion or drop)
    gappy = 0
    for t, fs in per_id.items():
        fs = set(fs)
        span = range(min(fs), max(fs) + 1)
        if sum(1 for f in span if f not in fs) > 0:
            gappy += 1
    return {
        "n_ids": len(per_id),
        "n_boxes": n_boxes,
        "boxes_per_frame": round(n_boxes / max(1, n_frames), 2),
        "track_span_frames_median": int(np.median(spans)),
        "track_span_s_median": round(float(np.median(spans)) / fps, 2),
        "track_span_s_p90": round(float(np.percentile(spans, 90)) / fps, 2),
        "short_tracks_lt_0p5s": short,
        "short_track_fraction": round(short / max(1, len(per_id)), 3),
        "late_births": late_births,
        "ids_with_gaps": gappy,
        "max_simultaneous": int(max(active) if active else 0),
        "mean_simultaneous": round(float(np.mean(active)), 2),
    }


def parse_spec(s):
    d = dict(kv.split("=", 1) for kv in s.split(","))
    return d["name"], d["tracks"]


def color(tid):
    return PALETTE[tid % len(PALETTE)]


def draw(im, rows, name):
    th = max(1, im.shape[0] // 360)
    for r in rows:
        x1, y1, x2, y2 = [int(v) for v in r["box"]]
        c = color(r["tid"])
        cv2.rectangle(im, (x1, y1), (x2, y2), c, th)
        cv2.putText(im, str(r["tid"]), (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * th, c, th, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(im, name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--a", required=True, help="name=..,tracks=..")
    ap.add_argument("--b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel-w", type=int, default=800)
    ap.add_argument("--video", default=None, help="source video for audio (optional)")
    a = ap.parse_args()

    fm = read_json(a.frames_meta)
    fps, W, H, img_dir, n = fm["fps"], fm["width"], fm["height"], fm["img_dir"], fm["n_frames"]
    (na, ta), (nb, tb) = parse_spec(a.a), parse_spec(a.b)
    pfa, pia = load(ta)
    pfb, pib = load(tb)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    stats = {a.seq: {na: metrics(pfa, pia, fps, n), nb: metrics(pfb, pib, fps, n),
                     "meta": {"fps": fps, "w": W, "h": H, "n_frames": n}}}
    write_json(out / f"{a.seq}_metrics.json", stats)

    # side-by-side video
    ph = int(H * a.panel_w / W)
    cw = a.panel_w * 2 + 8
    ff = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cw}x{ph}",
          "-r", f"{fps:.6f}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
          "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(out / f"{a.seq}_compare.mp4")]
    proc = subprocess.Popen(ff, stdin=subprocess.PIPE)
    for f in range(n):
        im = cv2.imread(str(frame_path(img_dir, f)))
        if im is None:
            break
        la = draw(cv2.resize(im.copy(), (a.panel_w, ph)), [dict(r, box=[v * a.panel_w / W if i % 2 == 0 else v * ph / H for i, v in enumerate(r["box"])]) for r in pfa.get(f, [])], na)
        lb = draw(cv2.resize(im.copy(), (a.panel_w, ph)), [dict(r, box=[v * a.panel_w / W if i % 2 == 0 else v * ph / H for i, v in enumerate(r["box"])]) for r in pfb.get(f, [])], nb)
        canvas = np.zeros((ph, cw, 3), np.uint8)
        canvas[:, :a.panel_w] = la
        canvas[:, a.panel_w + 8:] = lb
        proc.stdin.write(np.ascontiguousarray(canvas).tobytes())
    proc.stdin.close()
    proc.wait()

    # markdown summary
    ma, mb = stats[a.seq][na], stats[a.seq][nb]
    keys = list(ma.keys())
    lines = [f"# Bake-off: {a.seq}", "", f"Clip: {W}x{H} @ {fps:.1f} fps, {n} frames. No ground truth → GT-free proxies.", "",
             f"| metric | {na} | {nb} |", "|---|---|---|"]
    for k in keys:
        lines.append(f"| {k} | {ma[k]} | {mb[k]} |")
    lines += ["", "Lower `short_track_fraction` and `late_births` = less fragmentation. Higher `track_span_s_median` = more",
              "continuous ids. True ID switches need labels (see MOT17 leg). Judge continuity on `*_compare.mp4`."]
    (out / f"{a.seq}_compare.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[compare] video → {out / (a.seq + '_compare.mp4')}")


if __name__ == "__main__":
    main()
