#!/usr/bin/env python3
"""Outputs: debug video (every track box, identity id, locked profile), MOT text with identity ids, summary.

    python3 render.py --out out/clip [--video clip.mp4]
"""
import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import iter_jsonl, read_json, write_json
from frames import frame_path


class FfmpegWriter:
    def __init__(self, path, w, h, fps, audio_from=None, crf=20):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-"]
        if audio_from:
            cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a?", "-c:a", "aac", "-shortest"]
        cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        self.p.stdin.close()
        self.p.wait()


COL = {"female": (255, 0, 255), "male": (255, 140, 0), "child": (0, 200, 255)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default=None, help="source video (for the audio track)")
    a = ap.parse_args()
    out = Path(a.out)
    meta = read_json(next((out / "frames").glob("*/frames_meta.json")))
    fps, W, H, img_dir = meta["fps"], meta["width"], meta["height"], meta["img_dir"]
    ids = read_json(out / "identities.json")
    t2i = read_json(out / "track_to_identity.json")
    per_frame = defaultdict(list)
    for r in iter_jsonl(out / "tracks.jsonl"):
        per_frame[r["f"]].append(r)
    latest = {}
    for o in iter_jsonl(out / "attrs.jsonl"):
        latest[(o["f"], o["tid"])] = o
    stem = meta["seq_name"]
    (out / "render").mkdir(exist_ok=True)
    wr = FfmpegWriter(out / "render" / f"{stem}_debug.mp4", W, H, fps, a.video)
    mot = open(out / "render" / f"{stem}_identities_mot.txt", "w")
    th = max(1, int(round(min(W, H) / 540)))
    fs = 0.45 * th
    for f in range(meta["n_frames"]):
        im = cv2.imread(str(frame_path(img_dir, f)))
        if im is None:
            break
        for r in per_frame.get(f, []):
            k = t2i.get(str(r["tid"]))
            d = ids.get(str(k)) if k is not None else None
            x1, y1, x2, y2 = [int(round(v)) for v in r["box"]]
            if d is None:
                col, label = (160, 160, 160), f"t{r['tid']}"
            else:
                key = "child" if d.get("class") == "child" else d["gender"]
                col = COL[key]
                label = f"#{k} {d['gender'][0].upper()} {d['p_female'] if d['gender'] == 'female' else 1 - d['p_female']:.2f}"
                if d["age_mean"] is not None:
                    label += f" {d['age_mean']:.0f}±{d['age_std']:.0f}y"
                if d["locked"]:
                    label += " L"
                mot.write(f"{f + 1},{k},{x1},{y1},{x2 - x1},{y2 - y1},1,-1,-1,-1\n")
            o = latest.get((f, r["tid"]))
            cv2.rectangle(im, (x1, y1), (x2, y2), col, th)
            if o is not None:
                cv2.putText(im, f"q{o['quality']:.2f}", (x1, min(H - 2, y2 + 12 * th)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, col, th)
            cv2.putText(im, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
        wr.write(im)
    wr.release()
    mot.close()
    summary = {"n_identities": len(ids), "n_female": sum(1 for d in ids.values() if d["gender"] == "female"),
               "n_locked": sum(1 for d in ids.values() if d["locked"]), "n_children": sum(1 for d in ids.values() if d.get("class") == "child"),
               "tracks_meta": read_json(out / "tracks_meta.json") if (out / "tracks_meta.json").exists() else None,
               "attrs_meta": read_json(out / "attrs_meta.json") if (out / "attrs_meta.json").exists() else None}
    write_json(out / "render" / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
