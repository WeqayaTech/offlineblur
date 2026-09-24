#!/usr/bin/env python3
"""Draw a SAM 3 multi-prompt run: who the "woman" prompt picked, who it left behind, and the blur.

Consumes `masks.jsonl` from `adapters/sam3_track.py` (rows carry a `prompt` field) and writes up to
three videos in one pass over the frames:

  --out        LABELS. Gender-prompt masks filled and colour-coded (woman magenta, man blue); the
               control `person` masks drawn as a thin outline only. A person wearing an outline and no
               fill is an escape — SAM 3 found the human and no gender concept fired on them. That
               contrast is the whole point of the picture, so the control is never filled.
  --blur-out   PRODUCT. Exactly the `woman` masks pixelated per pixel, everything else untouched. This
               is what the prompt-as-selector pipeline would actually ship.
  --sbs-out    Original left, blurred right, for judging the blur without flipping between files.

Nothing here classifies anything: it draws SAM 3's own concept assignments. Verdict counts come from
`sam3_gender_report.py`; pass its person_identities.json as --verdicts to print escapes on screen.

    python3 render_gender.py --frames-meta <seq>/frames_meta.json --masks <out>/masks.jsonl \
        --out labels.mp4 --blur-out blur.mp4 [--sbs-out sbs.mp4] [--blur-prompt woman]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import frame_path, read_json
from render_masks import ffmpeg_writer, pixelate, rle_decode

# BGR. Fills for the gender concepts, a neutral outline for the control prompt.
PROMPT_COLOR = {"woman": (255, 0, 255), "man": (255, 160, 0), "girl": (200, 0, 255),   # BGR
                "boy": (255, 200, 80), "person": (190, 190, 190)}
DEFAULT_COLOR = (0, 255, 120)


def pcolor(prompt):
    if prompt in PROMPT_COLOR:
        return PROMPT_COLOR[prompt]
    h = (sum(ord(c) for c in prompt) % 12) * 21   # stable across runs, unlike hash()
    return tuple(int(v) for v in cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0][0])


def outline(img, mask, color, thickness=2):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cs, -1, color, thickness, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--masks", required=True, help="masks.jsonl from adapters/sam3_track.py (rows need a 'prompt')")
    ap.add_argument("--out", required=True, help="labels video path")
    ap.add_argument("--blur-out", default=None, help="pixelated-product video path")
    ap.add_argument("--sbs-out", default=None, help="original | blurred side-by-side video path")
    ap.add_argument("--blur-prompt", default="woman", help="the concept whose masks get pixelated")
    ap.add_argument("--control-prompt", default="person", help="drawn as an outline, never filled")
    ap.add_argument("--verdicts", default=None, help="person_identities.json from sam3_gender_report.py")
    ap.add_argument("--alpha", type=float, default=0.55)
    ap.add_argument("--min-score", type=float, default=0.0, help="hide masks below this detection score")
    ap.add_argument("--label", default="SAM 3")
    a = ap.parse_args()

    fm = read_json(a.frames_meta)
    fps, W, H, img_dir, n = fm["fps"], fm["width"], fm["height"], fm["img_dir"], fm["n_frames"]

    per_frame = defaultdict(list)
    prompts_seen, n_masks = set(), 0
    for line in open(a.masks):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("score", 1.0) < a.min_score:
            continue
        p = r.get("prompt", "?")
        per_frame[r["f"]].append((p, r["tid"], r["rle"]))
        prompts_seen.add(p)
        n_masks += 1
    print(f"[render-gender] {n_masks} masks, prompts {sorted(prompts_seen)}, across {len(per_frame)} frames")

    escaped = set()
    if a.verdicts and Path(a.verdicts).exists():
        v = read_json(a.verdicts)
        escaped = {int(k) for k, d in v.items() if d["verdict"] in ("ungendered", "flicker")}
        print(f"[render-gender] {len(escaped)} escaped identities marked from {a.verdicts}")

    wl = ffmpeg_writer(a.out, W, H, fps)
    wb = ffmpeg_writer(a.blur_out, W, H, fps) if a.blur_out else None
    ws = ffmpeg_writer(a.sbs_out, W * 2, H, fps) if a.sbs_out else None

    for f in range(n):
        im = cv2.imread(str(frame_path(img_dir, f)))
        if im is None:
            break
        entries = per_frame.get(f, [])
        counts = defaultdict(int)
        labels = im.copy()

        # control first (outline only), so gender fills draw over it
        for prompt, tid, rle in entries:
            counts[prompt] += 1
            if prompt != a.control_prompt:
                continue
            m = rle_decode(rle)
            if not m.any():
                continue
            esc = tid in escaped
            outline(labels, m, (0, 0, 255) if esc else pcolor(prompt), 2 if esc else 1)
            if esc:
                ys, xs = np.where(m)
                cv2.putText(labels, "ESCAPE", (max(0, int(xs.mean()) - 30), max(24, int(ys.min()) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # blur prompt last: its fill must never be hidden by a conflicting concept, so what you see
        # filled in this video is exactly what gets pixelated in the blur video
        gender = [e for e in entries if e[0] != a.control_prompt]
        gender.sort(key=lambda e: e[0] == a.blur_prompt)
        for prompt, tid, rle in gender:
            m = rle_decode(rle)
            if not m.any():
                continue
            c = np.array(pcolor(prompt), dtype=np.float32)
            blend = (im.astype(np.float32) * (1 - a.alpha) + c * a.alpha).astype(np.uint8)
            labels[m] = blend[m]
            ys, xs = np.where(m)
            cv2.putText(labels, f"{prompt}:{tid}", (max(0, int(xs.mean()) - 24), max(12, int(ys.min()) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, pcolor(prompt), 2, cv2.LINE_AA)

        cv2.rectangle(labels, (0, 0), (W, 34), (0, 0, 0), -1)
        head = "  ".join(f"{p}={counts.get(p, 0)}" for p in sorted(prompts_seen))
        cv2.putText(labels, f"{a.label} concept prompts - {head} - frame {f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        wl.stdin.write(np.ascontiguousarray(labels).tobytes())

        if wb is not None or ws is not None:
            blurred = im.copy()
            for prompt, tid, rle in entries:
                if prompt != a.blur_prompt:
                    continue
                m = rle_decode(rle)
                ys, xs = np.where(m)
                if len(xs) == 0:
                    continue
                y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                roi = im[y1:y2, x1:x2]
                pix = pixelate(roi, 0.045 * (y2 - y1))
                mm = m[y1:y2, x1:x2]
                region = blurred[y1:y2, x1:x2]
                region[mm] = pix[mm]
                blurred[y1:y2, x1:x2] = region
            if wb is not None:
                wb.stdin.write(np.ascontiguousarray(blurred).tobytes())
            if ws is not None:
                sbs = np.hstack([im, blurred])
                cv2.rectangle(sbs, (0, 0), (W * 2, 34), (0, 0, 0), -1)
                cv2.putText(sbs, "original", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(sbs, f"blurred: prompt '{a.blur_prompt}' ({counts.get(a.blur_prompt, 0)})",
                            (W + 10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
                ws.stdin.write(np.ascontiguousarray(sbs).tobytes())

        if f % 100 == 0:
            print(f"[render-gender] frame {f}/{n}")

    for w, path in ((wl, a.out), (wb, a.blur_out), (ws, a.sbs_out)):
        if w is not None:
            w.stdin.close(); w.wait()
            print(f"[render-gender] -> {path}")


if __name__ == "__main__":
    main()
