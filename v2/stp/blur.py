#!/usr/bin/env python3
"""Phase 5 — render the blurred deliverable from v2 identities.

v2 tracks are boxes. For pixel masks each frame is segmented once with YOLO11x-seg (person class,
retina masks) and every target box is matched to the instance mask with the best IoU; the mask is
clipped to the (slightly enlarged) track box so it cannot bleed onto the neighbour, dilated, feathered
and pixelated. Boxes without a matching instance (heavy occlusion) fall back to a rounded box mask.

Target = identities whose pooled gender is the viewer's opposite (male viewer → women), never children.
    --min-p 0.5   blur an identity when P(female) ≥ min-p (0.5 = every identity judged female, locked or not)
    --locked-only blur only locked identities (strict "no false blur")

    python3 blur.py --out out/clip --video clip.mp4 [--viewer male]

Writes  out/render/<stem>_blurred.mp4  pixelated targets, feathered edges, original audio
        out/render/<stem>_matte.mp4    white-on-black soft matte
        out/render/masks_rle.jsonl     {"f", "identity", "cls", "blurred", "rle"} per (frame, identity)
        out/render/blur_summary.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import box_iou, iter_jsonl, read_json, write_json
from frames import frame_path
from render import FfmpegWriter

VIEWER_TARGET = {"male": "female", "female": "male"}


def mask_to_rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def pixelate(roi, block):
    h, w = roi.shape[:2]
    block = max(2, int(block))
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def rounded_box_mask(shape, box):
    m = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    r = max(2, int(0.15 * (x2 - x1)))
    cv2.rectangle(m, (x1 + r, y1), (x2 - r, y2), 255, -1)
    cv2.rectangle(m, (x1, y1 + r), (x2, y2 - r), 255, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--viewer", default="male", choices=["male", "female"])
    ap.add_argument("--min-p", type=float, default=0.6, help="blur when P(target gender) >= min-p or the identity is locked; 0.5-0.6 is the review band")
    ap.add_argument("--locked-only", action="store_true")
    ap.add_argument("--seg-weights", default="yolo11x-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--dilate-frac", type=float, default=0.03, help="mask dilation as a fraction of box height")
    ap.add_argument("--feather", type=float, default=0.02, help="edge feather as a fraction of box height")
    ap.add_argument("--block-frac", type=float, default=0.04, help="pixel block as a fraction of box height")
    ap.add_argument("--min-iou", type=float, default=0.4)
    ap.add_argument("--device", default=0)
    a = ap.parse_args()

    out = Path(a.out)
    meta = read_json(next((out / "frames").glob("*/frames_meta.json")))
    fps, W, H, img_dir, stem = meta["fps"], meta["width"], meta["height"], meta["img_dir"], meta["seq_name"]
    ids = read_json(out / "identities.json")
    t2i = read_json(out / "track_to_identity.json")
    target = VIEWER_TARGET[a.viewer]
    blur_ids = set()
    for k, d in ids.items():
        p = d["p_female"] if target == "female" else 1 - d["p_female"]
        if d.get("class") in ("child", None) or (a.locked_only and not d["locked"]):
            continue
        if p < a.min_p and not (d["locked"] and d["gender"] == target):
            continue
        blur_ids.add(int(k))
    print(f"[blur] viewer {a.viewer}: blurring {len(blur_ids)} of {len(ids)} identities: {sorted(blur_ids)}")

    per_frame = defaultdict(list)
    for r in iter_jsonl(out / "tracks.jsonl"):
        k = t2i.get(str(r["tid"]))
        if k is not None:
            per_frame[r["f"]].append((int(k), r["box"]))

    from ultralytics import YOLO
    seg = YOLO(a.seg_weights)
    (out / "render").mkdir(exist_ok=True)
    wb = FfmpegWriter(out / "render" / f"{stem}_blurred.mp4", W, H, fps, a.video, crf=18)
    wm = FfmpegWriter(out / "render" / f"{stem}_matte.mp4", W, H, fps, None, crf=20)
    rle_f = open(out / "render" / "masks_rle.jsonl", "w")
    n_seg, n_box, t0 = 0, 0, time.time()
    for f in range(meta["n_frames"]):
        im = cv2.imread(str(frame_path(img_dir, f)))
        if im is None:
            break
        entries = per_frame.get(f, [])
        alpha = np.zeros((H, W), np.float32)
        inst = []
        if any(k in blur_ids for k, _ in entries):
            res = seg.predict(im, imgsz=a.imgsz, conf=0.15, classes=[0], device=a.device, verbose=False, retina_masks=True, half=True)[0]
            if res.masks is not None:
                for mk, bx in zip(res.masks.data.cpu().numpy(), res.boxes.xyxy.cpu().numpy().tolist()):
                    inst.append((bx, mk > 0.5))
        outim = im.copy()
        # largest people first so a small person's finer blocks are composited on top, never hidden under coarse ones
        for k, box in sorted(entries, key=lambda e: -(e[1][3] - e[1][1])):
            d = ids[str(k)]
            if k not in blur_ids:
                rle_f.write(json.dumps({"f": f, "identity": k, "cls": d.get("class"), "blurred": False,
                                        "rle": mask_to_rle(rounded_box_mask((H, W), box))}) + "\n")
                continue
            best, best_iou = None, a.min_iou
            for bx, mk in inst:
                iou = box_iou(box, bx)
                if iou > best_iou:
                    best, best_iou = mk, iou
            bh = box[3] - box[1]
            mx, my = 0.08 * (box[2] - box[0]), 0.05 * bh
            x1, y1 = int(max(0, box[0] - mx)), int(max(0, box[1] - my))
            x2, y2 = int(min(W, box[2] + mx)), int(min(H, box[3] + my))
            m = None
            if best is not None:
                m = np.zeros((H, W), bool)
                m[y1:y2, x1:x2] = best[y1:y2, x1:x2]
                if m.sum() < 0.15 * (box[2] - box[0]) * bh:
                    m = None
                else:
                    n_seg += 1
            if m is None:
                m = rounded_box_mask((H, W), box)
                n_box += 1
            dil = max(1, int(a.dilate_frac * bh))
            m8 = cv2.dilate(m.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)))
            fe = max(1, int(a.feather * bh)) | 1
            soft = cv2.GaussianBlur(m8.astype(np.float32) / 255.0, (fe * 2 + 1, fe * 2 + 1), 0)
            alpha = np.maximum(alpha, soft)
            rle_f.write(json.dumps({"f": f, "identity": k, "cls": d.get("class"), "blurred": True, "rle": mask_to_rle(m8 > 127)}) + "\n")
            # composite this person with a block size relative to *their* height (source pixels come from the original frame)
            pad = fe * 2 + dil
            ry1, ry2, rx1, rx2 = max(0, y1 - pad), min(H, y2 + pad), max(0, x1 - pad), min(W, x2 + pad)
            roi = im[ry1:ry2, rx1:rx2]
            pix = pixelate(roi, a.block_frac * bh)
            al = soft[ry1:ry2, rx1:rx2, None]
            outim[ry1:ry2, rx1:rx2] = (pix * al + outim[ry1:ry2, rx1:rx2] * (1 - al)).astype(np.uint8)
        wb.write(outim)
        wm.write((alpha * 255).astype(np.uint8)[:, :, None].repeat(3, 2))
        if f % 100 == 0:
            print(f"[blur] frame {f}/{meta['n_frames']} {f / max(1e-6, time.time() - t0):.1f} fps")
    wb.release()
    wm.release()
    rle_f.close()
    summary = {"viewer": a.viewer, "blurred_identities": sorted(blur_ids), "n_identities": len(ids), "masks_from_segmentation": n_seg,
               "masks_from_box_fallback": n_box, "seconds": round(time.time() - t0, 1),
               "blurred_video": str(out / "render" / f"{stem}_blurred.mp4")}
    write_json(out / "render" / "blur_summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
