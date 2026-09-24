#!/usr/bin/env python3
"""RF-DETR-Seg person detector: per-frame person boxes + instance masks, no tracking, no gender.

Stage 1 of the RF-DETR -> McByte -> CLIP pipeline (`run_rfdetr_mcbyte_clip.sh`). The detector answers
only "where is every person, pixel-exactly"; identities come from McByte (sam31_mcbyte.py) and gender
from a per-track CLIP vote (clip_track_gender.py). COCO-pretrained, so no fine-tuning is needed for
`person` (RF-DETR returns COCO category ids: person = 1, while `class_names` is 0-indexed).

Speed on an L4, one 1280x720 frame, fp16 `optimize_for_inference` (fp32 in brackets):
  Nano 312px 18 ms (13) · Small 384px 9 (14) · Medium 432px 9 (17) · Large 504px 10 (21) ·
  XLarge 624px 14 (37) · 2XLarge 768px 28 ms (67), peak 0.4-1.5 GB.

Outputs (same schema as sam31_detect.py, prompt = "person", so sam31_mcbyte.py consumes it unchanged):
  masks.jsonl    {f, tid=f*1000+k, det, prompt, score, box[x1,y1,x2,y2], rle}
  profile.jsonl  {f, det_s, n_det, gpu_alloc, gpu_peak}
  detect_meta.json

    python3 rfdetr_seg_detect.py --frames-meta <seq>/frames_meta.json --out <dir> --model 2XLarge
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import frame_path, read_json, write_json

GB = 1024 ** 3
PERSON = 1   # COCO category id


def mask_to_rle(mask):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def run(frames_meta, out_dir, model_size="2XLarge", threshold=0.15, batch=1, fp16=True, max_frames=None,
        save_masks=True):
    import psutil
    import rfdetr
    import torch
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir, n_frames = frames_meta["img_dir"], int(frames_meta["n_frames"])
    if max_frames:
        n_frames = min(n_frames, max_frames)
    proc = psutil.Process()

    t = time.perf_counter()
    model = getattr(rfdetr, f"RFDETRSeg{model_size}")()
    if fp16:
        model.optimize_for_inference(dtype=torch.float16)
    warm = Image.open(frame_path(img_dir, 0)).convert("RGB")
    for _ in range(3):
        model.predict([warm] * batch if batch > 1 else warm, threshold=threshold)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t
    torch.cuda.reset_peak_memory_stats()

    out = open(out_dir / "masks.jsonl", "w") if save_masks else None
    prof = open(out_dir / "profile.jsonl", "w")
    read_s = det_s = io_s = 0.0
    n_det = 0
    host_peak = 0.0
    t_all = time.perf_counter()
    for f0 in range(0, n_frames, batch):
        fs = list(range(f0, min(n_frames, f0 + batch)))
        t0 = time.perf_counter()
        ims = [Image.open(frame_path(img_dir, f)).convert("RGB") for f in fs]
        t1 = time.perf_counter()
        res = model.predict(ims if len(ims) > 1 else ims[0], threshold=threshold)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        res = res if isinstance(res, list) else [res]
        for f, d in zip(fs, res):
            keep = d.class_id == PERSON
            d = d[keep]
            if out is not None:
                for k in range(len(d)):
                    out.write(json.dumps({
                        "f": f, "tid": f * 1000 + k, "det": k, "prompt": "person",
                        "score": round(float(d.confidence[k]), 3),
                        "box": [round(float(v), 1) for v in d.xyxy[k]],
                        "rle": mask_to_rle(d.mask[k])}) + "\n")
            n_det += len(d)
            prof.write(json.dumps({"f": f, "det_s": round((t2 - t1) / len(fs), 4), "n_det": int(len(d)),
                                   "gpu_alloc": round(torch.cuda.memory_allocated() / GB, 3),
                                   "gpu_peak": round(torch.cuda.max_memory_allocated() / GB, 3)}) + "\n")
        t3 = time.perf_counter()
        read_s += t1 - t0
        det_s += t2 - t1
        io_s += t3 - t2
        host_peak = max(host_peak, proc.memory_info().rss / GB)
        if f0 % 50 < batch:
            print(f"[rfdetr] frame {f0}/{n_frames}  {len(d)} persons  {det_s / (fs[-1] + 1) * 1000:.0f} ms/frame "
                  f"peak {torch.cuda.max_memory_allocated() / GB:.2f} GB", flush=True)
    wall = time.perf_counter() - t_all
    prof.close()
    if out is not None:
        out.close()
    meta = {
        "detector": f"rf-detr-seg-{model_size.lower()}", "resolution": int(model.model.resolution),
        "fp16_optimized": fp16, "threshold": threshold, "batch": batch, "gpu": torch.cuda.get_device_name(0),
        "n_frames": n_frames, "n_dets": n_det, "dets_per_frame": round(n_det / n_frames, 2),
        "load_s": round(load_s, 1), "frame_read_s": round(read_s, 2), "detector_s": round(det_s, 2),
        "rle_io_s": round(io_s, 2), "wall_s": round(wall, 2), "detector_fps": round(n_frames / det_s, 2),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / GB, 3), "host_rss_peak_gb": round(host_peak, 3),
    }
    write_json(out_dir / "detect_meta.json", meta)
    print(f"[rfdetr] done: {n_det} persons ({meta['dets_per_frame']}/frame)  detector {meta['detector_fps']} fps "
          f"({det_s / n_frames * 1000:.0f} ms/frame) wall {wall:.1f}s  peak {meta['gpu_peak_gb']} GB", flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="2XLarge", choices=["Nano", "Small", "Medium", "Large", "XLarge", "2XLarge"])
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="low on purpose: McByte's second association uses detections below its high threshold")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--fp32", action="store_true", help="skip optimize_for_inference(fp16)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-masks", action="store_true")
    a = ap.parse_args()
    run(read_json(a.frames_meta), a.out, a.model, a.threshold, a.batch, not a.fp32, a.max_frames, not a.no_masks)
