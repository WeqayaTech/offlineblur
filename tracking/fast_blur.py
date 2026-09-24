#!/usr/bin/env python3
"""Video in -> blurred video out, in one process: RF-DETR-Seg persons -> McByte ids -> per-track CLIP gender
-> GPU pixelation -> NVENC. The production-shaped version of run_rfdetr_mcbyte_clip.sh, built for speed.

What makes it fast (each measured against the script pipeline on the same pod):
  - no masks on disk and no full-resolution masks at all: RF-DETR's own predict() upsamples every detection's
    mask to 1280x720 and copies it to the CPU (~30 masks x 0.9 MB per frame). Here the model's native-resolution
    mask LOGITS are kept, cropped to each person's box, and only upsampled (inside the box, on the GPU) when a
    crop is classified or a woman is blurred. Tracking only needs boxes.
  - frames batched through a TensorRT-free but compiled fp16 RF-DETR (`optimize_for_inference`, fixed batch).
  - CLIP candidates are cut from the frame already on the GPU during pass 1 (best view per 5-frame window, 2K
    windows per track), so
    no frame is read twice for classification; preprocessing happens on the GPU.
  - decode runs in a thread; pass 2 pixelates on the GPU and pipes raw frames to ffmpeg h264_nvenc.

Same decisions as the script pipeline: person threshold 0.15, McByte (box overlap) high/activation 0.5,
K=10 views at >= 5 frames apart, masked crops padded square, P(woman)+P(girl) >= 0.25 -> blur.

    python3 fast_blur.py --video in.mp4 --out out_blurred.mp4 [--rf-model 2XLarge --batch 4 --encoder h264_nvenc]
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

GB = 1024 ** 3
PERSON = 1                                     # COCO category id == RF-DETR logit slot
CLASSES = ["woman", "man", "girl", "boy"]
TEMPLATES = ["a photo of a {}.", "a photo of a {} walking on the street.", "a cropped photo of a {}.",
             "a low resolution photo of a {}.", "a photo of a {} seen from behind.",
             "a blurry photo of a {}.", "a photo of a {} in a crowd."]


class Timer:
    def __init__(self, torch):
        self.t, self.torch = defaultdict(float), torch

    def __call__(self, key, sync=False):
        timer = self

        class _T:
            def __enter__(s):
                if sync:
                    timer.torch.cuda.synchronize()
                s.t0 = time.perf_counter()

            def __exit__(s, *a):
                if sync:
                    timer.torch.cuda.synchronize()
                timer.t[key] += time.perf_counter() - s.t0
        return _T()


def reader(path, q, stop):
    import cv2
    cap = cv2.VideoCapture(str(path))
    t = 0.0
    while not stop.is_set():
        t0 = time.perf_counter()
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = bgr[:, :, ::-1].copy()
        t += time.perf_counter() - t0
        q.put(rgb)
    q.put(None)
    q.put(("decode_s", t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="blurred output .mp4")
    ap.add_argument("--rf-model", default="2XLarge")
    ap.add_argument("--batch", type=int, default=4, help="frames per RF-DETR forward")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--high-conf", type=float, default=0.5)
    ap.add_argument("--clip-model", default="ViT-L-14-336")
    ap.add_argument("--clip-pretrained", default="openai")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-gap", type=int, default=5)
    ap.add_argument("--blur-min", type=float, default=0.25)
    ap.add_argument("--encoder", default="h264_nvenc", help="h264_nvenc or libx264")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--dump-masks", default=None, help="write labelled full-res masks.jsonl (untimed, for GT eval)")
    ap.add_argument("--cmc", action="store_true",
                    help="give McByte the frame so its camera-motion compensation runs (off without a frame)")
    ap.add_argument("--lost-buffer", type=int, default=30,
                    help="McByte lost_track_buffer, in 30 fps frames (rescaled: 30 -> 25 frames = 1.0 s at 25 fps)")
    ap.add_argument("--lost-seconds", type=float, default=None,
                    help="keep a lost (e.g. occluded) track alive this many seconds; overrides --lost-buffer. McByte "
                         "counts its buffer in 30 fps frames and rescales to the video fps, so this sets buffer = s * 30")
    ap.add_argument("--activation", type=float, default=None, help="new-track threshold (default: --high-conf)")
    ap.add_argument("--iou", default="iou", choices=["iou", "biou", "giou", "diou"],
                    help="box similarity for association; biou = buffered IoU (boxes enlarged by --biou-buffer)")
    ap.add_argument("--biou-buffer", type=float, default=0.3)
    ap.add_argument("--assoc1", type=float, default=0.1, help="min similarity, 1st association (high-score dets)")
    ap.add_argument("--assoc2", type=float, default=0.5, help="min similarity, 2nd association (low-score dets)")
    ap.add_argument("--assoc-unconfirmed", type=float, default=0.3, help="min similarity for unconfirmed tracks")
    ap.add_argument("--dump-dets", default=None, help="write per-frame person boxes+scores jsonl (for mcbyte_sweep.py)")
    ap.add_argument("--fill-gaps", type=int, default=0,
                    help="bridge detection holes up to N frames inside a track (box interpolated, nearest mask)")
    ap.add_argument("--dump-tracks", default=None, help="write {f, tid, box, filled} jsonl (untimed)")
    ap.add_argument("--debug-title", default="RF-DETR-Seg + McByte + CLIP", help="title bar text of --debug-out")
    ap.add_argument("--debug-out", default=None,
                    help="also write an annotated video (untimed): per-track mask colour, box coloured by gender, "
                         "'#id label P(female)'")
    a = ap.parse_args()

    import cv2
    import open_clip
    import rfdetr
    import supervision as sv
    import torch
    import torch.nn.functional as F
    from trackers import McByteTracker

    torch.set_grad_enabled(False)
    dev = torch.device("cuda")
    T = Timer(torch)
    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    # ---------------- models (load time reported separately; a service keeps them warm)
    t_load = time.perf_counter()
    rf = getattr(rfdetr, f"RFDETRSeg{a.rf_model}")()
    rf.optimize_for_inference(compile=True, batch_size=a.batch, dtype=torch.float16)
    res = int(rf.model.resolution)
    mean = torch.tensor(rf.means, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(rf.stds, device=dev).view(1, 3, 1, 1)
    clip, _, clip_pre = open_clip.create_model_and_transforms(a.clip_model, pretrained=a.clip_pretrained, device="cuda")
    clip.eval()
    csize = clip.visual.image_size
    csize = csize[0] if isinstance(csize, (tuple, list)) else csize
    cmean = torch.tensor(clip.visual.image_mean or (0.48145466, 0.4578275, 0.40821073), device=dev).view(1, 3, 1, 1)
    cstd = torch.tensor(clip.visual.image_std or (0.26862954, 0.26130258, 0.27577711), device=dev).view(1, 3, 1, 1)
    tok = open_clip.get_tokenizer(a.clip_model)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        txt = []
        for c in CLASSES:
            e = clip.encode_text(tok([t.format(c) for t in TEMPLATES]).cuda()).float()
            e = e / e.norm(dim=-1, keepdim=True)
            txt.append(e.mean(0) / e.mean(0).norm())
        txt = torch.stack(txt)
        scale = clip.logit_scale.exp().float()
    if a.lost_seconds is not None:
        a.lost_buffer = int(round(a.lost_seconds * 30))
    lost_frames = int(np.ceil(fps / 30.0 * a.lost_buffer)) if a.lost_buffer > 0 else 0   # McByte's own rescaling
    from trackers.utils.iou import BIoU, DIoU, GIoU, IoU
    iou_obj = {"iou": IoU(), "biou": BIoU(buffer_ratio=a.biou_buffer), "giou": GIoU(), "diou": DIoU()}[a.iou]
    tracker = McByteTracker(lost_track_buffer=a.lost_buffer, frame_rate=fps, iou=iou_obj,
                            minimum_iou_threshold_first_assoc=a.assoc1, minimum_iou_threshold_second_assoc=a.assoc2,
                            minimum_iou_threshold_unconfirmed_assoc=a.assoc_unconfirmed,
                            track_activation_threshold=a.activation if a.activation is not None else a.high_conf,
                            high_conf_det_threshold=a.high_conf, enable_mask_manager=False)
    # warm-up at the fixed batch size (compiled graph)
    rf.model.inference_model(torch.zeros(a.batch, 3, res, res, device=dev, dtype=torch.float16))
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load
    torch.cuda.reset_peak_memory_stats()

    # ---------------- pass 1: decode -> detect -> track -> collect CLIP candidates
    q = queue.Queue(maxsize=64)
    stop = threading.Event()
    th = threading.Thread(target=reader, args=(a.video, q, stop), daemon=True)
    t_pass1 = time.perf_counter()
    th.start()
    frames_keep = []                          # decoded frames, reused by pass 2 (RAM is plentiful here)
    store = defaultdict(dict)                 # tid -> {f: (x1,y1,x2,y2, rx1,ry1,rx2,ry2, logits_crop_gpu)}
    cands = defaultdict(dict)                 # tid -> {time window: (quality, f, crop uint8 [3,S,S] on GPU)}
    last_seen = {}
    n_det = 0
    dets_log = open(a.dump_dets, "w") if a.dump_dets else None
    decode_s = 0.0
    f_base = 0
    done = False
    while not done:
        batch = []
        while len(batch) < a.batch:
            item = q.get()
            if item is None:
                done = True
                decode_s = q.get()[1]
                break
            batch.append(item)
        if a.max_frames:
            batch = batch[:max(0, a.max_frames - f_base)]
            done = done or f_base + len(batch) >= a.max_frames
        if not batch:
            break
        nb = len(batch)
        with T("upload+preprocess", sync=True):
            fr = torch.from_numpy(np.stack(batch)).to(dev, non_blocking=True)          # [B,H,W,3] uint8
            x = fr.permute(0, 3, 1, 2).float().div_(255)
            x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
            x = ((x - mean) / std).half()
            if nb < a.batch:                                                          # compiled at fixed batch
                x = torch.cat([x, x[-1:].expand(a.batch - nb, -1, -1, -1)])
        with T("rfdetr_forward", sync=True):
            boxes_n, logits, masks = rf.model.inference_model(x)
        with T("rfdetr_postprocess", sync=True):
            prob = logits[:nb, :, PERSON].float().sigmoid()                           # [B,Q]
            keep = prob > a.threshold
            cxcywh = boxes_n[:nb].float()
            xyxy = torch.stack([(cxcywh[..., 0] - cxcywh[..., 2] / 2) * W, (cxcywh[..., 1] - cxcywh[..., 3] / 2) * H,
                                (cxcywh[..., 0] + cxcywh[..., 2] / 2) * W, (cxcywh[..., 1] + cxcywh[..., 3] / 2) * H], -1)
            xyxy[..., 0::2] = xyxy[..., 0::2].clamp(0, W)
            xyxy[..., 1::2] = xyxy[..., 1::2].clamp(0, H)
            Hm, Wm = masks.shape[-2:]
            per_frame = []
            for b in range(nb):
                idx = keep[b].nonzero(as_tuple=True)[0]
                per_frame.append((xyxy[b, idx].cpu().numpy(), prob[b, idx].cpu().numpy(), masks[b, idx]))
        for b in range(nb):
            f = f_base + b
            bx, sc, mk = per_frame[b]
            n_det += len(bx)
            if dets_log is not None:
                dets_log.write(json.dumps({"f": f, "boxes": np.round(bx, 1).tolist(),
                                           "scores": np.round(sc, 4).tolist()}) + "\n")
            with T("mcbyte"):
                res_d = tracker.update(sv.Detections(xyxy=bx.astype(np.float32), confidence=sc.astype(np.float32),
                                                     data={"i": np.arange(len(bx))}),
                                       frame=batch[b] if a.cmc else None)
            with T("mask_store+candidates", sync=True):
                for j in range(len(res_d)):
                    tid = int(res_d.tracker_id[j])
                    if tid < 0:
                        continue
                    i = int(res_d.data["i"][j])
                    x1, y1, x2, y2 = (float(v) for v in bx[i])
                    # crop mask logits to the box (mask coords), remember the image region it maps to
                    mx1, my1 = int(np.floor(x1 / W * Wm)), int(np.floor(y1 / H * Hm))
                    mx2, my2 = max(mx1 + 1, int(np.ceil(x2 / W * Wm))), max(my1 + 1, int(np.ceil(y2 / H * Hm)))
                    crop = mk[i, my1:my2, mx1:mx2].clone()
                    region = (mx1 * W / Wm, my1 * H / Hm, mx2 * W / Wm, my2 * H / Hm)
                    store[tid][f] = (x1, y1, x2, y2, *region, crop)
                    last_seen[tid] = f
                    q_ = float(sc[i]) * max(1.0, (x2 - x1) * (y2 - y1)) ** 0.5
                    if x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2:
                        q_ *= 0.5
                    # best view per `min_gap`-frame window, keeping the 2K best windows: spread over time by
                    # construction (a plain top-2K keeps neighbouring frames, which the gap rule then discards)
                    wins, wb = cands[tid], f // a.min_gap
                    if (wb in wins and q_ <= wins[wb][0]) or (
                            wb not in wins and len(wins) >= 2 * a.k and q_ <= min(v[0] for v in wins.values())):
                        continue
                    wins[wb] = (q_, f, clip_crop(fr[b], (x1, y1, x2, y2), region, crop, csize, W, H, F, torch))
                    if len(wins) > 2 * a.k:
                        del wins[min(wins, key=lambda k: wins[k][0])]
            frames_keep.append(batch[b])
        f_base += nb
    stop.set()
    pass1_s = time.perf_counter() - t_pass1
    if dets_log is not None:
        dets_log.close()
    n_frames = f_base

    # ---------------- classify: K spread views per track, one batched CLIP pass
    with T("clip", sync=True):
        views, owners = [], []
        for tid, wins in cands.items():
            chosen = []
            for q_, f, c in sorted(wins.values(), key=lambda t: -t[0]):
                if all(abs(f - g) >= a.min_gap for g, _ in chosen):
                    chosen.append((f, c))
                    if len(chosen) == a.k:
                        break
            for f, c in chosen:
                views.append(c)
                owners.append(tid)
        probs = []
        with torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(views), 128):
                xb = torch.stack(views[i:i + 128]).float().div_(255)
                xb = (xb - cmean) / cstd
                e = clip.encode_image(xb).float()
                e = e / e.norm(dim=-1, keepdim=True)
                probs.append(torch.softmax(e @ txt.T * scale, -1))
        probs = torch.cat(probs).cpu().numpy() if probs else np.zeros((0, 4))
    acc = defaultdict(list)
    for tid, p in zip(owners, probs):
        acc[tid].append(p)
    labels = {}
    for tid, ps in acc.items():
        p = np.mean(ps, 0)
        pf = float(p[0] + p[2])
        labels[tid] = {"label": "woman" if pf >= a.blur_min else "man", "p_female": round(pf, 3), "n_crops": len(ps)}
    women = {t for t, v in labels.items() if v["label"] == "woman"}
    filled = set()
    if a.fill_gaps:
        with T("fill_gaps"):
            for tid, recs in store.items():
                fs = sorted(recs)
                for f0, f1 in zip(fs, fs[1:]):
                    if 1 < f1 - f0 <= a.fill_gaps + 1:
                        r0, r1 = recs[f0], recs[f1]
                        for g in range(f0 + 1, f1):
                            t_ = (g - f0) / (f1 - f0)
                            geo = tuple((1 - t_) * u + t_ * v for u, v in zip(r0[:8], r1[:8]))
                            recs[g] = (*geo, (r0 if t_ < 0.5 else r1)[8])
                            filled.add((tid, g))

    # ---------------- pass 2: pixelate women's masks on the GPU, encode
    by_frame = defaultdict(list)
    for tid in women:
        for f, rec in store[tid].items():
            by_frame[f].append(rec)
    enc = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps:.6f}", "-i", "-", "-c:v", a.encoder]
    enc += (["-preset", "p4", "-cq", "20"] if "nvenc" in a.encoder else ["-preset", "veryfast", "-crf", "20"])
    enc += ["-pix_fmt", "yuv420p", a.out]
    t_pass2 = time.perf_counter()
    proc = subprocess.Popen(enc, stdin=subprocess.PIPE)
    wq = queue.Queue(maxsize=32)

    def writer():
        while True:
            buf = wq.get()
            if buf is None:
                break
            proc.stdin.write(buf)
    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    for f in range(n_frames):
        with T("blur", sync=True):
            recs = by_frame.get(f)
            if recs:
                fr = torch.from_numpy(frames_keep[f]).to(dev)
                for (x1, y1, x2, y2, rx1, ry1, rx2, ry2, crop) in recs:
                    blur_region(fr, (x1, y1, x2, y2), (rx1, ry1, rx2, ry2), crop, W, H, F, torch)
                out = fr.cpu().numpy()
            else:
                out = frames_keep[f]
        with T("encode_queue"):
            wq.put(out.tobytes())
    wq.put(None)
    wt.join()
    proc.stdin.close()
    proc.wait()
    pass2_s = time.perf_counter() - t_pass2

    t = dict(T.t)
    t["decode(thread)"] = decode_s
    gpu = t.get("upload+preprocess", 0) + t.get("rfdetr_forward", 0) + t.get("rfdetr_postprocess", 0) + t.get("clip", 0)
    total = pass1_s + t.get("clip", 0) + pass2_s
    meta = {
        "video": a.video, "frames": n_frames, "size": [W, H], "gpu": torch.cuda.get_device_name(0),
        "rf_model": a.rf_model, "resolution": res, "batch": a.batch, "encoder": a.encoder, "cmc": a.cmc,
        "high_conf": a.high_conf, "threshold": a.threshold, "lost_buffer": a.lost_buffer,
        "lost_frames": lost_frames, "iou": a.iou, "biou_buffer": a.biou_buffer, "assoc1": a.assoc1,
        "assoc2": a.assoc2, "assoc_unconfirmed": a.assoc_unconfirmed, "lost_seconds": round(lost_frames / fps, 2), "fps": round(fps, 3), "activation": a.activation, "fill_gaps": a.fill_gaps, "filled": len(filled),
        "clip": f"{a.clip_model} ({a.clip_pretrained})", "tracks": len(labels),
        "women": len(women), "crops": len(views), "persons": n_det, "load_s": round(load_s, 1),
        "pass1_s": round(pass1_s, 2), "pass2_s": round(pass2_s, 2), "total_s": round(total, 2),
        "stage_ms_per_frame": {k: round(1000 * v / n_frames, 2) for k, v in sorted(t.items())},
        "end_to_end_hz": round(n_frames / total, 1), "pass1_hz": round(n_frames / pass1_s, 1),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / GB, 2),
    }
    Path(a.out).with_suffix(".json").write_text(json.dumps({"meta": meta, "labels": {str(k): v for k, v in labels.items()}}, indent=1))
    print(json.dumps(meta, indent=1), flush=True)

    if a.dump_tracks:
        with open(a.dump_tracks, "w") as fh:
            for tid, recs in store.items():
                for f, rec in sorted(recs.items()):
                    fh.write(json.dumps({"f": f, "tid": tid, "box": [round(v, 1) for v in rec[:4]],
                                         "filled": (tid, f) in filled}) + "\n")

    if a.debug_out:                            # untimed: masks + track ids + gender, for eyeballing
        render_debug(a.debug_out, frames_keep, store, labels, W, H, fps, F, torch, a.debug_title)

    if a.dump_masks:                           # untimed: full-res masks for gender_gt_eval.py
        from pycocotools import mask as mu
        with open(a.dump_masks, "w") as fh:
            for tid, recs in store.items():
                lab = labels.get(tid, {"label": "man"})["label"]
                for f, rec in recs.items():
                    m = torch.zeros(H, W, dtype=torch.bool, device=dev)
                    region_mask(m, rec, W, H, F, torch)
                    r = mu.encode(np.asfortranarray(m.cpu().numpy().astype(np.uint8)))
                    fh.write(json.dumps({"f": f, "tid": tid, "prompt": lab,
                                         "rle": {"size": [H, W], "counts": r["counts"].decode()}}) + "\n")


PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
           (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180)]   # RGB


def render_debug(path, frames, store, labels, W, H, fps, F, torch, title="RF-DETR-Seg + McByte + CLIP"):
    """Original frames with every tracked person's mask (colour = track), box + '#id label P(female)' (box and
    text colour = gender: magenta woman, cyan man). Nothing is blurred, so what the pipeline saw stays visible."""
    import cv2
    by_frame = defaultdict(list)
    for tid, recs in store.items():
        for f, rec in recs.items():
            by_frame[f].append((tid, rec))
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "20", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    for f, frame in enumerate(frames):
        img = frame.copy()
        n_w = n_m = 0
        for tid, rec in by_frame.get(f, []):
            sub, (sx1, sy1, sx2, sy2) = _upsampled(rec, W, H, F, torch)
            if sub.numel():
                m = sub.cpu().numpy()
                roi = img[sy1:sy2, sx1:sx2]
                col = np.array(PALETTE[tid % len(PALETTE)], np.float32)
                roi[m] = (0.5 * roi[m] + 0.5 * col).astype(np.uint8)
            lab = labels.get(tid, {"label": "?", "p_female": float("nan")})
            gcol = (255, 0, 255) if lab["label"] == "woman" else (0, 220, 255)
            n_w += lab["label"] == "woman"
            n_m += lab["label"] != "woman"
            x1, y1, x2, y2 = (int(v) for v in rec[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), gcol, 2)
            txt = f"#{tid} {lab['label']} {lab['p_female']:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            ty = max(th + 4, y1 - 4)
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), gcol, -1)
            cv2.putText(img, txt, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        head = f"{title}   frame {f}   tracked {n_w + n_m}   women {n_w}   men {n_m}"
        cv2.rectangle(img, (0, 0), (W, 28), (0, 0, 0), -1)
        cv2.putText(img, head, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    enc.wait()


def _upsampled(rec, W, H, F, torch):
    """Box-local boolean mask (image pixels) for one stored detection, upsampled only inside its region."""
    x1, y1, x2, y2, rx1, ry1, rx2, ry2, crop = rec
    ix1, iy1, ix2, iy2 = int(rx1), int(ry1), int(np.ceil(rx2)), int(np.ceil(ry2))
    ix2, iy2 = min(ix2, W), min(iy2, H)
    up = F.interpolate(crop[None, None].float(), size=(max(1, iy2 - iy1), max(1, ix2 - ix1)),
                       mode="bilinear", align_corners=False)[0, 0] > 0
    # the model's box bounds the person; clip the region-sized mask to it
    bx1, by1, bx2, by2 = int(x1), int(y1), int(np.ceil(x2)), int(np.ceil(y2))
    sx1, sy1 = max(bx1, ix1), max(by1, iy1)
    sx2, sy2 = min(bx2, ix2), min(by2, iy2)
    return up[sy1 - iy1:sy2 - iy1, sx1 - ix1:sx2 - ix1], (sx1, sy1, sx2, sy2)


def region_mask(m, rec, W, H, F, torch):
    sub, (sx1, sy1, sx2, sy2) = _upsampled(rec, W, H, F, torch)
    if sub.numel():
        m[sy1:sy2, sx1:sx2] |= sub


def blur_region(fr, box, region, crop, W, H, F, torch):
    sub, (sx1, sy1, sx2, sy2) = _upsampled((*box, *region, crop), W, H, F, torch)
    if not sub.numel() or not sub.any():
        return
    roi = fr[sy1:sy2, sx1:sx2].permute(2, 0, 1)[None].float()                     # [1,3,h,w]
    h, w = roi.shape[-2:]
    block = max(2, int(0.045 * h))
    small = F.adaptive_avg_pool2d(roi, (max(1, h // block), max(1, w // block)))
    pix = F.interpolate(small, size=(h, w), mode="nearest")[0].permute(1, 2, 0).round().to(fr.dtype)
    fr[sy1:sy2, sx1:sx2] = torch.where(sub[..., None], pix, fr[sy1:sy2, sx1:sx2])


def clip_crop(frame, box, region, crop, S, W, H, F, torch):
    """Masked, square-padded, S x S uint8 CLIP view cut from the GPU frame (grey outside the dilated mask)."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.1
    cx1, cy1, cx2, cy2 = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(W, x2 + mx)), int(min(H, y2 + my))
    c = frame[cy1:cy2, cx1:cx2].permute(2, 0, 1).float()                            # [3,h,w]
    m = torch.zeros(cy2 - cy1, cx2 - cx1, device=frame.device, dtype=torch.bool)
    sub, (sx1, sy1, sx2, sy2) = _upsampled((*box, *region, crop), W, H, F, torch)
    if sub.numel():
        m[sy1 - cy1:sy2 - cy1, sx1 - cx1:sx2 - cx1] = sub
    m = F.max_pool2d(m[None, None].float(), 15, stride=1, padding=7)[0, 0] > 0      # 15x15 dilation
    c = torch.where(m[None], c, torch.full_like(c, 127.0))
    h, w = c.shape[-2:]
    s = max(h, w)
    sq = torch.full((3, s, s), 127.0, device=frame.device)
    sq[:, (s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = c
    sq = F.interpolate(sq[None], size=(S, S), mode="bicubic", align_corners=False)[0]
    return sq.clamp(0, 255).to(torch.uint8)


if __name__ == "__main__":
    main()
