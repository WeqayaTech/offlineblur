#!/usr/bin/env python3
"""Video in -> blurred video out: RF-DETR-Seg persons -> EdgeTAM mask tracking -> per-track zero-shot gender
(PE-Core-L by default) -> GPU pixelation. fast_blur.py with McByte replaced by EdgeTAM (facebookresearch/EdgeTAM,
an on-device SAM 2: each person is a promptable object with its own memory of past masks).

EdgeTAM's video predictor (SAM 2's) refuses new objects once tracking has started, and people keep entering a
street scene, so this drives the model's per-frame `track_step` directly, one memory bank per person:
  1. every live track is propagated to the new frame from its own memory (mask + object-present score);
  2. RF-DETR detections (>= --threshold) are matched to the propagated masks by mask IoU (>= --match-iou);
  3. an unmatched detection >= --new-track whose mask is not already covered by tracked people (< --cover) starts
     a new track, prompted with its box on that frame (a conditioning frame, as a user click would be);
  4. a track whose mask overlaps an older track's by >= --dup-iou is dropped (same person twice), and a track no
     detection has confirmed for --lost-seconds is ended.
The blur uses EdgeTAM's own mask on every frame the object is present, including frames the detector missed.
Classifier views are only taken on frames where a detection confirmed the track (so they show a person).

The work lives in `EdgeTAMEngine` (models loaded once; `reset` per video or chunk; `run_batch` per batch of frames;
`write_blurred` for pass 2) so dist_edgetam.py can run the same code on chunks of a video across GPUs.

    python3 fast_blur_edgetam.py --video in.mp4 --out out_blurred.mp4 [--debug-out labels.mp4]
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

from fast_blur import (GB, PERSON, Timer, blur_region, classify_tracks, clip_crop, load_classifier, reader,
                       region_mask, render_debug)

TAM_MEAN, TAM_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def add_args(ap):
    ap.add_argument("--rf-model", default="2XLarge")
    ap.add_argument("--batch", type=int, default=8, help="frames per RF-DETR forward")
    ap.add_argument("--threshold", type=float, default=0.15, help="detections used for matching")
    ap.add_argument("--new-track", type=float, default=0.5, help="min detection score to start a track")
    ap.add_argument("--match-iou", type=float, default=0.3, help="detection-to-track mask IoU that confirms a track")
    ap.add_argument("--cover", type=float, default=0.5,
                    help="an unmatched detection whose mask is covered this much by tracked masks starts no track")
    ap.add_argument("--dup-iou", type=float, default=0.7, help="mask IoU at which the younger of two tracks is dropped")
    ap.add_argument("--lost-seconds", type=float, default=4.0, help="end a track no detection confirmed for this long")
    ap.add_argument("--edgetam-cfg", default="edgetam.yaml")
    ap.add_argument("--edgetam-ckpt", default="/root/EdgeTAM/checkpoints/edgetam.pt")
    ap.add_argument("--clip-model", default="PE-Core-L-14-336")
    ap.add_argument("--clip-pretrained", default="meta")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--min-gap", type=int, default=5)
    ap.add_argument("--blur-min", type=float, default=0.25)
    ap.add_argument("--encoder", default="libx264", help="h264_nvenc or libx264")
    ap.add_argument("--no-batch-people", action="store_true",
                    help="step EdgeTAM one person per call (the original loop) instead of batched groups")


class Track:
    def __init__(self, tid, f):
        self.tid, self.born, self.last_det = tid, f, f
        self.od = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}   # EdgeTAM memory bank of this person


def compact(out, torch):
    """What later frames need from a track_step output (as SAM2VideoPredictor keeps it)."""
    mf = out["maskmem_features"]
    return {"maskmem_features": mf.to(torch.bfloat16) if mf is not None else None,
            "maskmem_pos_enc": out["maskmem_pos_enc"], "pred_masks": out["pred_masks"], "obj_ptr": out["obj_ptr"],
            "object_score_logits": out["object_score_logits"]}


def _memory_capture(torch):
    """A stand-in for EdgeTAM's memory_attention that records its inputs instead of running it, so upstream's own
    _prepare_memory_conditioned_features assembles each person's memory (frames, pointers, temporal positions)."""
    class Capture(torch.nn.Module):
        def forward(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0, num_spatial_mem=-1):
            self.got = (memory, memory_pos, num_obj_ptr_tokens, num_spatial_mem)
            return curr[0] if isinstance(curr, list) else curr
    return Capture()


def propagate_batched(tam, torch, f, vfeats, vpos, fsizes, tracks, capture):
    """One EdgeTAM tracking step for many people at once. Each person's memory is gathered exactly as upstream does
    (per person, cheap); people whose memory has the same layout are then run through memory attention, the mask
    decoder and the memory encoder as one batch. Returns [(out_dict_for_that_person)] in `tracks` order."""
    real = tam.memory_attention
    mems = []
    tam.memory_attention = capture
    try:
        for tr in tracks:
            tam._prepare_memory_conditioned_features(
                frame_idx=f, is_init_cond_frame=False, current_vision_feats=vfeats[-1:],
                current_vision_pos_embeds=vpos[-1:], feat_sizes=fsizes[-1:], output_dict=tr.od, num_frames=10 ** 7)
            mems.append(capture.got)
    finally:
        tam.memory_attention = real
    groups = {}
    for i, (mem, pos, nptr, nsp) in enumerate(mems):
        groups.setdefault((mem.shape[0], nptr, nsp), []).append(i)
    outs = [None] * len(tracks)
    C = tam.hidden_dim
    H, W = fsizes[-1]
    for (_, nptr, nsp), idx in groups.items():
        B = len(idx)
        memory = torch.cat([mems[i][0] for i in idx], dim=1)
        memory_pos = torch.cat([mems[i][1] for i in idx], dim=1)
        vf = [x.expand(-1, B, -1).contiguous() for x in vfeats]
        vp = [x.expand(-1, B, -1).contiguous() for x in vpos]
        pix = real(curr=vf[-1:], curr_pos=vp[-1:], memory=memory, memory_pos=memory_pos,
                   num_obj_ptr_tokens=nptr, num_spatial_mem=nsp)
        pix = pix.permute(1, 2, 0).reshape(B, C, H, W)
        high_res = [x.permute(1, 2, 0).reshape(B, x.size(2), *s) for x, s in zip(vf[:-1], fsizes[:-1])]
        _, _, _, low_res_masks, high_res_masks, obj_ptr, obj_score = tam._forward_sam_heads(
            backbone_features=pix, point_inputs=None, mask_inputs=None, high_res_features=high_res,
            multimask_output=tam._use_multimask(False, None))
        cur = {"point_inputs": None, "mask_inputs": None}
        tam._encode_memory_in_output(vf, fsizes, None, True, high_res_masks, obj_score, cur)
        for j, i in enumerate(idx):
            outs[i] = {"maskmem_features": cur["maskmem_features"][j:j + 1],
                       "maskmem_pos_enc": [t[j:j + 1] for t in cur["maskmem_pos_enc"]],
                       "pred_masks": low_res_masks[j:j + 1], "obj_ptr": obj_ptr[j:j + 1],
                       "object_score_logits": obj_score[j:j + 1]}
    return outs


class EdgeTAMEngine:
    """RF-DETR-Seg + EdgeTAM + classifier, models loaded once. Per video (or chunk): reset(), run_batch() over the
    frames in order, then classify() and write_blurred()."""

    def __init__(self, a):
        import rfdetr
        import torch
        import torch.nn.functional as F
        from sam2.build_sam import build_sam2_video_predictor
        torch.set_grad_enabled(False)
        self.a, self.torch, self.F = a, torch, F
        self.dev = dev = torch.device("cuda")
        t0 = time.perf_counter()
        self.rf = getattr(rfdetr, f"RFDETRSeg{a.rf_model}")()
        self.rf.optimize_for_inference(compile=True, batch_size=a.batch, dtype=torch.float16)
        self.res = int(self.rf.model.resolution)
        self.mean = torch.tensor(self.rf.means, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(self.rf.stds, device=dev).view(1, 3, 1, 1)
        self.tam = build_sam2_video_predictor(a.edgetam_cfg, a.edgetam_ckpt, device="cuda")
        self.S = self.tam.image_size                                        # 1024: frames squashed to S x S
        self.G = self.S // 4                                                # EdgeTAM low-res mask grid
        self.tmean = torch.tensor(TAM_MEAN, device=dev).view(1, 3, 1, 1)
        self.tstd = torch.tensor(TAM_STD, device=dev).view(1, 3, 1, 1)
        self.keep_mem = max(self.tam.num_maskmem, self.tam.max_obj_ptrs_in_encoder) + 1
        self.capture = _memory_capture(torch)
        self.clf = load_classifier(a.clip_model, a.clip_pretrained, dev)
        self.rf.model.inference_model(torch.zeros(a.batch, 3, self.res, self.res, device=dev, dtype=torch.float16))
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - t0
        self.mem_log = None

    def reset(self, W, H, fps, collect_from=0):
        """New video or chunk. Classifier views are only collected on frames >= collect_from (a chunk's own frames;
        the warm-up frames before them belong to the previous chunk)."""
        self.W, self.H, self.fps = W, H, fps
        self.lost_frames = int(round(self.a.lost_seconds * fps))
        self.collect_from = collect_from
        self.T = Timer(self.torch)
        self.store = defaultdict(dict)        # tid -> {f: (x1,y1,x2,y2, rx1,ry1,rx2,ry2, logits_crop_gpu)}
        self.cands = defaultdict(dict)        # tid -> {window: (quality, f, crop)}
        self.live = {}                        # tid -> Track
        self.next_tid = 0
        self.stats = defaultdict(int)

    def run_batch(self, batch, f_base):
        """batch: list of RGB uint8 frames (numpy), consecutive, the first being frame f_base."""
        a, torch, F, dev, T = self.a, self.torch, self.F, self.dev, self.T
        W, H, S, G, res = self.W, self.H, self.S, self.G, self.res
        tam, live, store, cands, stats = self.tam, self.live, self.store, self.cands, self.stats
        nb = len(batch)
        with T("upload+preprocess", sync=True):
            fr = torch.from_numpy(np.stack(batch)).to(dev, non_blocking=True)          # [B,H,W,3] uint8
            x = fr.permute(0, 3, 1, 2).float().div_(255)
            xr = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
            xr = ((xr - self.mean) / self.std).half()
            if nb < a.batch:
                xr = torch.cat([xr, xr[-1:].expand(a.batch - nb, -1, -1, -1)])
            xt = F.interpolate(x, size=(S, S), mode="bilinear", align_corners=False, antialias=True)
            xt = (xt - self.tmean) / self.tstd
        with T("rfdetr_forward", sync=True):
            boxes_n, logits, masks = self.rf.model.inference_model(xr)
        with T("rfdetr_postprocess", sync=True):
            prob = logits[:nb, :, PERSON].float().sigmoid()
            keepd = prob > a.threshold
            cxcywh = boxes_n[:nb].float()
            xyxy = torch.stack([(cxcywh[..., 0] - cxcywh[..., 2] / 2) * W, (cxcywh[..., 1] - cxcywh[..., 3] / 2) * H,
                                (cxcywh[..., 0] + cxcywh[..., 2] / 2) * W, (cxcywh[..., 1] + cxcywh[..., 3] / 2) * H], -1)
            xyxy[..., 0::2] = xyxy[..., 0::2].clamp(0, W)
            xyxy[..., 1::2] = xyxy[..., 1::2].clamp(0, H)
            per_frame = []
            for b in range(nb):
                idx = keepd[b].nonzero(as_tuple=True)[0]
                # detection masks on EdgeTAM's low-res grid (both grids map linearly onto the frame)
                if len(idx):
                    dm = masks[b, idx].float()
                    dm = F.interpolate(dm[:, None], size=(G, G), mode="bilinear", align_corners=False)[:, 0] > 0
                else:
                    dm = torch.zeros(0, G, G, dtype=torch.bool, device=dev)
                per_frame.append((xyxy[b, idx].cpu().numpy(), prob[b, idx].cpu().numpy(), dm))
        with T("edgetam_image", sync=True), torch.autocast("cuda", dtype=torch.bfloat16):
            bo_all = tam.forward_image(xt)                                    # the whole batch in one pass
        for b in range(nb):
            f = f_base + b
            bx, sc, dmask = per_frame[b]
            stats["persons"] += len(bx)
            with T("edgetam_image", sync=True), torch.autocast("cuda", dtype=torch.bfloat16):
                bo = {"backbone_fpn": [t[b:b + 1] for t in bo_all["backbone_fpn"]],
                      "vision_pos_enc": [t[b:b + 1] for t in bo_all["vision_pos_enc"]]}
                _, vfeats, vpos, fsizes = tam._prepare_backbone_features(bo)

            def step(track, point_inputs=None):
                out = tam.track_step(frame_idx=f, is_init_cond_frame=point_inputs is not None and not track.od[
                    "cond_frame_outputs"], current_vision_feats=vfeats, current_vision_pos_embeds=vpos,
                    feat_sizes=fsizes, point_inputs=point_inputs, mask_inputs=None, output_dict=track.od,
                    num_frames=10 ** 7, run_mem_encoder=True)
                key = "cond_frame_outputs" if point_inputs is not None else "non_cond_frame_outputs"
                track.od[key][f] = compact(out, torch)
                for g in [g for g in track.od["non_cond_frame_outputs"] if g < f - self.keep_mem]:
                    del track.od["non_cond_frame_outputs"][g]
                return out["pred_masks"][0, 0], float(out["object_score_logits"].float().max())

            def keep(track, out):
                track.od["non_cond_frame_outputs"][f] = compact(out, torch)
                for g in [g for g in track.od["non_cond_frame_outputs"] if g < f - self.keep_mem]:
                    del track.od["non_cond_frame_outputs"][g]

            # 1) propagate every live track from its own memory
            tids, lowres = [], []
            with T("edgetam_track", sync=True), torch.autocast("cuda", dtype=torch.bfloat16):
                if a.no_batch_people:
                    for tid, tr in live.items():
                        m, s = step(tr)
                        if s > 0:
                            tids.append(tid)
                            lowres.append(m)
                elif live:
                    items = list(live.items())
                    outs = propagate_batched(tam, torch, f, vfeats, vpos, fsizes, [tr for _, tr in items], self.capture)
                    scores = torch.stack([o["object_score_logits"].float().max() for o in outs]).cpu().tolist()
                    for (tid, tr), out, s in zip(items, outs, scores):
                        keep(tr, out)
                        if s > 0:
                            tids.append(tid)
                            lowres.append(out["pred_masks"][0, 0])
            with T("associate", sync=True):
                tm = torch.stack(lowres).float() if lowres else torch.zeros(0, G, G, device=dev)
                tb = tm > 0
                # 4a) duplicates: the younger of two tracks on one person goes
                if len(tids) > 1:
                    fl = tb.flatten(1).half()
                    inter = fl @ fl.T
                    area = fl.sum(1)
                    iou = (inter / (area[:, None] + area[None] - inter).clamp(min=1)).float().cpu().numpy()
                    drop = set()
                    for i in range(len(tids)):
                        for j in range(i + 1, len(tids)):
                            if iou[i, j] >= a.dup_iou and tids[i] not in drop and tids[j] not in drop:
                                drop.add(max(tids[i], tids[j]))
                    if drop:
                        stats["dup_dropped"] += len(drop)
                        keep_i = [i for i, t in enumerate(tids) if t not in drop]
                        for t in drop:
                            live.pop(t, None)
                        tids = [tids[i] for i in keep_i]
                        tm, tb = tm[keep_i], tb[keep_i]
                # 2) match detections to propagated masks (greedy on mask IoU)
                matched = {}                                                 # tid -> det index
                if len(bx) and len(tids):
                    dfl, tfl = dmask.flatten(1).half(), tb.flatten(1).half()
                    inter = dfl @ tfl.T
                    iou = (inter / (dfl.sum(1)[:, None] + tfl.sum(1)[None] - inter).clamp(min=1)).float().cpu().numpy()
                    used_d = set()
                    for d, t in sorted(np.argwhere(iou >= a.match_iou).tolist(), key=lambda p: -iou[p[0], p[1]]):
                        if d not in used_d and tids[t] not in matched:
                            matched[tids[t]] = d
                            used_d.add(d)
                            live[tids[t]].last_det = f
                # 3) new tracks from unmatched confident detections not covered by tracked people
                union = tb.any(0) if len(tids) else torch.zeros(G, G, dtype=torch.bool, device=dev)
                new = [d for d in np.argsort(-sc) if sc[d] >= a.new_track and d not in matched.values()]
            for d in new:
                with T("associate", sync=True):
                    dm = dmask[d]
                    cov = float((dm & union).sum()) / max(1.0, float(dm.sum()))
                if cov >= a.cover:
                    continue
                tr = Track(self.next_tid, f)
                self.next_tid += 1
                x1, y1, x2, y2 = (float(v) for v in bx[d])
                pts = {"point_coords": torch.tensor([[[x1 * S / W, y1 * S / H], [x2 * S / W, y2 * S / H]]], device=dev),
                       "point_labels": torch.tensor([[2, 3]], dtype=torch.int32, device=dev)}
                with T("edgetam_new", sync=True), torch.autocast("cuda", dtype=torch.bfloat16):
                    m, s = step(tr, pts)
                live[tr.tid] = tr
                stats["tracks_started"] += 1
                if s > 0:
                    tids.append(tr.tid)
                    mb = (m > 0)
                    tm = torch.cat([tm, m.float()[None]])
                    tb = torch.cat([tb, mb[None]])
                    union = union | mb
                    matched[tr.tid] = int(d)
            # 4b) end tracks no detection has confirmed for too long
            for tid in [t for t, tr in live.items() if f - tr.last_det > self.lost_frames]:
                del live[tid]
            # store masks + classifier views
            with T("mask_store+candidates", sync=True):
                if len(tids):
                    tbb = tm > 0
                    rows, cols = tbb.any(2), tbb.any(1)                          # [N,G]
                    ar = torch.arange(G, device=dev)
                    big = torch.full_like(ar, G)
                    y0 = torch.where(rows, ar, big).min(1).values
                    y1_ = torch.where(rows, ar, -1).max(1).values + 1
                    x0 = torch.where(cols, ar, big).min(1).values
                    x1_ = torch.where(cols, ar, -1).max(1).values + 1
                    bb = torch.stack([x0, y0, x1_, y1_], 1).cpu().numpy()
                for i, tid in enumerate(tids):
                    mx1, my1, mx2, my2 = (int(v) for v in bb[i])
                    if mx2 <= mx1 or my2 <= my1:
                        continue
                    region = (mx1 * W / G, my1 * H / G, mx2 * W / G, my2 * H / G)
                    crop = tm[i, my1:my2, mx1:mx2].half().clone()
                    store[tid][f] = (*region, *region, crop)
                    d = matched.get(tid)
                    if d is None:
                        stats["mask_frames_without_det"] += 1
                        continue
                    if f < self.collect_from:
                        continue
                    rx1, ry1, rx2, ry2 = region
                    q_ = float(sc[d]) * max(1.0, (rx2 - rx1) * (ry2 - ry1)) ** 0.5
                    if rx1 <= 2 or ry1 <= 2 or rx2 >= W - 2 or ry2 >= H - 2:
                        q_ *= 0.5
                    wins, wb = cands[tid], f // a.min_gap
                    if (wb in wins and q_ <= wins[wb][0]) or (
                            wb not in wins and len(wins) >= 2 * a.k and q_ <= min(v[0] for v in wins.values())):
                        continue
                    wins[wb] = (q_, f, clip_crop(fr[b], region, region, crop, self.clf["size"], W, H, F, torch))
                    if len(wins) > 2 * a.k:
                        del wins[min(wins, key=lambda k: wins[k][0])]
            stats["tracked_person_frames"] += len(tids)
            if self.mem_log:
                import psutil
                self.mem_log.write(json.dumps({
                    "f": f, "live": len(live), "present": len(tids),
                    "tam_mem_entries": sum(len(tr.od["cond_frame_outputs"]) + len(tr.od["non_cond_frame_outputs"])
                                           for tr in live.values()),
                    "stored_masks": sum(len(v) for v in store.values()),
                    "cand_views": sum(len(v) for v in cands.values()),
                    "gpu_alloc_gb": round(torch.cuda.memory_allocated() / GB, 3),
                    "host_rss_gb": round(psutil.Process().memory_info().rss / GB, 3)}) + "\n")

    def classify(self):
        with self.T("clip", sync=True):
            return classify_tracks(self.clf, self.cands, self.a.k, self.a.min_gap, self.a.blur_min)

    def grid_mask(self, rec):
        """A stored mask back on EdgeTAM's low-res grid (bool numpy [G, G]) — for stitching chunks."""
        rx1, ry1, rx2, ry2, crop = rec[4], rec[5], rec[6], rec[7], rec[8]
        G = self.G
        mx1, my1 = int(round(rx1 * G / self.W)), int(round(ry1 * G / self.H))
        m = np.zeros((G, G), bool)
        c = (crop > 0).cpu().numpy()
        m[my1:my1 + c.shape[0], mx1:mx1 + c.shape[1]] = c
        return m

    def write_blurred(self, frames, f0, women, out, first=0, union_rle=None):
        """Pixelate the masks of `women` (tids) on frames[first:] (frames[i] is frame f0 + i) and encode to `out`.
        union_rle: optional dict filled with {f: full-res RLE of everything blurred on f} (accuracy comparison)."""
        torch, F, T, dev, W, H = self.torch, self.F, self.T, self.dev, self.W, self.H
        by_frame = defaultdict(list)
        for tid in women:
            for f, rec in self.store[tid].items():
                by_frame[f].append(rec)
        enc = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
               "-r", f"{self.fps:.6f}", "-i", "-", "-c:v", self.a.encoder]
        enc += (["-preset", "p4", "-cq", "20"] if "nvenc" in self.a.encoder else ["-preset", "veryfast", "-crf", "20"])
        enc += ["-pix_fmt", "yuv420p", out]
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
        if union_rle is not None:
            from pycocotools import mask as mu
        for i in range(first, len(frames)):
            f = f0 + i
            with T("blur", sync=True):
                recs = by_frame.get(f)
                if recs:
                    frt = torch.from_numpy(frames[i]).to(dev)
                    for (x1, y1, x2, y2, rx1, ry1, rx2, ry2, crop) in recs:
                        blur_region(frt, (x1, y1, x2, y2), (rx1, ry1, rx2, ry2), crop, W, H, F, torch)
                    buf = frt.cpu().numpy()
                else:
                    buf = frames[i]
            if union_rle is not None and recs:
                m = torch.zeros(H, W, dtype=torch.bool, device=dev)
                for rec in recs:
                    region_mask(m, rec, W, H, F, torch)
                r = mu.encode(np.asfortranarray(m.cpu().numpy().astype(np.uint8)))
                union_rle[f] = {"size": [H, W], "counts": r["counts"].decode()}
            with T("encode_queue"):
                wq.put(buf.tobytes())
        wq.put(None)
        wt.join()
        proc.stdin.close()
        proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="blurred output .mp4")
    add_args(ap)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--dump-tracks", default=None, help="write {f, tid, box, filled} jsonl (untimed)")
    ap.add_argument("--dump-masks", default=None, help="write labelled full-res masks.jsonl (untimed, for GT eval)")
    ap.add_argument("--dump-blur", default=None, help="write the full-res union of blurred pixels per frame (jsonl)")
    ap.add_argument("--mem-log", default=None,
                    help="per-frame jsonl: live tracks, EdgeTAM memory entries, GPU allocated, host RSS (memory growth)")
    ap.add_argument("--debug-title", default="RF-DETR-Seg + EdgeTAM + PE-Core-L")
    ap.add_argument("--debug-out", default=None, help="annotated video (untimed): mask colour = track, box = gender")
    a = ap.parse_args()

    import cv2
    cap = cv2.VideoCapture(a.video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    eng = EdgeTAMEngine(a)
    torch = eng.torch
    eng.reset(W, H, fps)
    eng.mem_log = open(a.mem_log, "w") if a.mem_log else None
    torch.cuda.reset_peak_memory_stats()

    # ---------------- pass 1: decode -> detect -> EdgeTAM track -> collect classifier views
    q = queue.Queue(maxsize=64)
    stop = threading.Event()
    th = threading.Thread(target=reader, args=(a.video, q, stop), daemon=True)
    t_pass1 = time.perf_counter()
    th.start()
    frames_keep = []
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
        eng.run_batch(batch, f_base)
        frames_keep.extend(batch)
        f_base += len(batch)
    stop.set()
    pass1_s = time.perf_counter() - t_pass1
    n_frames = f_base

    labels, n_views = eng.classify()
    women = {t for t, v in labels.items() if v["label"] == "woman"}

    # ---------------- pass 2: pixelate women's masks on the GPU, encode
    t_pass2 = time.perf_counter()
    union = {} if a.dump_blur else None
    eng.write_blurred(frames_keep, 0, women, a.out, union_rle=union)
    pass2_s = time.perf_counter() - t_pass2

    t = dict(eng.T.t)
    t["decode(thread)"] = decode_s
    total = pass1_s + t.get("clip", 0) + pass2_s
    meta = {
        "video": a.video, "frames": n_frames, "size": [W, H], "gpu": torch.cuda.get_device_name(0),
        "rf_model": a.rf_model, "resolution": eng.res, "batch": a.batch, "encoder": a.encoder,
        "tracker": f"EdgeTAM ({a.edgetam_cfg})", "threshold": a.threshold, "new_track": a.new_track,
        "match_iou": a.match_iou, "cover": a.cover, "dup_iou": a.dup_iou, "lost_seconds": a.lost_seconds,
        "clip": f"{a.clip_model} ({a.clip_pretrained})", "k": a.k, "blur_min": a.blur_min,
        "tracks": len(eng.store), "labelled_tracks": len(labels), "women": len(women), "crops": n_views,
        **dict(eng.stats), "tracked_per_frame": round(eng.stats["tracked_person_frames"] / max(1, n_frames), 1),
        "load_s": round(eng.load_s, 1), "pass1_s": round(pass1_s, 2), "pass2_s": round(pass2_s, 2),
        "total_s": round(total, 2),
        "stage_ms_per_frame": {k: round(1000 * v / n_frames, 2) for k, v in sorted(t.items())},
        "end_to_end_hz": round(n_frames / total, 1), "pass1_hz": round(n_frames / pass1_s, 1),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / GB, 2),
    }
    Path(a.out).with_suffix(".json").write_text(
        json.dumps({"meta": meta, "labels": {str(k): v for k, v in labels.items()}}, indent=1))
    print(json.dumps(meta, indent=1), flush=True)

    if a.dump_blur:
        with open(a.dump_blur, "w") as fh:
            for f in sorted(union):
                fh.write(json.dumps({"f": f, "tid": -1, "prompt": "woman", "rle": union[f]}) + "\n")
    if a.dump_tracks:
        with open(a.dump_tracks, "w") as fh:
            for tid, recs in eng.store.items():
                for f, rec in sorted(recs.items()):
                    fh.write(json.dumps({"f": f, "tid": tid, "box": [round(v, 1) for v in rec[:4]],
                                         "filled": False}) + "\n")
    if a.debug_out:
        render_debug(a.debug_out, frames_keep, eng.store, labels, W, H, fps, eng.F, torch, a.debug_title)
    if a.dump_masks:
        from pycocotools import mask as mu
        with open(a.dump_masks, "w") as fh:
            for tid, recs in eng.store.items():
                lab = labels.get(tid, {"label": "man"})["label"]
                for f, rec in recs.items():
                    m = torch.zeros(H, W, dtype=torch.bool, device=eng.dev)
                    region_mask(m, rec, W, H, eng.F, torch)
                    r = mu.encode(np.asfortranarray(m.cpu().numpy().astype(np.uint8)))
                    fh.write(json.dumps({"f": f, "tid": tid, "prompt": lab,
                                         "rle": {"size": [H, W], "counts": r["counts"].decode()}}) + "\n")


if __name__ == "__main__":
    main()
