#!/usr/bin/env python3
"""SAM 3.1 as a per-frame concept DETECTOR only — no tracker, no memory, no identities.

Each frame independently gets the masks SAM 3.1's detector produces for a text prompt (default "woman"):
the exact detection stage of the video pipeline (`run_backbone_and_detection`: tri-neck backbone ->
grounding transformer -> mask head -> NMS -> score threshold -> edge suppression), minus everything the
tracker adds. The point is to measure what the detector costs on its own, so a cheap external tracker
can supply identities afterwards.

Several prompts (`--text woman,man,child`) share ONE backbone pass per frame: the frame is queried once
per prompt through the detector's own multi-query path (FindStage `text_ids`), and `_get_img_feats` only
runs the ViT for unique image ids. The ViT is ~75% of a frame on an L4, so extra prompts are cheap.
There is no per-frame state, so memory does not grow with clip length. Batch 1 is the fastest setting
on an L4 (measured: 3.86 fps at 1, 3.1-3.3 fps at 2/4/8, OOM at 16); `--batch` only applies to the
single-prompt `--faithful` check path.

Outputs (masks.jsonl schema, so render_gender.py / render_masks.py read it; `tid` is NOT an identity —
it is a unique per-detection id `f * 1000 + k`, and `det` is k):
  masks.jsonl    {f, tid, det, prompt, score, box[x1,y1,x2,y2], rle}   one row per prompt's detection
  profile.jsonl  one row per frame: detector seconds, postprocess seconds, detections, GPU memory
  detect_meta.json

    python3 sam31_detect.py --frames-meta <seq>/frames_meta.json --out <dir> --text woman,man,child
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_json, write_json
from sam31_track import GB, Probe, build_predictor, mask_to_rle, start_session


def detect_frame(model, state, f, feature_cache, batch, faithful):
    """Detection outputs for frame f: (scores[K], boxes_xyxy_norm[K,4], low-res mask logits[K,h,w]),
    already filtered exactly as the video pipeline filters them (threshold + edge suppression; NMS ran
    inside the detector)."""
    input_batch = state["input_batch"]
    geo = state["per_frame_geometric_prompt"][f]
    geo = state["constants"]["empty_geometric_prompt"] if geo is None else geo
    if faithful:
        # the pipeline's own call, which also builds the tracker's backbone features (wasted here)
        det_out, pos = model.run_backbone_and_detection(
            frame_idx=f, num_frames=state["num_frames"], input_batch=input_batch, geometric_prompt=geo,
            feature_cache=feature_cache, reverse=False, use_batched_grounding=True,
            batched_grounding_batch_size=batch)
        det_out = {"scores": det_out["scores"], "bbox": det_out["bbox"], "mask": det_out["mask"]}
    else:
        # steps 1-2 of run_backbone_and_detection verbatim, without step 3 (tracker features)
        out, _ = model.detector.forward_video_grounding_batched_multigpu(
            backbone_out={"img_batch_all_stages": input_batch.img_batch, **feature_cache["text_outputs"]},
            find_inputs=input_batch.find_inputs, geometric_prompt=geo, frame_idx=f,
            num_frames=state["num_frames"], grounding_cache=feature_cache.setdefault("grounding_cache", {}),
            track_in_reverse=False, return_sam2_backbone_feats=False,
            run_nms=model.det_nms_thresh > 0.0, nms_prob_thresh=model.score_threshold_detection,
            nms_iou_thresh=model.det_nms_thresh, nms_use_iom=model.det_nms_use_iom,
            feature_cache=feature_cache, batch_size=batch)
        det_out = {"scores": out["pred_logits"].squeeze(-1).sigmoid(), "bbox": out["pred_boxes_xyxy"],
                   "mask": out["pred_masks"]}
        pos = det_out["scores"] > model.score_threshold_detection
        if model.suppress_det_close_to_boundary:
            pos = pos & model._suppress_detections_close_to_boundary(det_out["bbox"])
    keep = pos[0]
    return det_out["scores"][0][keep], det_out["bbox"][0][keep], det_out["mask"][0][keep]


def detect_frame_multi(model, state, f, text_outputs, n_prompts):
    """All prompts on frame f in one grounding call sharing one backbone pass. Returns, per prompt,
    (scores[K], boxes_xyxy_norm[K,4], low-res mask logits[K,h,w]) filtered exactly like the pipeline:
    NMS within each prompt, score threshold, edge suppression."""
    import dataclasses
    import numpy as np
    import torch
    from sam3.model.sam3_multiplex_detector_utils import nms_masks

    det = model.detector
    fi = state["input_batch"].find_inputs[f]
    fis = [dataclasses.replace(fi, text_ids=torch.full_like(fi.text_ids, k)) for k in range(n_prompts)]
    batched = det._batch_find_inputs(fis, 0, n_prompts)
    # _batch_find_inputs numbers images by chunk position; every query here is the same frame
    batched = dataclasses.replace(batched, img_ids=torch.full_like(batched.img_ids, f),
                                  img_ids_np=np.full(n_prompts, f))
    geo = det._batch_geometric_prompts_from_list([det._get_geo_prompt_from_find_input(x) for x in fis])
    out = det.forward_grounding(
        backbone_out={"img_batch_all_stages": state["input_batch"].img_batch, **text_outputs},
        find_input=batched, find_target=None, geometric_prompt=geo, feature_cache={})
    probs = out["pred_logits"].squeeze(-1).sigmoid()
    if model.det_nms_thresh > 0.0:
        keep = nms_masks(pred_probs=probs, pred_masks=out["pred_masks"],
                         prob_threshold=model.score_threshold_detection, iou_threshold=model.det_nms_thresh,
                         nms_use_iom=model.det_nms_use_iom, do_compile=False, running_in_prod=False)
        probs = probs * keep
    pos = probs > model.score_threshold_detection
    if model.suppress_det_close_to_boundary:
        pos = pos & model._suppress_detections_close_to_boundary(out["pred_boxes_xyxy"])
    return [(probs[k][pos[k]], out["pred_boxes_xyxy"][k][pos[k]], out["pred_masks"][k][pos[k]])
            for k in range(n_prompts)]


def run(frames_meta, out_dir, prompts, checkpoint, batch=8, max_frames=None, save_masks=True,
        faithful=False, offload_video=False, overrides=None):
    import torch
    import torch.nn.functional as F

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = frames_meta["img_dir"]
    W, H = int(frames_meta["width"]), int(frames_meta["height"])
    n_total = int(frames_meta["n_frames"])
    n_frames = min(n_total, max_frames) if max_frames else n_total
    if n_frames < n_total:          # same frame-subset trick as sam31_track.py
        sub = out_dir / "frames_subset"
        sub.mkdir(exist_ok=True)
        for p in sorted(Path(img_dir).glob("*.jpg"))[:n_frames]:
            if not (sub / p.name).exists():
                (sub / p.name).symlink_to(p)
        img_dir = str(sub)
    device = torch.device("cuda")
    probe = Probe(device)

    t = time.perf_counter()
    pred, applied = build_predictor(checkpoint, 128, 16, False, False, overrides or {})
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t
    model = pred.model

    t = time.perf_counter()
    sid = start_session(pred, img_dir, offload_video)
    state = pred._all_inference_states[sid]["state"]
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t

    # add_prompt sets the text on every frame and computes the text features once; it also runs one
    # full (tracked) frame-0 inference, which is setup cost and excluded from the per-frame numbers
    if faithful and len(prompts) != 1:
        raise SystemExit("--faithful is the single-prompt pipeline call; pass one --text")
    t = time.perf_counter()
    pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompts[0]})
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        text_outputs = model.detector.backbone.forward_text(list(prompts), device=device)
    torch.cuda.synchronize()
    prompt_s = time.perf_counter() - t
    single = {"language_features": state["backbone_out"]["language_features"],
              "language_mask": state["backbone_out"]["language_mask"]}
    feature_cache = {"text_outputs": single, "text": {tuple(state["input_batch"].find_text_batch): single}}
    print(f"[sam31-det] {torch.cuda.get_device_name(0)} {n_frames} frames {W}x{H} prompts={prompts} "
          f"batch={batch} {'faithful' if faithful else 'lean'}  build {build_s:.1f}s load {load_s:.1f}s "
          f"prompt {prompt_s:.1f}s  {probe.read()}", flush=True)

    probe.reset()
    mask_out = open(out_dir / "masks.jsonl", "w") if save_masks else None
    prof = open(out_dir / "profile.jsonl", "w")
    det_s = post_s = io_s = 0.0
    n_det = 0
    t_all = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for f in range(n_frames):
            t0 = time.perf_counter()
            if faithful:
                per_prompt = [detect_frame(model, state, f, feature_cache, batch, faithful)]
            else:
                per_prompt = detect_frame_multi(model, state, f, text_outputs, len(prompts))
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            # the pipeline keeps a per-frame backbone cache for the tracker; nothing here reads it back
            for k in [k for k in feature_cache if isinstance(k, int)]:
                del feature_cache[k]
            rows, k = [], 0
            for prompt, (scores, boxes, low) in zip(prompts, per_prompt):
                if not len(scores):
                    continue
                masks = F.interpolate(low.unsqueeze(1).float(), size=(H, W), mode="bilinear",
                                      align_corners=False)[:, 0] > 0
                masks, scores, boxes = masks.cpu().numpy(), scores.float().cpu().numpy(), boxes.float().cpu().numpy()
                for i in range(len(scores)):
                    rows.append((k, prompt, float(scores[i]), boxes[i], masks[i]))
                    k += 1
            t2 = time.perf_counter()
            if mask_out is not None:
                for k, prompt, score, box, mask in rows:
                    x1, y1, x2, y2 = (float(v) for v in box)
                    mask_out.write(json.dumps({
                        "f": f, "tid": f * 1000 + k, "det": k, "prompt": prompt, "score": round(score, 3),
                        "box": [round(x1 * W, 1), round(y1 * H, 1), round(x2 * W, 1), round(y2 * H, 1)],
                        "rle": mask_to_rle(mask)}) + "\n")
            t3 = time.perf_counter()
            det_s += t1 - t0
            post_s += t2 - t1
            io_s += t3 - t2
            n_det += len(rows)
            row = {"f": f, "det_s": round(t1 - t0, 4), "post_s": round(t2 - t1, 4), "n_det": len(rows)}
            row.update(probe.read())
            prof.write(json.dumps(row) + "\n")
            if f % 50 == 0:
                print(f"[sam31-det] frame {f}/{n_frames}  {len(rows)} dets  "
                      f"{(time.perf_counter() - t_all) / (f + 1):.3f} s/frame  peak {row['gpu_peak']:.1f} GB",
                      flush=True)
    wall = time.perf_counter() - t_all
    prof.close()
    if mask_out is not None:
        mask_out.close()
    meta = {
        "tracker": None, "detector": "sam3.1", "prompts": list(prompts), "checkpoint": checkpoint,
        "gpu": torch.cuda.get_device_name(0), "batch": batch, "mode": "faithful" if faithful else "lean",
        "offload_video": offload_video, "config_overrides": applied, "n_frames": n_frames,
        "width": W, "height": H, "n_dets": n_det, "dets_per_frame": round(n_det / n_frames, 2),
        "build_s": round(build_s, 1), "load_frames_s": round(load_s, 1), "add_prompt_s": round(prompt_s, 1),
        "detector_s": round(det_s, 2), "postprocess_s": round(post_s, 2), "rle_io_s": round(io_s, 2),
        "wall_s": round(wall, 2),
        "detector_fps": round(n_frames / det_s, 2), "detector_plus_masks_fps": round(n_frames / (det_s + post_s), 2),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated(device) / GB, 3),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved(device) / GB, 3),
        "host_rss_peak_gb": round(probe.host_peak, 3),
    }
    write_json(out_dir / "detect_meta.json", meta)
    print(f"[sam31-det] done: {n_det} dets ({meta['dets_per_frame']}/frame)  detector {meta['detector_fps']} fps, "
          f"+masks {meta['detector_plus_masks_fps']} fps, wall {wall:.1f}s  peak {meta['gpu_peak_gb']} GB", flush=True)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="woman,man,child", help="comma-separated; all share one backbone pass")
    ap.add_argument("--checkpoint", default="/workspace/SAM 3.1/sam3.1_multiplex.pt")
    ap.add_argument("--batch", type=int, default=1, help="frames per batched pass (--faithful path only)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-masks", action="store_true", help="skip RLE/json writing (pure speed)")
    ap.add_argument("--faithful", action="store_true",
                    help="use the pipeline's run_backbone_and_detection (also builds unused tracker features)")
    ap.add_argument("--offload-video", action="store_true")
    ap.add_argument("--score-threshold-detection", type=float, default=None, help="upstream 0.4")
    a = ap.parse_args()
    prompts = [p.strip() for p in a.text.split(",") if p.strip()]
    run(read_json(a.frames_meta), a.out, prompts, a.checkpoint, a.batch, a.max_frames, not a.no_masks,
        a.faithful, a.offload_video,
        {"score_threshold_detection": a.score_threshold_detection, "batched_grounding_batch_size": a.batch})
