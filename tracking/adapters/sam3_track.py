#!/usr/bin/env python3
"""SAM 3 — text-prompted open-vocabulary detection + tracking, and (step 1) gender as the prompt itself.

Unlike every other tracker in this module, SAM 3 needs no external detector: `Sam3VideoModel` takes one
or more short noun-phrase concept prompts and detects, segments *and* tracks every matching instance
across the whole video in one session, keeping identities stable. That makes it the only candidate here
that can collapse detect -> track -> "who do we blur?" into a single model: prompt it with "woman" and
the returned identities *are* the blur set, with no separate classifier, ReID pass or VLM vote.

Multi-prompt is native and cheap: `processor.add_text_prompt` accepts a list, vision features are shared
across prompts in one propagation pass, and `postprocess_outputs` returns `prompt_to_obj_ids` so every
object knows which concept produced it. The default prompt set is therefore

    woman,man,person

- `woman`  the actual selector under test (the blur set)
- `man`    a contrastive prompt: a person claimed by both "woman" and "man" is a disagreement, i.e. a
           low-confidence gender call worth counting rather than trusting
- `person` a recall control: anyone `person` finds that neither gender prompt covers is an *escape* —
           the failure mode that matters for blurring, since an escape ships unblurred.

`sam3_gender_report.py` turns those three into escape / conflict numbers; `render_gender.py` draws them.

Outputs (same schema family as the other adapters, plus a `prompt` field):
  tracks.jsonl   {f, tid, box[x1,y1,x2,y2], score, prompt}
  masks.jsonl    {f, tid, prompt, score, rle}   per-pixel RLE, for render_gender.py / render_masks.py
  prompts.json   {prompt: [tid, ...]}           which concept owns which identity
  tracks_meta.json

Needs gated access to facebook/sam3 (accept the licence at https://huggingface.co/facebook/sam3, then
`hf auth login` or set HF_TOKEN) and a transformers new enough to ship `Sam3VideoModel`.

    python3 sam3_track.py --frames-meta <seq>/frames_meta.json --out <dir> --text woman,man,person
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


def mask_to_rle(mask: np.ndarray):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def _as_list(x):
    """postprocess_outputs may hand back a tensor, an ndarray or a plain list depending on version."""
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def build_model(model_id, device, dtype, overrides: dict):
    """Load Sam3VideoModel, applying any config overrides (detection thresholds) before instantiation."""
    import torch
    from transformers import Sam3VideoConfig, Sam3VideoModel, Sam3VideoProcessor

    cfg = Sam3VideoConfig.from_pretrained(model_id)
    applied = {}
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            print(f"[sam3] warning: config has no '{k}' on this transformers version, ignored")
            continue
        applied[k] = (getattr(cfg, k), v)
        setattr(cfg, k, v)
    for k, (old, new) in applied.items():
        print(f"[sam3] config {k}: {old} -> {new}")
    model = Sam3VideoModel.from_pretrained(model_id, config=cfg).to(device, dtype=dtype)
    model.eval()
    processor = Sam3VideoProcessor.from_pretrained(model_id)
    return model, processor, applied


def run(frames_meta, out_dir, prompts, model_id="facebook/sam3", gpu="0", save_masks=True,
        dtype_name="bfloat16", max_frames=None, min_score=0.0, overrides=None, state_device="cpu"):
    import torch
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = frames_meta["img_dir"]
    W, H = frames_meta["width"], frames_meta["height"]
    n_frames = frames_meta["n_frames"] if max_frames is None else min(max_frames, frames_meta["n_frames"])
    device = f"cuda:{gpu}"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]

    print(f"[sam3] loading {model_id} ({dtype_name}) on {device}, state on {state_device}", flush=True)
    model, processor, applied = build_model(model_id, device, dtype, overrides or {})

    frames = [Image.open(frame_path(img_dir, f)).convert("RGB") for f in range(n_frames)]
    print(f"[sam3] {len(frames)} frames ({W}x{H}) loaded; prompts = {prompts}", flush=True)

    # inference_state_device defaults to inference_device, i.e. the GPU. That quietly parks every
    # tracked object's memory bank in VRAM, so usage grows with frames x objects and a crowd clip OOMs
    # mid-run (measured: 30 GB by frame 99 of 300 with 81 objects on a 32 GB card). Host RAM is plentiful
    # and the transfers are non_blocking, so the state belongs on the CPU.
    session = processor.init_video_session(
        video=frames, inference_device=device, inference_state_device=state_device,
        processing_device="cpu", video_storage_device="cpu", dtype=dtype,
    )
    # a list here runs every concept in ONE propagation pass, sharing vision features
    session = processor.add_text_prompt(inference_session=session, text=list(prompts)) or session

    tracks_path, masks_path = out_dir / "tracks.jsonl", out_dir / "masks.jsonl"
    mask_out = open(masks_path, "w") if save_masks else None
    prompt_tids = {p: set() for p in prompts}
    per_prompt_obs = {p: 0 for p in prompts}
    n_obs, n_dropped, t0 = 0, 0, time.time()

    with open(tracks_path, "w") as out, torch.inference_mode():
        for model_outputs in model.propagate_in_video_iterator(inference_session=session):
            f = int(model_outputs.frame_idx)
            processed = processor.postprocess_outputs(session, model_outputs)
            obj_ids = _as_list(processed.get("object_ids"))
            if not obj_ids:
                continue
            scores = _as_list(processed.get("scores"))
            boxes = _as_list(processed.get("boxes"))
            masks = processed.get("masks")
            # {prompt: [obj_id,...]} -> {obj_id: prompt}; absent on older builds, then everything is
            # attributed to the single prompt we asked for (and multi-prompt is not trustworthy).
            p2o = processed.get("prompt_to_obj_ids") or {}
            owner = {int(o): p for p, ids in p2o.items() for o in _as_list(ids)}
            if not owner and len(prompts) > 1:
                raise SystemExit("[sam3] postprocess_outputs has no prompt_to_obj_ids — this transformers "
                                 "build cannot attribute objects to prompts; run one prompt per pass instead")

            for i, oid in enumerate(obj_ids):
                oid = int(oid)
                prompt = owner.get(oid, prompts[0])
                score = float(scores[i]) if i < len(scores) else 1.0
                if score < min_score:
                    n_dropped += 1
                    continue
                b = boxes[i]
                box = [max(0.0, float(b[0])), max(0.0, float(b[1])),
                       min(float(W), float(b[2])), min(float(H), float(b[3]))]
                if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                    n_dropped += 1
                    continue
                out.write(json.dumps({"f": f, "tid": oid, "box": [round(v, 1) for v in box],
                                      "score": round(score, 3), "prompt": prompt}) + "\n")
                if mask_out is not None and masks is not None:
                    m = masks[i]
                    m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                    m = np.squeeze(m) > 0.5
                    mask_out.write(json.dumps({"f": f, "tid": oid, "prompt": prompt,
                                               "score": round(score, 3), "rle": mask_to_rle(m)}) + "\n")
                prompt_tids.setdefault(prompt, set()).add(oid)
                per_prompt_obs[prompt] = per_prompt_obs.get(prompt, 0) + 1
                n_obs += 1

            if f % 25 == 0:
                mem = torch.cuda.max_memory_allocated(device) / 1e9
                per = " ".join(f"{p}={len(t)}" for p, t in prompt_tids.items())
                print(f"[sam3] frame {f}/{n_frames}  {n_obs} obs  ids[{per}]  "
                      f"peak {mem:.1f} GB  {(time.time() - t0) / max(1, f + 1):.2f} s/frame", flush=True)

    if mask_out is not None:
        mask_out.close()
    elapsed = time.time() - t0
    write_json(out_dir / "prompts.json", {p: sorted(t) for p, t in prompt_tids.items()})
    write_json(out_dir / "tracks_meta.json", {
        "tracker": "sam3", "model_id": model_id, "prompts": list(prompts), "dtype": dtype_name,
        "state_device": state_device,
        "n_obs": n_obs, "n_dropped": n_dropped, "n_frames": n_frames,
        "n_tracks": sum(len(t) for t in prompt_tids.values()),
        "n_tracks_per_prompt": {p: len(t) for p, t in prompt_tids.items()},
        "n_obs_per_prompt": per_prompt_obs, "min_score": min_score,
        "config_overrides": {k: v[1] for k, v in applied.items()},
        "seconds": round(elapsed, 1), "sec_per_frame": round(elapsed / max(1, n_frames), 3),
        "masks": str(masks_path) if save_masks else None,
    })
    per = ", ".join(f"{p}: {len(t)} ids / {per_prompt_obs.get(p, 0)} obs" for p, t in prompt_tids.items())
    print(f"[sam3] done in {elapsed / 60:.1f} min — {per}  ({n_dropped} dropped) -> {tracks_path}")
    return tracks_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="woman,man,person",
                    help="comma-separated concept prompts, all run in one pass (default: woman,man,person)")
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--state-device", default="cpu",
                    help="where per-object memory banks live. 'cpu' (default) keeps VRAM flat; 'cuda:0' "
                         "is faster but grows with frames x objects and OOMs on crowd clips")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=0.0, help="drop observations below this score")
    ap.add_argument("--no-masks", action="store_true")
    # Sam3VideoConfig knobs worth sweeping — left at the model defaults unless passed.
    ap.add_argument("--score-threshold-detection", type=float, default=None, help="keep detections above (default 0.5)")
    ap.add_argument("--new-det-thresh", type=float, default=None, help="start a new object above (default 0.7)")
    ap.add_argument("--det-nms-thresh", type=float, default=None, help="detection NMS IoU (default 0.1)")
    ap.add_argument("--max-num-objects", type=int, default=None, help="cap tracked objects (default 10000; lower to bound VRAM)")
    a = ap.parse_args()

    prompts = [p.strip() for p in a.text.split(",") if p.strip()]
    if not prompts:
        raise SystemExit("--text needs at least one prompt")
    run(read_json(a.frames_meta), a.out, prompts, a.model_id, a.gpu, not a.no_masks, a.dtype,
        a.max_frames, a.min_score, {
            "score_threshold_detection": a.score_threshold_detection,
            "new_det_thresh": a.new_det_thresh,
            "det_nms_thresh": a.det_nms_thresh,
            "max_num_objects": a.max_num_objects,
        }, a.state_device)
