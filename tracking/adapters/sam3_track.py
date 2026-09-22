#!/usr/bin/env python3
"""Bake-off adapter D — SAM 3, text-prompted open-vocabulary detection + tracking (transformers).

Unlike every other tracker in this module, SAM 3 needs no external detector: prompted with the text
"person", `Sam3VideoModel` is documented to detect *and* track every matching instance across the whole
video natively, in one session — true multi-object support, not the per-object singleton-state
workaround SAMURAI needed (`sam3_track.py` here uses transformers' from-scratch SAM3 video pipeline, not
the SAM2-derived one SAMURAI patches). Whether its internal detector actually re-detects new entrants
as they appear (as opposed to only seeding once, SAMURAI's problem) is documented behavior we have not
yet verified against real footage — this adapter is written against the HF model card's API and needs a
first real run to confirm before trusting it for the bake-off. Requires facebook/sam3 access
(https://huggingface.co/facebook/sam3 — gated, request access and log in with `hf auth login`).

Writes the same tracks.jsonl schema as the other adapters, plus masks.jsonl (RLE, same schema as
samurai_track.py) since SAM3 also returns real per-pixel masks, for render_masks.py.

    python3 sam3_track.py --frames-meta <seq>/frames_meta.json --out <out_dir> --text person [--gpu 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import frame_path, read_json, write_json


def mask_to_rle(mask: np.ndarray):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


def run(frames_meta, out_dir, text="person", model_id="facebook/sam3", gpu="0", save_masks=True):
    import torch
    from PIL import Image
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir, W, H, n_frames = frames_meta["img_dir"], frames_meta["width"], frames_meta["height"], frames_meta["n_frames"]
    device = f"cuda:{gpu}"

    print(f"[sam3] loading {model_id}")
    model = Sam3VideoModel.from_pretrained(model_id).to(device, dtype=torch.bfloat16)
    processor = Sam3VideoProcessor.from_pretrained(model_id)

    frames = [Image.open(frame_path(img_dir, f)).convert("RGB") for f in range(n_frames)]
    print(f"[sam3] {len(frames)} frames loaded, prompting text='{text}'")

    inference_session = processor.init_video_session(
        video=frames, inference_device=device, processing_device="cpu",
        video_storage_device="cpu", dtype=torch.bfloat16,
    )
    inference_session = processor.add_text_prompt(inference_session=inference_session, text=text)

    tracks_path, masks_path = out_dir / "tracks.jsonl", out_dir / "masks.jsonl"
    mask_out = open(masks_path, "w") if save_masks else None
    n, tids = 0, set()
    with open(tracks_path, "w") as out:
        for model_outputs in model.propagate_in_video_iterator(inference_session=inference_session):
            f = model_outputs.frame_idx
            processed = processor.postprocess_outputs(inference_session, model_outputs)
            obj_ids = processed["object_ids"].tolist()
            scores = processed["scores"].tolist()
            boxes = processed["boxes"].tolist()  # xyxy, absolute
            masks = processed["masks"]  # [n_obj, H, W] boolean-ish
            for i, oid in enumerate(obj_ids):
                box = boxes[i]
                box = [max(0.0, box[0]), max(0.0, box[1]), min(float(W), box[2]), min(float(H), box[3])]
                if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                    continue
                out.write(json.dumps({"f": int(f), "tid": int(oid), "box": [round(v, 1) for v in box],
                                      "score": round(float(scores[i]), 3)}) + "\n")
                if mask_out is not None:
                    m = masks[i].cpu().numpy() > 0.5
                    mask_out.write(json.dumps({"f": int(f), "tid": int(oid), "rle": mask_to_rle(m)}) + "\n")
                tids.add(oid)
                n += 1
            if f % 25 == 0:
                mem = torch.cuda.max_memory_allocated(device) / 1e9
                print(f"[sam3] frame {f}/{n_frames}, {n} boxes, {len(tids)} ids so far, "
                      f"peak GPU mem so far {mem:.2f} GB")

    if mask_out is not None:
        mask_out.close()
    write_json(out_dir / "tracks_meta.json", {"tracker": "sam3", "n_obs": n, "n_tracks": len(tids),
               "model_id": model_id, "text": text, "masks": str(masks_path) if save_masks else None})
    print(f"[sam3] {n} boxes, {len(tids)} ids → {tracks_path}")
    return tracks_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="person", help="open-vocabulary text prompt")
    ap.add_argument("--model-id", default="facebook/sam3")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--no-masks", action="store_true")
    a = ap.parse_args()
    run(read_json(a.frames_meta), a.out, a.text, a.model_id, a.gpu, not a.no_masks)
