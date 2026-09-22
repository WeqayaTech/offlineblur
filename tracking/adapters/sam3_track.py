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

Long videos: one session's memory grows with the number of tracked objects, so `--chunk-frames N`
resets SAM 3 every N frames and re-links identities across the reset by mean per-pixel mask IoU over
a `--chunk-overlap` window (matching only within the same prompt). Peak memory then depends on the
chunk, not the video length. The stitch is geometric, so it carries an identity through a reset but
does not re-identify anyone who left and came back — that still needs appearance ReID.

    python3 sam3_track.py --frames-meta <seq>/frames_meta.json --out <dir> --text woman,man,person
    python3 sam3_track.py ... --chunk-frames 150 --chunk-overlap 10     # bounded memory, long clips
"""
from __future__ import annotations

import argparse
import gc
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


def rle_iou(dt, gt):
    """IoU matrix between two lists of RLE dicts (pycocotools wants `counts` as bytes)."""
    from pycocotools import mask as mu
    if not dt or not gt:
        return [[0.0] * len(gt) for _ in dt]
    def b(r):
        c = r["counts"]
        return {"size": r["size"], "counts": c.encode("ascii") if isinstance(c, str) else c}
    return mu.iou([b(r) for r in dt], [b(r) for r in gt], [0] * len(gt)).tolist()


def stitch(prev, cur, iou_thresh):
    """Match a new chunk's objects to the previous chunk's across their shared overlap frames.

    prev: {prompt: {gid:   {abs_frame: rle}}}   already-numbered objects from the previous chunk
    cur:  {prompt: {local: {abs_frame: rle}}}   this chunk's objects, ids local to its session
    Returns {(prompt, local): gid}.

    Objects are only ever matched WITHIN a prompt: a "woman" object must not inherit a "man"
    object's identity just because they sit on the same pixels. Score is the mean per-pixel mask
    IoU over the frames where both are present, so a single lucky frame cannot carry a match.
    Greedy best-first, which is enough here because a person overlapping two candidates above the
    threshold on a multi-frame average is already a failure case no assignment rule fixes.
    """
    out = {}
    for prompt, curobjs in cur.items():
        prevobjs = prev.get(prompt, {})
        if not prevobjs or not curobjs:
            continue
        pairs = []
        for gid, pframes in prevobjs.items():
            for local, cframes in curobjs.items():
                shared = set(pframes) & set(cframes)
                if not shared:
                    continue
                ious = [rle_iou([pframes[f]], [cframes[f]])[0][0] for f in sorted(shared)]
                m = sum(ious) / len(ious)
                if m >= iou_thresh:
                    pairs.append((m, gid, local))
        pairs.sort(reverse=True)
        used_g, used_c = set(), set()
        for m, gid, local in pairs:
            if gid in used_g or local in used_c:
                continue
            out[(prompt, local)] = gid
            used_g.add(gid); used_c.add(local)
    return out


def prune_session_memory(session, current_frame, keep, cond_keep=0):
    """Drop per-object memory entries the model can no longer read. Returns bytes freed by device.

    SAM 3's tracker stores `maskmem_features` / `maskmem_pos_enc` / `high_res_masks` for EVERY object
    on EVERY frame and never removes them, so a session's footprint grows linearly with video length.
    Measured on a 250-frame 720p clip with the person prompt: ~40 MB/frame on the GPU and more again
    in host RAM, which is why long videos OOM.

    Almost all of it is unreachable. `_get_temporal_positions_and_previous_outputs` walks
    `range(num_maskmem - 1, 0, -1)` at stride 1, so only the previous `num_maskmem - 1` (6 by default)
    non-conditioning frames are ever read. Conditioning frames are a separate, capped store
    (`max_cond_frame_num` = 4) and are never touched here.

`cond_keep` additionally trims conditioning frames, and is OFF by default because the experiment
    said so. The reasoning looked sound — attention takes only the `max_cond_frame_num` (4) closest and
    tracking is forward-only — but measured output differed from unpruned (7136 observations vs 7131)
    while memory barely moved (8.5 vs 8.6 GB at frame 275). It costs exactness and buys nothing, so it
    stays off unless someone finds a use for it.

    `keep` must stay above the non-cond read window AND above `hotstart_delay` (15), since hotstart
    resolves tracklets using recent frames. At the default of 20 this is lossless: same output, flat
    memory. Verified byte-for-byte against an unpruned run rather than assumed.
    """
    cutoff = current_frame - keep
    freed = {}

    def _drop(entry):
        if not isinstance(entry, dict):
            return
        for v in entry.values():
            for t in (v if isinstance(v, (list, tuple)) else [v]):
                if hasattr(t, "numel") and hasattr(t, "element_size"):
                    dev = str(t.device)
                    freed[dev] = freed.get(dev, 0) + t.numel() * t.element_size()

    for obj_idx, d in getattr(session, "output_dict_per_obj", {}).items():
        nc = d.get("non_cond_frame_outputs")
        if nc and cutoff > 0:
            for f in [f for f in nc if f < cutoff]:
                _drop(nc.pop(f, None))
        cf = d.get("cond_frame_outputs")
        if cf and cond_keep and len(cf) > cond_keep:
            for f in sorted(cf)[:-cond_keep]:
                _drop(cf.pop(f, None))
    return freed


def drop_removed_objects(session, model_outputs):
    """Free storage for objects SAM 3 has itself declared dead. Returns how many were dropped.

    Pruning old frames bounds per-object state, but the number of OBJECTS still only grows: every
    person ever seen keeps an entry for the rest of the video, and per-frame work scales with that
    count. Measured on a 900-frame clip: object count climbed past 70 and peak VRAM kept rising even
    with frame pruning on.

    This is safe precisely because we do not decide who is dead. `Sam3VideoSegmentationOutput`
    reports `removed_obj_ids` — objects the model's own keep-alive and hotstart heuristics have
    already discarded — so dropping their storage cannot change any future output. Anything still
    alive is left untouched.
    """
    removed = getattr(model_outputs, "removed_obj_ids", None)
    if not removed:
        return 0
    n = 0
    for oid in list(removed):
        try:
            session.remove_object(int(oid), strict=False)
            n += 1
        except Exception:
            pass
    return n


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


def _propagate(model, processor, frames, prompts, device, dtype, state_device, base_f,
               W, H, min_score, want_mask_rle, memory_keep=0, cond_keep=0, drop_removed=False):
    """One SAM 3 session over `frames`; yields (abs_frame, [entry, ...]).

    entry = {"prompt","oid","score","box",["rle"]}. The session is torn down and the GPU cache
    dropped on exit, which is what makes chunking bound memory rather than merely delay the OOM.
    """
    import torch
    session = processor.init_video_session(
        video=frames, inference_device=device, inference_state_device=state_device,
        processing_device="cpu", video_storage_device="cpu", dtype=dtype,
    )
    session = processor.add_text_prompt(inference_session=session, text=list(prompts)) or session
    try:
        with torch.inference_mode():
            for model_outputs in model.propagate_in_video_iterator(inference_session=session):
                lf = int(model_outputs.frame_idx)
                processed = processor.postprocess_outputs(session, model_outputs)
                obj_ids = _as_list(processed.get("object_ids"))
                if not obj_ids:
                    if memory_keep:
                        prune_session_memory(session, lf, memory_keep, cond_keep)
                    if drop_removed:
                        drop_removed_objects(session, model_outputs)
                    yield base_f + lf, []
                    continue
                scores = _as_list(processed.get("scores"))
                boxes = _as_list(processed.get("boxes"))
                masks = processed.get("masks")
                p2o = processed.get("prompt_to_obj_ids") or {}
                owner = {int(o): pr for pr, ids in p2o.items() for o in _as_list(ids)}
                if not owner and len(prompts) > 1:
                    raise SystemExit("[sam3] postprocess_outputs has no prompt_to_obj_ids — this "
                                     "transformers build cannot attribute objects to prompts; run "
                                     "one prompt per pass instead")
                entries = []
                for i, oid in enumerate(obj_ids):
                    oid = int(oid)
                    score = float(scores[i]) if i < len(scores) else 1.0
                    if score < min_score:
                        continue
                    b = boxes[i]
                    box = [max(0.0, float(b[0])), max(0.0, float(b[1])),
                           min(float(W), float(b[2])), min(float(H), float(b[3]))]
                    if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                        continue
                    e = {"prompt": owner.get(oid, prompts[0]), "oid": oid,
                         "score": score, "box": box}
                    if want_mask_rle and masks is not None:
                        m = masks[i]
                        m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
                        e["rle"] = mask_to_rle(np.squeeze(m) > 0.5)
                    entries.append(e)
                if memory_keep:
                    prune_session_memory(session, lf, memory_keep, cond_keep)
                if drop_removed:
                    drop_removed_objects(session, model_outputs)
                yield base_f + lf, entries
    finally:
        del session
        gc.collect()
        torch.cuda.empty_cache()


def run(frames_meta, out_dir, prompts, model_id="facebook/sam3", gpu="0", save_masks=True,
        dtype_name="bfloat16", max_frames=None, min_score=0.0, overrides=None, state_device="cpu",
        chunk_frames=0, chunk_overlap=10, stitch_iou=0.3, reid_window=60, reid_iou=0.3,
        memory_keep=20, cond_keep=0, drop_removed=False):
    """Track `prompts` across the clip, optionally in chunks with identity stitched across resets.

    chunk_frames=0 runs the whole clip in one session, which is the most accurate option but whose
    memory grows with clip length. Any positive value resets SAM 3 every chunk_frames frames, keeping
    `chunk_overlap` frames of context to re-link identities by mask IoU, so peak memory is set by the
    chunk rather than by the video length.

    Re-linking is two-stage, because the overlap window alone is not enough. Measured on the trial
    clip: of the identities alive across a boundary, a fifth were momentarily occluded during a
    10-frame window, so the window could not see them and they came back with a new id. Stage 2
    therefore matches an object appearing within `reid_window` frames of a reset against the LAST
    mask of any identity last seen before that reset. It is deliberately scoped to just after a
    reset: it repairs what chunking broke, it is not general re-entry ReID.
    """
    import torch
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = frames_meta["img_dir"]
    W, H = frames_meta["width"], frames_meta["height"]
    n_frames = frames_meta["n_frames"] if max_frames is None else min(max_frames, frames_meta["n_frames"])
    device = f"cuda:{gpu}"
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]

    chunk = n_frames if not chunk_frames else min(chunk_frames, n_frames)
    overlap = 0 if chunk >= n_frames else max(1, min(chunk_overlap, chunk - 1))
    step = chunk - overlap
    n_chunks = 1 if overlap == 0 else max(1, -(-(n_frames - overlap) // step))
    print(f"[sam3] loading {model_id} ({dtype_name}) on {device}, state on {state_device}", flush=True)
    model, processor, applied = build_model(model_id, device, dtype, overrides or {})
    print(f"[sam3] {n_frames} frames ({W}x{H}); prompts = {prompts}; "
          f"{n_chunks} chunk(s) of {chunk} frames, overlap {overlap}", flush=True)

    tracks_path, masks_path = out_dir / "tracks.jsonl", out_dir / "masks.jsonl"
    mask_out = open(masks_path, "w") if save_masks else None
    prompt_tids = {p: set() for p in prompts}
    per_prompt_obs = {p: 0 for p in prompts}
    n_obs, next_gid, n_stitched, n_relinked, t0 = 0, 1, 0, 0, time.time()
    carry = {}          # {prompt: {gid: {abs_frame: rle}}} — previous chunk's tail, for stitching
    last_seen = {}      # {prompt: {gid: (abs_frame, rle)}} — every identity's most recent mask
    written_upto = -1   # highest absolute frame already written

    with open(tracks_path, "w") as out:
        for ci in range(n_chunks):
            c0 = ci * step
            c1 = min(c0 + chunk, n_frames)
            if c0 >= n_frames:
                break
            frames = [Image.open(frame_path(img_dir, f)).convert("RGB") for f in range(c0, c1)]
            local2gid, buf, tail = {}, {}, {}
            matched = overlap == 0 or ci == 0

            def relink(e, af):
                """Stage 2: an object appearing just after a reset may be a pre-reset identity that was
                occluded through the overlap window. Match its current mask against the last mask of
                any identity last seen before this chunk began and not already claimed here."""
                if ci == 0 or af - c0 > reid_window or "rle" not in e:
                    return None
                pool = last_seen.get(e["prompt"], {})
                taken = set(local2gid.values())
                cands = [(gid, rle) for gid, (lf, rle) in pool.items()
                         if gid not in taken and lf < c0 + overlap]
                if not cands:
                    return None
                ious = rle_iou([e["rle"]], [r for _, r in cands])[0]
                best = max(range(len(cands)), key=lambda i: ious[i])
                return cands[best][0] if ious[best] >= reid_iou else None

            def emit(af, entries):
                nonlocal n_obs, n_relinked
                for e in entries:
                    gid = local2gid.get((e["prompt"], e["oid"]))
                    if gid is None:
                        gid = relink(e, af)
                        if gid is not None:
                            n_relinked += 1
                        else:
                            gid = _new_gid()
                        local2gid[(e["prompt"], e["oid"])] = gid
                    out.write(json.dumps({"f": af, "tid": gid, "box": [round(v, 1) for v in e["box"]],
                                          "score": round(e["score"], 3), "prompt": e["prompt"]}) + "\n")
                    if mask_out is not None and "rle" in e:
                        mask_out.write(json.dumps({"f": af, "tid": gid, "prompt": e["prompt"],
                                                   "score": round(e["score"], 3), "rle": e["rle"]}) + "\n")
                    prompt_tids.setdefault(e["prompt"], set()).add(gid)
                    per_prompt_obs[e["prompt"]] = per_prompt_obs.get(e["prompt"], 0) + 1
                    n_obs += 1
                    if "rle" in e:
                        last_seen.setdefault(e["prompt"], {})[gid] = (af, e["rle"])

            def _new_gid():
                nonlocal next_gid
                next_gid += 1
                return next_gid - 1

            for af, entries in _propagate(model, processor, frames, prompts, device, dtype,
                                          state_device, c0, W, H, min_score,
                                          save_masks or overlap > 0, memory_keep, cond_keep,
                                          drop_removed):
                if not matched and af > written_upto:
                    # the overlap window is complete: link this chunk's objects to the previous one
                    m = stitch(carry, buf, stitch_iou)
                    local2gid.update(m)
                    n_stitched += len(m)
                    print(f"[sam3] chunk {ci}: stitched {len(m)} of "
                          f"{sum(len(v) for v in buf.values())} objects visible in the overlap "
                          f"({n_relinked} re-linked after resets so far)", flush=True)
                    matched = True
                if af <= written_upto:
                    for e in entries:                       # overlap frame: match only, never rewrite
                        if "rle" in e:
                            buf.setdefault(e["prompt"], {}).setdefault(e["oid"], {})[af] = e["rle"]
                    continue
                emit(af, entries)
                written_upto = af
                if af >= c1 - overlap and overlap:
                    for e in entries:                       # this chunk's tail becomes next carry
                        if "rle" in e:
                            gid = local2gid.get((e["prompt"], e["oid"]))
                            if gid is not None:
                                tail.setdefault(e["prompt"], {}).setdefault(gid, {})[af] = e["rle"]
                if af % 25 == 0:
                    mem = torch.cuda.max_memory_allocated(device) / 1e9
                    per = " ".join(f"{p}={len(t)}" for p, t in prompt_tids.items())
                    print(f"[sam3] frame {af}/{n_frames}  {n_obs} obs  ids[{per}]  "
                          f"peak {mem:.1f} GB  {(time.time() - t0) / max(1, af + 1):.2f} s/frame", flush=True)
            carry = tail
            del frames

    if mask_out is not None:
        mask_out.close()
    elapsed = time.time() - t0
    write_json(out_dir / "prompts.json", {p: sorted(t) for p, t in prompt_tids.items()})
    write_json(out_dir / "tracks_meta.json", {
        "tracker": "sam3", "model_id": model_id, "prompts": list(prompts), "dtype": dtype_name,
        "state_device": state_device, "chunk_frames": chunk_frames, "chunk_overlap": overlap,
        "n_chunks": n_chunks, "stitch_iou": stitch_iou, "n_stitched": n_stitched,
        "reid_window": reid_window, "reid_iou": reid_iou, "n_relinked": n_relinked,
        "memory_keep": memory_keep, "cond_keep": cond_keep, "drop_removed": drop_removed,
        "n_obs": n_obs, "n_frames": n_frames,
        "n_tracks": sum(len(t) for t in prompt_tids.values()),
        "n_tracks_per_prompt": {p: len(t) for p, t in prompt_tids.items()},
        "n_obs_per_prompt": per_prompt_obs, "min_score": min_score,
        "config_overrides": {k: v[1] for k, v in applied.items()},
        "seconds": round(elapsed, 1), "sec_per_frame": round(elapsed / max(1, n_frames), 3),
        "masks": str(masks_path) if save_masks else None,
    })
    per = ", ".join(f"{p}: {len(t)} ids / {per_prompt_obs.get(p, 0)} obs" for p, t in prompt_tids.items())
    print(f"[sam3] done in {elapsed / 60:.1f} min — {per}"
          + (f"  ({n_stitched} stitched in-window + {n_relinked} re-linked after reset, "
                 f"{n_chunks} chunks)" if n_chunks > 1 else "")
          + f" -> {tracks_path}", flush=True)
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
    # chunking: reset the tracker periodically so peak memory is set by the chunk, not the video length
    ap.add_argument("--chunk-frames", type=int, default=0,
                    help="reset SAM 3 every N frames and re-link identities across the reset "
                         "(0 = one session for the whole clip)")
    ap.add_argument("--chunk-overlap", type=int, default=10,
                    help="frames shared between consecutive chunks, used only to re-link identities")
    ap.add_argument("--stitch-iou", type=float, default=0.3,
                    help="mean per-pixel mask IoU over the overlap required to carry an identity across a reset")
    ap.add_argument("--reid-window", type=int, default=60,
                    help="frames after a reset during which a new object may be re-linked to an identity "
                         "that was occluded through the overlap window (0 disables stage 2)")
    ap.add_argument("--reid-iou", type=float, default=0.3,
                    help="mask IoU against an identity's last-seen mask required for that re-link")
    ap.add_argument("--memory-keep-frames", type=int, default=20,
                    help="per object, keep only this many recent non-conditioning memory frames. The "
                         "model reads at most num_maskmem-1 (6) and hotstart needs ~15, so 20 is "
                         "lossless and makes a session's footprint flat instead of linear in video "
                         "length. 0 disables pruning (the stock behaviour that OOMs on long video)")
    ap.add_argument("--drop-removed-objects", action="store_true",
                    help="free storage for objects SAM 3 itself reports in removed_obj_ids. Bounds the "
                         "object count, which frame pruning alone does not")
    ap.add_argument("--cond-keep-frames", type=int, default=0,
                    help="per object, keep only this many recent CONDITIONING frames. OFF by default: "
                         "measured NOT lossless (7136 obs vs 7131 unpruned at cond-keep 8) and it barely "
                         "moved memory (8.5 vs 8.6 GB), so it buys nothing and costs exactness")
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
        }, a.state_device, a.chunk_frames, a.chunk_overlap, a.stitch_iou,
        a.reid_window, a.reid_iou, a.memory_keep_frames, a.cond_keep_frames,
        a.drop_removed_objects)
