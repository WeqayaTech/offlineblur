#!/usr/bin/env python3
"""SAM 3.1 (Object Multiplex) — text-prompted detection + tracking, with time and memory profiling.

SAM 3.1 is the same concept-prompted detector as SAM 3 with a new tracker: instead of running the memory
decoder once per object, Object Multiplex packs objects into fixed-capacity buckets (`--multiplex-count`,
16) and runs each bucket jointly. Meta reports ~7x at 128 objects on an H100; this adapter exists to
measure what that buys on our crowd clips, on our GPUs, next to `sam3_track.py`.

It does NOT run through transformers. The HF repo `facebook/sam3.1` ships only `sam3.1_multiplex.pt`
(no transformers integration), so this uses the `sam3` package from github.com/facebookresearch/sam3
(`build_sam3_multiplex_video_predictor`, the request API from its sam3.1 notebook). Set that up with
`pod_setup_sam31.sh`, never in the transformers SAM 3 env.

Differences from `sam3_track.py` that matter when comparing numbers:
- One text prompt per session. The SAM 3 transformers API ran woman,man,person in one shared pass;
  here every prompt is its own session over the clip, so cost scales with the number of prompts.
- `max_num_objects` defaults to 16 upstream — far below a street crowd (60+ person ids on the trial
  clip). Anything past the cap is silently never tracked, so it is raised here (`--max-num-objects`).
- Upstream detector defaults differ from SAM 3: score_threshold_detection 0.4 (not 0.5), new_det_thresh
  0.65 (not 0.7). They are left as shipped unless overridden.
- Upstream crashes ("No points are provided") when an object removal leaves a multiplex bucket without a
  conditioning frame; `install_cond_frame_guard` prevents exactly that case (see its docstring).
- FlashAttention 3 is Hopper-only, so it is off unless `--fa3` is passed.
- The detector runs batched over `batched_grounding_batch_size` frames (16 upstream, sized for an 80 GB
  H100). On a 22 GB L4 that OOMs at frame 16 of a 720p crowd clip, so `--grounding-batch` defaults to 4.

Memory: like SAM 3, the tracker stores memory features, positional encodings, image features and masks
for every past frame in each bucket's `output_dict[_per_obj]["non_cond_frame_outputs"]` and never frees
them — allocated VRAM grows with *frames*, not objects (measured with sam31_state_audit.py: 4.9 GB of it
at frame 120 of the trial clip, OOM at frame 280 on a 22 GB L4). The tracker only reads the previous 6
non-conditioning frames (num_maskmem=7, stride 1) and object pointers from the previous 15
(max_obj_ptrs_in_encoder=16), so `--memory-keep-frames` (20, above both and hotstart_delay=15) drops
everything older. Conditioning frames are never pruned.

Outputs (same schema as sam3_track.py, so render_masks.py / render_gender.py / sam3_gender_report.py work):
  tracks.jsonl   {f, tid, box[x1,y1,x2,y2], score, prompt}
  masks.jsonl    {f, tid, prompt, score, rle}
  prompts.json   {prompt: [tid, ...]}
  tracks_meta.json   run config, counts, and the profile summary
  profile.jsonl  one row per yielded frame: model seconds, objects, GPU allocated/reserved/peak, host RSS

    python3 sam31_track.py --frames-meta <seq>/frames_meta.json --out <dir> --text person \
        --checkpoint "/workspace/SAM 3.1/sam3.1_multiplex.pt" --max-num-objects 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import read_json, write_json

GB = 1024 ** 3


def mask_to_rle(mask: np.ndarray):
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(r["size"][0]), int(r["size"][1])], "counts": r["counts"].decode("ascii")}


class Probe:
    """GPU + host memory readings. Peak is torch's allocator peak since the last reset()."""

    def __init__(self, device):
        import psutil
        import torch
        self.torch, self.device, self.proc = torch, device, psutil.Process()
        self.host_peak = 0.0

    def reset(self):
        self.torch.cuda.reset_peak_memory_stats(self.device)

    def read(self):
        c = self.torch.cuda
        free, total = c.mem_get_info(self.device)
        host = self.proc.memory_info().rss / GB
        self.host_peak = max(self.host_peak, host)
        return {"gpu_alloc": round(c.memory_allocated(self.device) / GB, 3),
                "gpu_reserved": round(c.memory_reserved(self.device) / GB, 3),
                "gpu_peak": round(c.max_memory_allocated(self.device) / GB, 3),
                "gpu_device_used": round((total - free) / GB, 3),
                "host_rss": round(host, 3)}


def build_predictor(checkpoint, max_num_objects, multiplex_count, use_fa3, compile_model, overrides):
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    install_cond_frame_guard()
    pred = build_sam3_multiplex_video_predictor(
        checkpoint_path=checkpoint, max_num_objects=max_num_objects, multiplex_count=multiplex_count,
        use_fa3=use_fa3, compile=compile_model, warm_up=compile_model, async_loading_frames=False)
    applied = {}
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(pred.model, k):
            raise SystemExit(f"[sam31] model has no attribute {k!r}; cannot override it")
        applied[k] = (getattr(pred.model, k), v)
        setattr(pred.model, k, v)
    for k, (old, new) in applied.items():
        print(f"[sam31] override {k}: {old} -> {new}", flush=True)
    return pred, {k: v[1] for k, v in applied.items()}


def start_session(pred, resource_path, offload_video):
    """Open a session without `handle_request(start_session)`, which is broken for SAM 3.1 at upstream
    HEAD 2345a4a: the shared base predictor passes `offload_state_to_cpu`, which the multiplex model's
    `init_state` does not accept (TypeError). Registers the session exactly as the base class does."""
    import uuid
    import torch
    with torch.inference_mode():
        state = pred.model.init_state(resource_path=resource_path, offload_video_to_cpu=offload_video,
                                      async_loading_frames=pred.async_loading_frames)
    sid = str(uuid.uuid4())
    now = time.time()
    pred._all_inference_states[sid] = {"state": state, "session_id": sid, "start_time": now,
                                       "last_use_time": now}
    return sid


GUARD = {"promoted_frames": 0, "fired": 0}


def _input_frames(sub, obj_idxs):
    out = set()
    for i in obj_idxs:
        out |= set(sub["point_inputs_per_obj"].get(i, {})) | set(sub["mask_inputs_per_obj"].get(i, {}))
    return out


def _promote_before_removal(sub, obj_ids):
    """Keep a bucket from losing its last conditioning frame when some of its objects are removed.

    Upstream bug (2345a4a, reproduced with the `woman` prompt on the trial clip, stock and pruned alike):
    an object added to an existing bucket later (obj 17, input mask at frame 92) has its input frame
    stored as NON-conditioning; the bucket's only conditioning frame belongs to the first object (obj 16,
    frame 90). When hotstart removes obj 16, frame 90 is downgraded, the bucket has no conditioning frame
    left, `_reset_tracking_results` wipes the whole bucket including obj 17's input and memory, and the
    next frame dies with "No points are provided". Here, only when a removal would leave the surviving
    objects with no conditioning frame, their own input frames are promoted to conditioning — what a
    frame holding an input mask is meant to be. Every run that did not crash is unaffected."""
    rm = {sub["obj_id_to_idx"][o] for o in obj_ids if o in sub["obj_id_to_idx"]}
    keep = [i for i in range(len(sub["obj_ids"])) if i not in rm]
    if not rm or not keep:
        return
    od = sub["output_dict"]
    cond = od["cond_frame_outputs"]
    removed_f, kept_f = _input_frames(sub, rm), _input_frames(sub, keep)
    if any(t not in removed_f or t in kept_f for t in cond):
        return                                  # a conditioning frame survives the removal
    GUARD["fired"] += 1
    cfi = sub.get("consolidated_frame_inds")
    for t in sorted(kept_f):
        out = od["non_cond_frame_outputs"].pop(t, None)
        if out is None:
            continue
        cond[t] = out
        for d in sub["output_dict_per_obj"].values():
            o = d["non_cond_frame_outputs"].pop(t, None)
            if o is not None:
                d["cond_frame_outputs"][t] = o
        if cfi:
            cfi["non_cond_frame_outputs"].discard(t)
            cfi["cond_frame_outputs"].add(t)
        GUARD["promoted_frames"] += 1


def install_cond_frame_guard():
    from sam3.model.video_tracking_multiplex_demo import Sam3VideoTrackingMultiplexDemo as T
    if getattr(T, "_offlineblur_cond_guard", False):
        return
    orig = T.remove_objects

    def remove_objects(self, inference_state, obj_ids, *a, **k):
        obj_ids = list(obj_ids)
        _promote_before_removal(inference_state, obj_ids)
        return orig(self, inference_state, obj_ids, *a, **k)

    T.remove_objects = remove_objects
    T._offlineblur_cond_guard = True


def prune_session_memory(state, current_frame, keep):
    """Drop non-conditioning tracker memory older than `keep` frames before `current_frame`.

    `current_frame` is the last frame the predictor *yielded*; the model itself is further ahead
    (hotstart + batched postprocessing buffer ~30 frames), so pruning relative to the yielded frame is
    conservative. Returns the number of per-frame entries dropped."""
    cutoff = current_frame - keep
    if cutoff <= 0:
        return 0
    n = 0
    for sub in state.get("sam2_inference_states", []):
        # an object's own input frame may need promoting to conditioning later (see the guard above)
        protect = _input_frames(sub, range(len(sub["obj_ids"])))
        dicts = [sub.get("output_dict", {})]
        for key in ("output_dict_per_obj", "temp_output_dict_per_obj"):
            dicts.extend(sub.get(key, {}).values())
        for d in dicts:
            nc = d.get("non_cond_frame_outputs")
            if not nc:
                continue
            for t in [t for t in nc if t < cutoff and t not in protect]:
                del nc[t]
                n += 1
    return n


def run(frames_meta, out_dir, prompts, checkpoint, gpu="0", save_masks=True, max_frames=None,
        min_score=0.0, max_num_objects=128, multiplex_count=16, use_fa3=False, compile_model=False,
        offload_video=False, memory_keep=20, overrides=None):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    import torch

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = frames_meta["img_dir"]
    W, H = int(frames_meta["width"]), int(frames_meta["height"])
    n_total = int(frames_meta["n_frames"])
    n_frames = min(n_total, max_frames) if max_frames else n_total
    if n_frames < n_total:
        # Never bound propagation with `max_frame_num_to_track`: upstream (2345a4a) is off by one there —
        # the propagation loop runs max+1 frames but batched grounding caps its chunks at start+max, so
        # the last frame gets an empty feature slice and the tracker crashes. Load only the first N
        # frames instead (which also keeps the unused frames out of VRAM).
        sub = out_dir / "frames_subset"
        sub.mkdir(exist_ok=True)
        for p in sorted(Path(img_dir).glob("*.jpg"))[:n_frames]:
            if not (sub / p.name).exists():
                (sub / p.name).symlink_to(p)
        img_dir = str(sub)
    device = torch.device("cuda")
    probe = Probe(device)
    print(f"[sam31] {torch.cuda.get_device_name(0)}  {n_frames} frames {W}x{H}  prompts={prompts}  "
          f"max_objects={max_num_objects} multiplex={multiplex_count} fa3={use_fa3} compile={compile_model}",
          flush=True)

    t = time.perf_counter()
    pred, applied = build_predictor(checkpoint, max_num_objects, multiplex_count, use_fa3, compile_model,
                                    overrides or {})
    torch.cuda.synchronize()
    phases = {"build_s": round(time.perf_counter() - t, 2), "after_build": probe.read()}
    print(f"[sam31] model built in {phases['build_s']} s  {phases['after_build']}", flush=True)

    tracks_path = out_dir / "tracks.jsonl"
    masks_path = out_dir / "masks.jsonl"
    out = open(tracks_path, "w")
    mask_out = open(masks_path, "w") if save_masks else None
    prof = open(out_dir / "profile.jsonl", "w")

    gid_of, prompt_tids, per_prompt = {}, {}, {}
    n_obs = 0
    t_all = time.perf_counter()
    for prompt in prompts:
        pp = per_prompt[prompt] = {}
        probe.reset()

        t = time.perf_counter()
        sid = start_session(pred, img_dir, offload_video)
        torch.cuda.synchronize()
        pp["start_session_s"] = round(time.perf_counter() - t, 2)
        pp["after_frames_loaded"] = probe.read()

        t = time.perf_counter()
        pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompt})
        torch.cuda.synchronize()
        pp["add_prompt_s"] = round(time.perf_counter() - t, 2)
        print(f"[sam31] {prompt}: frames loaded in {pp['start_session_s']} s, first-frame prompt "
              f"{pp['add_prompt_s']} s  {pp['after_frames_loaded']}", flush=True)

        # time spent inside the model (between asking for a frame and receiving it) is kept apart from
        # our own RLE/json work; outputs are .cpu().numpy() so every yield is already synchronised
        model_s = io_s = 0.0
        n_yield = n_pruned = 0
        state = pred._all_inference_states[sid]["state"]
        ids = set()
        stream = pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                             "propagation_direction": "forward",
                                             "evict_cached_frame_outputs": True})
        t_prop = time.perf_counter()
        while True:
            t0 = time.perf_counter()
            try:
                resp = next(stream)
            except StopIteration:
                break
            dt = time.perf_counter() - t0
            model_s += dt
            t1 = time.perf_counter()
            f, o = int(resp["frame_index"]), resp["outputs"]
            oids, probs = o["out_obj_ids"], o["out_probs"]
            boxes, masks = o["out_boxes_xywh"], o["out_binary_masks"]
            for i in range(len(oids)):
                s = float(probs[i])
                if s < min_score:
                    continue
                key = (prompt, int(oids[i]))
                if key not in gid_of:
                    gid_of[key] = len(gid_of)
                gid = gid_of[key]
                x, y, w, h = (float(v) for v in boxes[i])
                box = [round(x * W, 1), round(y * H, 1), round((x + w) * W, 1), round((y + h) * H, 1)]
                out.write(json.dumps({"f": f, "tid": gid, "box": box, "score": round(s, 3),
                                      "prompt": prompt}) + "\n")
                if mask_out is not None:
                    mask_out.write(json.dumps({"f": f, "tid": gid, "prompt": prompt, "score": round(s, 3),
                                               "rle": mask_to_rle(masks[i])}) + "\n")
                ids.add(gid)
                n_obs += 1
            if memory_keep:
                n_pruned += prune_session_memory(state, f, memory_keep)
            io_s += time.perf_counter() - t1
            row = {"prompt": prompt, "f": f, "model_s": round(dt, 4), "n_obj": int(len(oids))}
            row.update(probe.read())
            prof.write(json.dumps(row) + "\n")
            n_yield += 1
            if f % 25 == 0:
                el = time.perf_counter() - t_prop
                print(f"[sam31] {prompt} frame {f}/{n_frames}  {len(ids)} ids  {len(oids)} live  "
                      f"gpu alloc {row['gpu_alloc']:.1f} peak {row['gpu_peak']:.1f} GB  "
                      f"host {row['host_rss']:.1f} GB  {el / max(1, n_yield):.3f} s/frame", flush=True)
        pp.update({
            "propagate_s": round(time.perf_counter() - t_prop, 2), "propagate_model_s": round(model_s, 2),
            "consumer_io_s": round(io_s, 2), "frames": n_yield,
            "model_fps": round(n_yield / model_s, 2) if model_s else None,
            "gpu_peak_gb": round(torch.cuda.max_memory_allocated(device) / GB, 3),
            "gpu_reserved_end_gb": round(torch.cuda.memory_reserved(device) / GB, 3),
            "n_ids": len(ids), "n_pruned_entries": n_pruned,
        })
        prompt_tids[prompt] = ids
        pred.handle_request({"type": "close_session", "session_id": sid})
        print(f"[sam31] {prompt}: {len(ids)} ids, {n_yield} frames, model {model_s:.1f} s "
              f"({pp['model_fps']} fps), our io {io_s:.1f} s, peak {pp['gpu_peak_gb']:.2f} GB", flush=True)

    out.close()
    prof.close()
    if mask_out is not None:
        mask_out.close()
    elapsed = time.perf_counter() - t_all
    write_json(out_dir / "prompts.json", {p: sorted(t) for p, t in prompt_tids.items()})
    write_json(out_dir / "tracks_meta.json", {
        "tracker": "sam3.1", "checkpoint": checkpoint, "gpu": torch.cuda.get_device_name(0),
        "prompts": list(prompts), "max_num_objects": max_num_objects, "multiplex_count": multiplex_count,
        "use_fa3": use_fa3, "compile": compile_model, "offload_video": offload_video,
        "memory_keep": memory_keep, "cond_guard": dict(GUARD),
        "config_overrides": applied, "min_score": min_score, "n_frames": n_frames, "width": W, "height": H,
        "n_obs": n_obs, "n_tracks": len(gid_of),
        "n_tracks_per_prompt": {p: len(t) for p, t in prompt_tids.items()},
        "seconds": round(elapsed, 1), "sec_per_frame": round(elapsed / max(1, n_frames), 3),
        "host_rss_peak_gb": round(probe.host_peak, 3),
        "profile": dict(phases, per_prompt=per_prompt),
        "masks": str(masks_path) if save_masks else None,
    })
    per = ", ".join(f"{p}: {len(t)} ids" for p, t in prompt_tids.items())
    print(f"[sam31] done in {elapsed / 60:.1f} min — {per}, {n_obs} obs -> {tracks_path}", flush=True)
    return tracks_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="person", help="comma-separated; each prompt runs its own session")
    ap.add_argument("--checkpoint", default="/workspace/SAM 3.1/sam3.1_multiplex.pt")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--max-num-objects", type=int, default=128,
                    help="upstream default is 16, which silently drops most of a crowd")
    ap.add_argument("--multiplex-count", type=int, default=16, help="objects per multiplex bucket")
    ap.add_argument("--fa3", action="store_true", help="FlashAttention 3 (Hopper only)")
    ap.add_argument("--compile", action="store_true", help="torch.compile + warm-up (slow first build)")
    ap.add_argument("--offload-video", action="store_true",
                    help="keep decoded 1008x1008 frames on CPU instead of VRAM (long clips)")
    ap.add_argument("--grounding-batch", type=int, default=4,
                    help="frames per batched detector pass (upstream 16 OOMs a 22 GB card)")
    ap.add_argument("--postprocess-batch", type=int, default=None, help="upstream 16")
    ap.add_argument("--memory-keep-frames", type=int, default=20,
                    help="keep this many past frames of tracker memory (0 = stock, unbounded growth)")
    ap.add_argument("--score-threshold-detection", type=float, default=None, help="upstream 0.4")
    ap.add_argument("--new-det-thresh", type=float, default=None, help="upstream 0.65")
    a = ap.parse_args()

    prompts = [p.strip() for p in a.text.split(",") if p.strip()]
    if not prompts:
        raise SystemExit("--text needs at least one prompt")
    run(read_json(a.frames_meta), a.out, prompts, a.checkpoint, a.gpu, not a.no_masks, a.max_frames,
        a.min_score, a.max_num_objects, a.multiplex_count, a.fa3, a.compile, a.offload_video,
        a.memory_keep_frames,
        {"score_threshold_detection": a.score_threshold_detection, "new_det_thresh": a.new_det_thresh,
         "batched_grounding_batch_size": a.grounding_batch, "postprocess_batch_size": a.postprocess_batch})
