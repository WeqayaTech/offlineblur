#!/usr/bin/env python3
"""Where does a SAM 3.1 session's memory go? Walks the live inference state and sums tensor bytes by
key path (integer keys collapsed to `#`), split by device, at a few frames of a real propagation.

Written because SAM 3.1's allocated VRAM climbs linearly with frames while the object count is flat
(9.4 -> 16.7 GB over 240 frames of the trial clip, then OOM on a 22 GB L4) — the same shape as the
SAM 3 state leak. This names the containers that grow before anything is pruned.

    python3 sam31_state_audit.py --frames-meta <seq>/frames_meta.json --at 40,80,120
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "adapters"))
from common import read_json
from sam31_track import build_predictor, start_session

GB = 1024 ** 3


def walk(obj, path, acc, seen):
    import torch
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        acc[(path, obj.device.type)][0] += obj.numel() * obj.element_size()
        acc[(path, obj.device.type)][1] += 1
    elif isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}/{'#' if isinstance(k, int) else k}", acc, seen)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            walk(v, f"{path}[]", acc, seen)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith("sam3"):
        walk(vars(obj), f"{path}<{type(obj).__name__}>", acc, seen)


def report(state, frame, top):
    import torch
    acc = defaultdict(lambda: [0, 0])
    walk(state, "", acc, set())
    tot = defaultdict(int)
    for (p, dev), (b, n) in acc.items():
        tot[dev] += b
    print(f"\n=== frame {frame}: state holds " + ", ".join(f"{d} {b / GB:.2f} GB" for d, b in tot.items())
          + f"   (torch allocated {torch.cuda.memory_allocated() / GB:.2f} GB)", flush=True)
    for (p, dev), (b, n) in sorted(acc.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"  {b / GB:7.3f} GB  {n:6d} tensors  {dev:4s}  {p}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--checkpoint", default="/workspace/SAM 3.1/sam3.1_multiplex.pt")
    ap.add_argument("--text", default="person")
    ap.add_argument("--at", default="40,80,120", help="frames at which to audit")
    ap.add_argument("--top", type=int, default=18)
    ap.add_argument("--grounding-batch", type=int, default=4)
    a = ap.parse_args()

    import torch
    at = sorted(int(x) for x in a.at.split(","))
    meta = read_json(a.frames_meta)
    pred, _ = build_predictor(a.checkpoint, 128, 16, False, False,
                              {"batched_grounding_batch_size": a.grounding_batch})
    sid = start_session(pred, meta["img_dir"], False)
    state = pred._all_inference_states[sid]["state"]
    pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": a.text})
    for resp in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid,
                                            "propagation_direction": "forward",
                                            "evict_cached_frame_outputs": True}):
        f = resp["frame_index"]
        if at and f >= at[0]:
            report(state, f, a.top)
            at.pop(0)
        if not at:
            break
