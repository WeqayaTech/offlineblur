#!/usr/bin/env python3
"""The RF-DETR + EdgeTAM + PE-Core blur (fast_blur_edgetam.py) split in time across GPUs.

Every frame still gets the full per-person EdgeTAM computation; the speed comes from running chunks of the video in
parallel. EdgeTAM only carries memory forward, so a chunk can start anywhere once it has seen a short warm-up:

  1. track   (per chunk, on a GPU)  decode [start - warm, end), detect + track every frame, score every classifier
             view of the chunk's OWN frames [start, end); return per-track views (frame, quality, class
             probabilities), low-res masks of the warm-up frames (head) and of the last `warm` own frames (tail),
             and the chunk's own-frame masks (packed, for step 3)
  2. vote    (coordinator)  a track of chunk i is the person of chunk i-1 whose tail masks it overlaps
             (mask IoU >= stitch_iou) over the warm-up frames; views of one person are pooled across chunks and the
             same K-view vote as the single process decides woman / man
  3. blur    (per chunk)  pixelate that chunk's women on its own frames, encode an mp4 segment
  4. join    (coordinator)  concatenate the segments without re-encoding

The steps are plain functions (track_chunk, stitch_vote, blur_chunk, join_segments) so any launcher can drive them:
modal_edgetam.py runs them on Modal GPUs; the `worker` / `run` commands here run them over HTTP on your own GPUs:

    CUDA_VISIBLE_DEVICES=0 python3 dist_edgetam.py worker --port 8100      # one per GPU
    python3 dist_edgetam.py run --video in.mp4 --out out.mp4 --workers http://host:8100,http://host:8101
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_blur import label_from_probs, score_views, select_views   # noqa: E402


# ------------------------------------------------------------------ shared steps

def probe(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return W, H, fps, n


def chunk_bounds(n, chunks):
    return [(round(i * n / chunks), round((i + 1) * n / chunks)) for i in range(chunks)]


def decode_range(path, s0, n, W, H, fps):
    """Frames s0 .. s0+n-1 as RGB uint8 arrays (frame-accurate ffmpeg input seek: decoding restarts at the keyframe
    before s0 and discards up to it)."""
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{s0 / fps:.6f}", "-i", str(path), "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    size = W * H * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    p.wait()


def rle(m):
    from pycocotools import mask as mu
    return mu.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()


def track_chunk(eng, path, start, end, warm, keep_frames=None, pack=True):
    """Step 1 on one GPU. Returns the result for the coordinator (JSON-able; `masks` = packed own-frame masks unless
    pack=False). keep_frames: a list that receives the chunk's own decoded frames (to blur in place, see blur_frames)."""
    W, H, fps, _ = probe(path)
    s0 = max(0, start - warm)
    t0 = time.perf_counter()
    eng.reset(W, H, fps, collect_from=start)
    dq = queue.Queue(maxsize=64)                                      # decode in a thread, overlapped with the GPU

    def feed():
        for fr in decode_range(path, s0, end - s0, W, H, fps):
            dq.put(fr)
        dq.put(None)
    threading.Thread(target=feed, daemon=True).start()
    batch, f_base, n_dec = [], s0, 0
    while (fr := dq.get()) is not None:
        if keep_frames is not None and s0 + n_dec + len(batch) >= start:
            keep_frames.append(fr)
        batch.append(fr)
        if len(batch) == eng.a.batch:
            eng.run_batch(batch, f_base)
            f_base += len(batch)
            n_dec += len(batch)
            batch = []
    if batch:
        eng.run_batch(batch, f_base)
        n_dec += len(batch)
    track_s = time.perf_counter() - t0
    t1 = time.perf_counter()                       # every candidate view; the coordinator picks K after pooling
    keys, crops = [], []
    for tid, wins in eng.cands.items():
        for q_, f, c in wins.values():
            keys.append((tid, f, q_))
            crops.append(c)
    probs = score_views(eng.clf, crops)
    views = defaultdict(list)
    for (tid, f, q_), p in zip(keys, probs):
        views[tid].append([f, round(q_, 3)] + [round(float(x), 5) for x in p])
    score_s = time.perf_counter() - t1
    tracks = {}
    for tid, recs in eng.store.items():
        head = {f: rle(eng.grid_mask(recs[f])) for f in range(s0, start) if f in recs}
        tail = {f: rle(eng.grid_mask(recs[f])) for f in range(max(start, end - warm), end) if f in recs}
        own = sum(1 for f in recs if f >= start)
        if own or head:
            tracks[str(tid)] = {"views": views.get(tid, []), "head": head, "tail": tail, "own_frames": own}
    return {"frames": end - start, "decoded": n_dec, "start": start, "end": end, "W": W, "H": H, "fps": fps,
            "track_s": round(track_s, 3), "score_s": round(score_s, 3), "grid": eng.G, "tracks": tracks,
            "masks": pack_masks(eng.store, start, end) if pack else None,
            "stage_ms_per_frame": {k: round(1000 * v / max(1, n_dec), 2) for k, v in eng.T.t.items()}}


def pack_masks(store, start, end):
    """Own-frame masks of a chunk as compressed bytes: {tid: {f: (region, float16 logits crop)}}."""
    out = {tid: {f: (tuple(rec[4:8]), rec[8].cpu().numpy()) for f, rec in recs.items() if start <= f < end}
           for tid, recs in store.items()}
    return zlib.compress(pickle.dumps(out, protocol=5), 1)


def unpack_masks(blob, torch, dev):
    store = defaultdict(dict)
    for tid, recs in pickle.loads(zlib.decompress(blob)).items():
        for f, (region, crop) in recs.items():
            store[tid][f] = (*region, *region, torch.from_numpy(crop).to(dev))
    return store


def mask_iou_over(fa, fb, G):
    """IoU of two tracks over the frames either appears on: sum of intersections / sum of unions (reference for
    boundary_iou, which computes all pairs at once)."""
    from pycocotools import mask as mu
    inter = union = 0.0
    for f in set(fa) | set(fb):
        ra = {"size": [G, G], "counts": fa[f].encode()} if f in fa else None
        rb = {"size": [G, G], "counts": fb[f].encode()} if f in fb else None
        a_ = float(mu.area(ra)) if ra else 0.0
        b_ = float(mu.area(rb)) if rb else 0.0
        i_ = float(mu.area(mu.merge([ra, rb], intersect=True))) if ra and rb else 0.0
        inter += i_
        union += a_ + b_ - i_
    return inter / union if union else 0.0


def boundary_iou(cur, prev, G):
    """IoU between every track of a chunk (its warm-up `head` masks) and every track of the previous chunk (its `tail`
    masks), accumulated over the boundary frames: sum of intersections / sum of unions (a frame where only one of
    the two has a mask adds that mask to the union). One vectorised C call per frame."""
    from pycocotools import mask as mu
    ct, pt = list(cur), list(prev)
    inter = np.zeros((len(ct), len(pt)))
    union = np.zeros((len(ct), len(pt)))
    frames = {f for t in ct for f in cur[t]["head"]} | {f for u in pt for f in prev[u]["tail"]}
    for f in frames:
        ci = [i for i, t in enumerate(ct) if f in cur[t]["head"]]
        pi = [j for j, u in enumerate(pt) if f in prev[u]["tail"]]
        ra = [{"size": [G, G], "counts": cur[ct[i]]["head"][f].encode()} for i in ci]
        rb = [{"size": [G, G], "counts": prev[pt[j]]["tail"][f].encode()} for j in pi]
        A, B = np.zeros(len(ct)), np.zeros(len(pt))
        if ra:
            A[ci] = mu.area(ra)
        if rb:
            B[pi] = mu.area(rb)
        I = np.zeros((len(ct), len(pt)))
        if ra and rb:
            iou = np.asarray(mu.iou(ra, rb, [0] * len(rb))).reshape(len(ra), len(rb))
            I[np.ix_(ci, pi)] = iou * (A[ci][:, None] + B[pi][None]) / (1 + iou)   # intersection from IoU + areas
        inter += I
        union += A[:, None] + B[None] - I
    return ct, pt, np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def stitch_vote(results, stitch_iou=0.5, k=10, min_gap=5, blur_min=0.25):
    """Step 2. results: chunk results in time order. Returns (women tids per chunk, labels per person, stats,
    person id of every chunk track: [{tid: person}] per chunk)."""
    N = len(results)
    gid_of, next_gid, links = {}, 0, 0                   # (chunk, local tid) -> person id
    for i in range(N):
        tr = results[i]["tracks"]
        if i > 0:
            ct, pt, iou = boundary_iou({t: v for t, v in tr.items() if v["head"]},
                                       {u: v for u, v in results[i - 1]["tracks"].items() if v["tail"]},
                                       results[i]["grid"])
            taken, used_u = set(), set()
            for a_, b_ in sorted(zip(*np.nonzero(iou >= stitch_iou)), key=lambda ab: -iou[ab]):
                t, u = ct[a_], pt[b_]
                if t not in taken and u not in used_u and (i - 1, u) in gid_of:
                    gid_of[(i, t)] = gid_of[(i - 1, u)]
                    taken.add(t)
                    used_u.add(u)
                    links += 1
        for t, v in tr.items():
            if (i, t) not in gid_of and v["own_frames"]:
                gid_of[(i, t)] = next_gid
                next_gid += 1
    pooled = defaultdict(dict)                           # person -> {window: (q, f, probs)}
    for i in range(N):
        for t, v in results[i]["tracks"].items():
            g = gid_of.get((i, t))
            if g is None:
                continue
            for row in v["views"]:
                f, q_, p = row[0], row[1], row[2:]
                wb = f // min_gap
                if wb not in pooled[g] or q_ > pooled[g][wb][0]:
                    pooled[g][wb] = (q_, f, p)
    labels = {}
    for g, wins in pooled.items():
        top = dict(sorted(wins.items(), key=lambda kv: -kv[1][0])[:2 * k])   # the 2K best windows, as one process keeps
        labels[g] = label_from_probs([np.asarray(p) for f, p in select_views(top, k, min_gap)], blur_min)
    women_g = {g for g, v in labels.items() if v["label"] == "woman"}
    women = [[int(t) for (c, t), g in gid_of.items() if c == i and g in women_g] for i in range(N)]
    persons = [{t: g for (c, t), g in gid_of.items() if c == i} for i in range(N)]
    return women, labels, {"people": len(labels), "women": len(women_g), "stitch_links": links}, persons


def blur_chunk(eng, path, res, women, dump_blur=False):
    """Step 3 on any GPU: re-decode the chunk's own frames, pixelate its women, return the encoded mp4 bytes."""
    torch = eng.torch
    t0 = time.perf_counter()
    start, end, W, H, fps = res["start"], res["end"], res["W"], res["H"], res["fps"]
    eng.reset(W, H, fps)
    eng.store = unpack_masks(res["masks"], torch, eng.dev)
    frames = list(decode_range(path, start, end - start, W, H, fps))
    union = {} if dump_blur else None
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "seg.mp4"
        eng.write_blurred(frames, start, set(women), str(out), union_rle=union)
        data = out.read_bytes()
    return {"blur_s": round(time.perf_counter() - t0, 3), "mp4": data, "union": union}


def blur_frames(eng, frames, start, women, dump_blur=False):
    """Step 3 in the container that tracked the chunk: its own frames and masks are still in memory."""
    t0 = time.perf_counter()
    union = {} if dump_blur else None
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "seg.mp4"
        eng.write_blurred(frames, start, set(women), str(out), union_rle=union)
        data = out.read_bytes()
    return {"blur_s": round(time.perf_counter() - t0, 3), "mp4": data, "union": union}


def render_ids(eng, frames, start, persons, labels, path, title):
    """The overlay video of a chunk's own frames: mask colour = person id, box/tag = gender '#id label P(female)'."""
    from fast_blur import render_debug
    store = defaultdict(dict)
    for tid, recs in eng.store.items():
        g = persons.get(str(tid), persons.get(tid))
        if g is None:
            continue
        for f, rec in recs.items():
            if f >= start:
                store[int(g)][f] = rec
    render_debug(path, frames, store, {int(k): v for k, v in labels.items()}, eng.W, eng.H, eng.fps, eng.F, eng.torch,
                 title, f0=start)


def join_segments(segments, out):
    """Step 4: concatenate mp4 segments (same encoder settings) without re-encoding."""
    with tempfile.TemporaryDirectory() as td:
        lst = Path(td) / "list.txt"
        with open(lst, "w") as fh:
            for i, data in enumerate(segments):
                p = Path(td) / f"seg_{i:04d}.mp4"
                p.write_bytes(data)
                fh.write(f"file '{p}'\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(out)], check=True)


def write_union(path, unions):
    with open(path, "w") as fh:
        for u in unions:
            for f, r in sorted((u or {}).items(), key=lambda kv: int(kv[0])):
                fh.write(json.dumps({"f": int(f), "tid": -1, "prompt": "woman", "rle": r}) + "\n")


# ------------------------------------------------------------------ HTTP launcher (your own GPUs)

CACHE = Path("/root/dist_cache")


def worker(a):
    from fast_blur_edgetam import EdgeTAMEngine
    eng = EdgeTAMEngine(a)
    torch = eng.torch
    gpu = torch.cuda.get_device_name(0)
    lock = threading.Lock()
    CACHE.mkdir(parents=True, exist_ok=True)
    print(f"[worker] {gpu} ready on :{a.port}, models loaded in {eng.load_s:.1f} s", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                return self.reply(200, {"gpu": gpu, "load_s": round(eng.load_s, 1)})
            if self.path.startswith("/video/"):
                ok = (CACHE / f"{self.path.split('/')[-1]}.mp4").exists()
                return self.reply(200 if ok else 404, {"have": ok})
            self.reply(404, {})

        def do_PUT(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            (CACHE / f"{self.path.split('/')[-1]}.mp4").write_bytes(data)
            self.reply(200, {"bytes": len(data)})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            path = CACHE / f"{req['sha']}.mp4"
            try:
                with lock, torch.inference_mode():
                    if self.path == "/track":
                        res = track_chunk(eng, path, req["start"], req["end"], req["warm"])
                        res["masks"] = base64.b64encode(res["masks"]).decode()
                        res["gpu"] = gpu
                    else:
                        r = dict(req["res"], masks=base64.b64decode(req["res"]["masks"]))
                        res = blur_chunk(eng, path, r, req["women"], req.get("dump_blur", False))
                        res["mp4"] = base64.b64encode(res["mp4"]).decode()
                self.reply(200, res)
            except Exception as e:                                            # report, keep serving
                import traceback
                traceback.print_exc()
                self.reply(500, {"error": repr(e)})

    HTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


def call(url, path, obj=None, data=None, method=None, timeout=3600):
    body = data if data is not None else (json.dumps(obj).encode() if obj is not None else None)
    req = urllib.request.Request(url + path, data=body, method=method or ("POST" if body is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise RuntimeError(f"{url}{path}: {e.code} {e.read()[:500]!r}")


def run(a):
    W, H, fps, n = probe(a.video)
    workers = [w.rstrip("/") for w in a.workers.split(",")]
    N = a.chunks or len(workers)
    bounds = chunk_bounds(n, N)
    data = Path(a.video).read_bytes()
    sha = hashlib.sha1(data).hexdigest()[:16]
    for w in workers:
        if call(w, f"/video/{sha}") is None:
            call(w, f"/video/{sha}", data=data, method="PUT")
    t0 = time.perf_counter()
    results = [None] * N
    with ThreadPoolExecutor(len(workers)) as ex:
        for i, r in zip(range(N), ex.map(lambda i: call(workers[i % len(workers)], "/track", {
                "sha": sha, "start": bounds[i][0], "end": bounds[i][1], "warm": a.warm}), range(N))):
            results[i] = r
    t1 = time.perf_counter()
    women, labels, st, _ = stitch_vote(results, a.stitch_iou, a.k, a.min_gap, a.blur_min)
    t2 = time.perf_counter()
    with ThreadPoolExecutor(len(workers)) as ex:
        segs = list(ex.map(lambda i: call(workers[i % len(workers)], "/blur", {
            "sha": sha, "res": results[i], "women": women[i], "dump_blur": bool(a.dump_blur)}), range(N)))
    t3 = time.perf_counter()
    join_segments([base64.b64decode(s["mp4"]) for s in segs], a.out)
    total = time.perf_counter() - t0
    if a.dump_blur:
        write_union(a.dump_blur, [s["union"] for s in segs])
    meta = {"frames": n, "chunks": N, "workers": len(workers), "warm": a.warm, **st,
            "track_wall_s": round(t1 - t0, 2), "vote_s": round(t2 - t1, 3), "blur_wall_s": round(t3 - t2, 2),
            "total_s": round(total, 2), "end_to_end_hz": round(n / total, 1)}
    print(json.dumps(meta, indent=1), flush=True)


def main():
    from fast_blur_edgetam import add_args
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--port", type=int, default=8100)
    add_args(w)
    r = sub.add_parser("run")
    r.add_argument("--video", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--workers", required=True, help="comma-separated worker URLs")
    r.add_argument("--chunks", type=int, default=None, help="default: one per worker")
    r.add_argument("--warm", type=int, default=16, help="frames a chunk tracks before its own first frame")
    r.add_argument("--stitch-iou", type=float, default=0.5)
    r.add_argument("--k", type=int, default=10)
    r.add_argument("--min-gap", type=int, default=5)
    r.add_argument("--blur-min", type=float, default=0.25)
    r.add_argument("--dump-blur", default=None, help="write the union of blurred pixels per frame (jsonl)")
    a = ap.parse_args()
    worker(a) if a.cmd == "worker" else run(a)


if __name__ == "__main__":
    main()
