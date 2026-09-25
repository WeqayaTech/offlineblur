"""RF-DETR + EdgeTAM + PE-Core person blur on Modal: one video split in time over many GPUs (dist_edgetam.py steps).

Every chunk runs in its own GPU container with the models already loaded (`Worker`); a CPU function coordinates
(`demo`): track all chunks in parallel -> stitch people across chunks + pooled K-view vote -> blur all chunks in
parallel -> join. Videos travel through a Modal Volume (uploaded once; each GPU decodes only its own frame range),
so the same code scales from a 12 s clip to long videos and from 1 to `max_containers` GPUs.

    modal run tracking/modal_edgetam.py --video clip.mp4 --chunks 8 [--loops 4] [--reference]
    EDGETAM_GPU=B200 modal run tracking/modal_edgetam.py --video clip.mp4 --chunks 8

--reference also runs the same video in ONE container (the single-process result) and compares the two blurs pixel by
pixel (as blur_agreement.py). Timing excludes container start and model load (reported separately): a warm-up call
first brings up `chunks` containers, as a deployed service keeps them (`modal deploy` + min_containers).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import modal

GPU = os.environ.get("EDGETAM_GPU", "H100")
HERE = Path(__file__).resolve().parent
EDGETAM_COMMIT = "7711e01"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("rfdetr==1.10.1", "trackers==2.6.0", "open_clip_torch==3.3.0", "pycocotools",
                 "opencv-python-headless", "numpy<2.3", "timm", "psutil", "hydra-core>=1.3.2", "iopath>=0.1.10")
    .run_commands(
        "git clone https://github.com/facebookresearch/EdgeTAM.git /opt/EdgeTAM",
        f"cd /opt/EdgeTAM && git checkout {EDGETAM_COMMIT}",
        "cd /opt/EdgeTAM && SAM2_BUILD_CUDA=0 pip install --no-deps -e .",
        # upstream bug: the memory encoder's perceiver breaks with >1 object per batch; .reshape = same result
        "sed -i 's/expand(B, -1, -1).view(/expand(B, -1, -1).reshape(/' /opt/EdgeTAM/sam2/modeling/perceiver.py",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_DISABLE_XET": "1", "PYTHONPATH": "/root/tracking"})
    .add_local_file(HERE / "fast_blur.py", "/root/tracking/fast_blur.py")
    .add_local_file(HERE / "fast_blur_edgetam.py", "/root/tracking/fast_blur_edgetam.py")
    .add_local_file(HERE / "dist_edgetam.py", "/root/tracking/dist_edgetam.py")
    .add_local_file(HERE / "blur_agreement.py", "/root/tracking/blur_agreement.py")
)
cache = modal.Volume.from_name("offlineblur-model-cache", create_if_missing=True)   # model weights, downloaded once
videos = modal.Volume.from_name("offlineblur-videos", create_if_missing=True)       # inputs in/, outputs out/
app = modal.App("offlineblur-edgetam", image=image)


def board_id(board) -> str:
    return board.get("job")


def _local(key: str) -> str:
    """Path of a video on the videos volume, reloading once if another container committed it after we started."""
    p = Path("/videos") / key
    if not p.exists():
        videos.reload()
    if not p.exists():
        raise FileNotFoundError(f"{key} not on the offlineblur-videos volume")
    return str(p)


@app.cls(gpu=GPU, cpu=8.0, memory=32768, volumes={"/cache": cache, "/videos": videos}, timeout=3600,
         scaledown_window=300, max_containers=64)   # EdgeTAM's per-person loop is CPU-bound: reserve real cores
class Worker:
    @modal.enter()
    def load(self):
        import argparse
        import torch
        from fast_blur_edgetam import EdgeTAMEngine, add_args
        os.makedirs("/cache/rfdetr", exist_ok=True)
        os.chdir("/cache/rfdetr")                          # rfdetr downloads its weights into the working directory
        ap = argparse.ArgumentParser()
        add_args(ap)
        self.eng = EdgeTAMEngine(ap.parse_args(["--edgetam-ckpt", "/opt/EdgeTAM/checkpoints/edgetam.pt"]))
        self.gpu = torch.cuda.get_device_name(0)
        try:
            cache.commit()
        except Exception:                                  # another container committed the same weights
            pass

    @modal.method()
    def ping(self, hold_s: float = 0.0) -> dict:
        time.sleep(hold_s)                                 # holding the call makes Modal start one container per call
        return {"gpu": self.gpu, "load_s": round(self.eng.load_s, 1), "container": os.environ.get("MODAL_TASK_ID")}

    @modal.method()
    def prime(self, key: str, hold_s: float = 20.0) -> dict:
        """Warm-up: run the real pipeline on the first frames (GPU kernels and autotuning happen on first use), then
        hold so that one call occupies one container."""
        from dist_edgetam import track_chunk
        t0 = time.perf_counter()
        with self.eng.torch.inference_mode():
            track_chunk(self.eng, _local(key), 0, 32, 0, pack=False)
        time.sleep(max(0.0, hold_s - (time.perf_counter() - t0)))
        return {"gpu": self.gpu, "load_s": round(self.eng.load_s, 1), "container": os.environ.get("MODAL_TASK_ID")}

    @modal.method()
    def process(self, key: str, i: int, start: int, end: int, warm: int, board: modal.Dict,
                dump_blur: bool = False, overlay: bool = False) -> dict:
        """Track chunk i, post its per-person votes to `board`, wait for the coordinator's decision, then blur the
        frames still in memory. One call per chunk: no re-decode, no mask transfer, no second scheduling round."""
        from dist_edgetam import blur_frames, track_chunk
        frames = []
        with self.eng.torch.inference_mode():
            res = track_chunk(self.eng, _local(key), start, end, warm, keep_frames=frames, pack=False)
            res["gpu"], res["container"] = self.gpu, os.environ.get("MODAL_TASK_ID")
            board[f"res{i}"] = res
            t0 = time.perf_counter()
            while (women := board.get(f"women{i}")) is None:
                if time.perf_counter() - t0 > 900:
                    raise TimeoutError("no decision from the coordinator")
                time.sleep(0.02)
            wait_s = time.perf_counter() - t0
            seg = blur_frames(self.eng, frames, start, women, dump_blur)
        job = Path("/videos/jobs") / board_id(board)
        job.mkdir(parents=True, exist_ok=True)
        (job / f"seg_{i:04d}.mp4").write_bytes(seg.pop("mp4"))
        if seg.get("union") is not None:
            import json
            (job / f"union_{i:04d}.json").write_text(json.dumps(seg.pop("union")))
        videos.commit()
        seg.pop("union", None)
        seg["wait_s"] = round(wait_s, 3)
        board[f"done{i}"] = True                           # the timed part ends here
        if overlay:                                        # untimed: masks + person ids + gender, this chunk's frames
            from dist_edgetam import render_ids
            ids = board.get(f"ids{i}")
            render_ids(self.eng, frames, start, ids["persons"], ids["labels"], str(job / f"ids_{i:04d}.mp4"),
                       ids["title"])
            videos.commit()
        return seg

    @modal.method()
    def render_single(self, key: str, title: str) -> dict:
        """The whole video in this one container (timed: track + vote + blur), then its overlay video (untimed)."""
        from dist_edgetam import blur_frames, probe, render_ids, stitch_vote, track_chunk
        n = probe(_local(key))[3]
        frames = []
        t0 = time.perf_counter()
        with self.eng.torch.inference_mode():
            r = track_chunk(self.eng, _local(key), 0, n, 0, keep_frames=frames, pack=False)
            women, labels, st, persons = stitch_vote([r])
            seg = blur_frames(self.eng, frames, 0, women[0])
            total = time.perf_counter() - t0
            base = Path("/videos/out") / key.split("/")[-1].replace(".mp4", f"_1x{GPU}")
            base.parent.mkdir(parents=True, exist_ok=True)
            Path(f"{base}_blur.mp4").write_bytes(seg["mp4"])
            render_ids(self.eng, frames, 0, persons[0], labels, f"{base}_ids.mp4", title)
        videos.commit()
        return {**st, "frames": n, "total_s": round(total, 2), "end_to_end_hz": round(n / total, 1),
                "track_s": r["track_s"], "blur": f"{base}_blur.mp4", "ids": f"{base}_ids.mp4"}

    @modal.method()
    def track(self, key: str, start: int, end: int, warm: int, batch_people: bool = True) -> dict:
        from dist_edgetam import track_chunk
        self.eng.a.no_batch_people = not batch_people
        with self.eng.torch.inference_mode():
            res = track_chunk(self.eng, _local(key), start, end, warm)
        res["gpu"], res["container"] = self.gpu, os.environ.get("MODAL_TASK_ID")
        return res

    @modal.method()
    def blur(self, key: str, res: dict, women: list, dump_blur: bool = False) -> dict:
        from dist_edgetam import blur_chunk
        with self.eng.torch.inference_mode():
            return blur_chunk(self.eng, _local(key), res, women, dump_blur)


@app.function(volumes={"/videos": videos}, timeout=3600, cpu=4)
def demo(key: str, chunks: int, warm: int = 16, reference: bool = False, loops: int = 1, clip_start: float = 0.0,
         clip_seconds: float = 0.0, overlay: bool = False) -> dict:
    import json
    import subprocess
    import uuid
    from concurrent.futures import ThreadPoolExecutor
    from dist_edgetam import boundary_iou, chunk_bounds, join_segments, mask_iou_over, probe, stitch_vote
    src = _local(key)
    if clip_seconds > 0:                                   # a continuous stretch of a long video (stream copy)
        key = key.replace(".mp4", f"_s{clip_start:g}_t{clip_seconds:g}.mp4")
        if not Path("/videos", key).exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{clip_start}", "-i", src, "-t",
                            f"{clip_seconds}", "-c", "copy", "-avoid_negative_ts", "make_zero", "-an",
                            f"/videos/{key}"], check=True)
            videos.commit()
        src = _local(key)
    if loops > 1:                                          # the clip back to back, as a longer test video
        key = key.replace(".mp4", f"_x{loops}.mp4")
        if not Path("/videos", key).exists():
            lst = Path("/tmp/loop.txt")
            lst.write_text("".join(f"file '{src}'\n" for _ in range(loops)))
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", f"/videos/{key}"], check=True)
            videos.commit()
    W, H, fps, n = probe(_local(key))
    bounds = chunk_bounds(n, chunks)
    w = Worker()

    t = time.perf_counter()                                # warm, primed containers (untimed, as a service)
    pings = list(w.prime.map([key] * chunks))
    warmup_s = time.perf_counter() - t
    avail = len({p["container"] for p in pings})
    if avail < chunks:                                     # every chunk must run at once (they wait for each other)
        print(f"[demo] only {avail} containers came up (GPU concurrency limit?): using {avail} chunks")
        chunks = avail
        bounds = chunk_bounds(n, chunks)

    t0 = time.perf_counter()
    job = uuid.uuid4().hex[:12]
    with modal.Dict.ephemeral() as board, ThreadPoolExecutor(max(1, chunks)) as ex:
        board["job"] = job
        calls = list(ex.map(lambda ib: w.process.spawn(key, ib[0], ib[1][0], ib[1][1], warm, board, reference,
                                                       overlay), enumerate(bounds)))
        results = [None] * chunks
        while any(r is None for r in results):
            for i in range(chunks):
                if results[i] is None:
                    results[i] = board.get(f"res{i}")
            time.sleep(0.02)
        t1 = time.perf_counter()
        women, labels, st, persons = stitch_vote(results)
        title = f"{chunks}x {GPU} split"
        for i in range(chunks):
            if overlay:
                gids = set(persons[i].values())
                board[f"ids{i}"] = {"persons": persons[i], "title": title,
                                    "labels": {g: v for g, v in labels.items() if g in gids}}
            board[f"women{i}"] = women[i]
        t2 = time.perf_counter()
        while not all(board.get(f"done{i}") for i in range(chunks)):
            time.sleep(0.02)
        t3 = time.perf_counter()                           # timed part done (overlays are drawn after this)
        segs = list(ex.map(lambda c: c.get(), calls))
    videos.reload()
    jd = Path("/videos/jobs") / job
    out = Path("/videos/out") / key.split("/")[-1].replace(".mp4", f"_edgetam_{chunks}x{GPU}_w{warm}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    tj = time.perf_counter()
    join_segments([(jd / f"seg_{i:04d}.mp4").read_bytes() for i in range(chunks)], out)
    join_s = time.perf_counter() - tj
    total = (t3 - t0) + join_s                             # timed: track -> vote -> blur -> join (no overlays)
    videos.commit()
    check = 0.0                                            # vectorised boundary IoU == the pair-by-pair reference
    for i in range(1, chunks):
        cur = {t: v for t, v in results[i]["tracks"].items() if v["head"]}
        prev = {u: v for u, v in results[i - 1]["tracks"].items() if v["tail"]}
        ct, pt, iou = boundary_iou(cur, prev, results[i]["grid"])
        for a_, t in enumerate(ct[:8]):
            for b_, u in enumerate(pt[:8]):
                check = max(check, abs(iou[a_, b_] - mask_iou_over(cur[t]["head"], prev[u]["tail"], results[i]["grid"])))
    meta = {"frames": n, "size": [W, H], "fps": round(fps, 3), "chunks": chunks, "warm": warm, "gpu": GPU,
            "video": key, "stitch_check_maxdiff": round(float(check), 6),
            "gpus_seen": sorted({p["gpu"] for p in pings}), "containers": len({r["container"] for r in results}),
            **st, "warmup_s": round(warmup_s, 1), "model_load_s": max(p["load_s"] for p in pings),
            "track_wall_s": round(t1 - t0, 2), "vote_s": round(t2 - t1, 3), "blur_wall_s": round(t3 - t2, 2),
            "join_s": round(join_s, 2), "total_s": round(total, 2), "end_to_end_hz": round(n / total, 1),
            "output": f"offlineblur-videos:/{out.relative_to('/videos')}",
            "chunks_detail": [{"frames": r["frames"], "decoded": r["decoded"], "track_s": r["track_s"],
                               "score_s": r["score_s"], "blur_s": s["blur_s"], "wait_s": s["wait_s"],
                               "chunk_hz": round(r["decoded"] / r["track_s"], 1),
                               "stage_ms_per_frame": r["stage_ms_per_frame"]} for r, s in zip(results, segs)]}
    print(json.dumps({k: v for k, v in meta.items() if k != "chunks_detail"}), flush=True)
    compare = None
    if overlay:                                            # untimed: single-GPU run + side-by-side comparisons
        ids_split = out.with_name(out.stem + "_ids.mp4")
        join_segments([(jd / f"ids_{i:04d}.mp4").read_bytes() for i in range(chunks)], ids_split)
        single = w.render_single.remote(key, f"1x {GPU}")
        videos.reload()
        a_lab = f"1x {GPU} - {single['end_to_end_hz']} Hz - {single['total_s']} s"      # no ':' (drawtext syntax)
        b_lab = f"{chunks}x {GPU} split - {meta['end_to_end_hz']} Hz - {meta['total_s']} s"
        txt = "drawtext=text='{}':x=12:y=h-44:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.75:boxborderw=8"
        cmp_ids = out.with_name(out.stem + "_compare_ids.mp4")
        cmp_blur = out.with_name(out.stem + "_compare_blur.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", single["ids"], "-i", str(ids_split),
                        "-filter_complex", f"[0]scale=960:-2,{txt.format(a_lab)}[a];[1]scale=960:-2,{txt.format(b_lab)}[b];"
                        "[a][b]hstack=2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
                        str(cmp_ids)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-i", single["blur"], "-i", str(out),
                        "-filter_complex", f"[0]scale=640:-2,{txt.format('original')}[o];"
                        f"[1]scale=640:-2,{txt.format(a_lab)}[a];[2]scale=640:-2,{txt.format(b_lab)}[b];"
                        "[o][a][b]hstack=3", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt",
                        "yuv420p", str(cmp_blur)], check=True)
        videos.commit()
        compare = {"single": {k: v for k, v in single.items() if k not in ("blur", "ids")},
                   "ids_mp4": cmp_ids.read_bytes(), "blur_mp4": cmp_blur.read_bytes()}
    ref_meta = agreement = None
    if reference:                                          # the same video in ONE container = single-process result
        from blur_agreement import compare_unions
        cached = Path("/videos/ref") / (key.split("/")[-1] + f".{GPU}.json")
        if cached.exists():
            ref = json.loads(cached.read_text())
        else:
            t4 = time.perf_counter()
            r = w.track.remote(key, 0, n, 0)
            rw, _, rst, _ = stitch_vote([r])
            rs = w.blur.remote(key, r, rw[0], True)
            ref_s = time.perf_counter() - t4
            ref = {"meta": {**rst, "total_s": round(ref_s, 2), "end_to_end_hz": round(n / ref_s, 1),
                            "chunk_hz": round(r["decoded"] / r["track_s"], 1)}, "union": rs["union"]}
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(ref))
            videos.commit()
        ref_meta = ref["meta"]
        merged = {}
        for i in range(chunks):
            merged.update({int(f): v for f, v in json.loads((jd / f"union_{i:04d}.json").read_text()).items()})
        agreement = compare_unions({int(f): v for f, v in ref["union"].items()}, merged)
    return {"meta": meta, "reference": ref_meta, "agreement": agreement, "mp4": out.read_bytes(), "compare": compare,
            "labels": {str(k): v for k, v in labels.items()}}


@app.function(volumes={"/videos": videos}, timeout=3600, cpu=4)
def check_batching(key: str) -> dict:
    """The whole video in one container twice: EdgeTAM one person per call vs people batched. Speed and blur agreement."""
    from blur_agreement import compare_unions
    from dist_edgetam import probe, stitch_vote
    n = probe(_local(key))[3]
    w, out = Worker(), {}
    for mode in (False, True):
        r = w.track.remote(key, 0, n, 0, mode)
        women, labels, st, _ = stitch_vote([r])
        b = w.blur.remote(key, r, women[0], True)
        out["batched" if mode else "per_person"] = {**st, "track_s": r["track_s"], "chunk_hz": round(n / r["track_s"], 1),
                                                    "stage_ms_per_frame": r["stage_ms_per_frame"], "union": b["union"]}
    agree = compare_unions({int(k): v for k, v in out["per_person"].pop("union").items()},
                           {int(k): v for k, v in out["batched"].pop("union").items()})
    return {**out, "agreement": agree}


@app.local_entrypoint()
def main(video: str, chunks: int = 8, warm: int = 16, reference: bool = False, loops: int = 1, out: str = "",
         check: bool = False, clip_start: float = 0.0, clip_seconds: float = 0.0, overlay: bool = False):
    import hashlib
    import json
    data = Path(video).read_bytes()
    key = f"in/{hashlib.sha1(data).hexdigest()[:16]}.mp4"
    try:
        with videos.batch_upload() as up:                  # once per video; later runs reuse it
            up.put_file(video, "/" + key)
    except FileExistsError:
        pass
    if check:
        print(json.dumps(check_batching.remote(key), indent=1))
        return
    r = demo.remote(key, chunks, warm, reference, loops, clip_start, clip_seconds, overlay)
    out = out or str(Path(video).with_name(Path(video).stem + f"_x{loops}_edgetam_{chunks}x{GPU}.mp4"))
    Path(out).write_bytes(r["mp4"])
    Path(out).with_suffix(".json").write_text(json.dumps({k: r[k] for k in ("meta", "reference", "agreement", "labels")},
                                                         indent=1))
    m = r["meta"]
    print(json.dumps({k: v for k, v in m.items() if k != "chunks_detail"}, indent=1))
    for i, c in enumerate(m["chunks_detail"]):
        print(f"  chunk {i:>2}: {c['decoded']:>4} frames, track {c['track_s']:6.2f} s ({c['chunk_hz']} Hz), "
              f"score {c['score_s']:.2f} s, blur {c['blur_s']:.2f} s")
    if r["reference"]:
        print("single container:", json.dumps(r["reference"]))
        print("blur agreement (single vs split):", json.dumps(r["agreement"]))
    if r.get("compare"):
        c = r["compare"]
        for name in ("ids", "blur"):
            p_ = Path(out).with_name(Path(out).stem + f"_compare_{name}.mp4")
            p_.write_bytes(c[f"{name}_mp4"])
            print(f"-> {p_}")
        print("single container:", json.dumps(c["single"]))
    print(f"-> {out}")


@app.function(volumes={"/videos": videos}, timeout=3600, cpu=8)
def side_by_side(key: str, split_out: str, split_label: str) -> bytes:
    """original | single container | split, for visual comparison. The single-container blur is rendered here (one
    GPU, the whole video as one chunk) and kept next to the split output on the volume."""
    import subprocess
    from dist_edgetam import probe, stitch_vote
    n = probe(_local(key))[3]
    single = Path("/videos/out") / key.split("/")[-1].replace(".mp4", f"_edgetam_1x{GPU}_single.mp4")
    if not single.exists():
        w = Worker()
        r = w.track.remote(key, 0, n, 0)
        women, _, _, _ = stitch_vote([r])
        single.write_bytes(w.blur.remote(key, r, women[0], False)["mp4"])
        videos.commit()
    out = Path("/tmp/sbs.mp4")
    lab = "drawtext=text='{}':x=12:y=12:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.7:boxborderw=6"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", _local(key), "-i", str(single), "-i", "/videos/" + split_out,
                    "-filter_complex",
                    f"[0]scale=640:-2,{lab.format('original')}[a];[1]scale=640:-2,{lab.format('single GPU')}[b];"
                    f"[2]scale=640:-2,{lab.format(split_label)}[c];[a][b][c]hstack=3",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out.read_bytes()


@app.local_entrypoint()
def sbs(key: str, split_out: str, label: str, out: str):
    Path(out).write_bytes(side_by_side.remote(key, split_out, label))
    print(f"-> {out}")
