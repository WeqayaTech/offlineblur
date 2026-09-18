#!/usr/bin/env python3
"""Phase 2 — end-to-end transformer tracking with track queries.

Two adapters, both driving the official research repos exactly the way their own README does
(subprocess on a MOT-layout jpg sequence, MOT text out), so the tracker code is never forked:

  memotr   MCG-NJU/MeMOTR — track queries + long-term memory; miss tolerance keeps an occluded query
           alive for N frames (default 300 = 10 s at 30 fps). Checkpoints: MOT17 (street pedestrians,
           default) or DanceTrack. Output: <submit_dir>/<split>/tracker/<seq>.txt
  motrv2   megvii-research/MOTRv2 — MOTR with detector proposals as anchor queries. Proposals here come
           from a YOLO person detector (det_db json in the repo's format). Output: <out>/tracker/<seq>.txt

Both write x1,y1,w,h in source pixels, frame numbers 1-based. Result is normalised to tracks.jsonl:
    {"f": frame_index, "tid": track_id, "box": [x1,y1,x2,y2], "score": 1.0}

Note on resolution: both trackers resample the frame to 800 (short side) / 1536 (long side) internally —
that is the resolution their queries were trained at. The dense giant backbone of phase 1/3 does not
have that limit; if 4K background people are missed by the tracker, that is where the miss happens.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from common import write_json


def _symlink(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(Path(src).resolve())


def mot_txt_to_jsonl(txt_path, out_path, w=None, h=None) -> int:
    n = 0
    with open(txt_path) as fh, open(out_path, "w") as out:
        for line in fh:
            p = line.strip().split(",")
            if len(p) < 6:
                continue
            f = int(float(p[0])) - 1
            tid = int(float(p[1]))
            x, y, bw, bh = map(float, p[2:6])
            box = [x, y, x + bw, y + bh]
            if w and h:
                box = [max(0.0, box[0]), max(0.0, box[1]), min(float(w), box[2]), min(float(h), box[3])]
            if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                continue
            out.write(json.dumps({"f": f, "tid": tid, "box": [round(v, 1) for v in box], "score": 1.0}) + "\n")
            n += 1
    return n


def _run(cmd, cwd, log_path, env=None):
    print("[tracker] $", " ".join(map(str, cmd)))
    with open(log_path, "w") as log:
        p = subprocess.run(list(map(str, cmd)), cwd=str(cwd), stdout=log, stderr=subprocess.STDOUT, env=env)
    if p.returncode != 0:
        tail = Path(log_path).read_text().splitlines()[-40:]
        raise SystemExit(f"tracker failed (exit {p.returncode}); last lines of {log_path}:\n" + "\n".join(tail))


# ------------------------------------------------------------------------------------------------
def run_memotr(frames_meta: dict, out_dir, repo, ckpt, train_config=None, miss_tolerance=300, det_thresh=0.5,
               track_thresh=0.5, result_thresh=0.5, use_motion=True, gpu="0") -> Path:
    repo, out_dir = Path(repo), Path(out_dir)
    seq_dir, seq = Path(frames_meta["seq_dir"]), frames_meta["seq_name"]
    work = out_dir / "memotr_work"
    train_config = Path(train_config) if train_config else repo / "configs" / "train_mot17.yaml"
    cfg = yaml.safe_load(Path(train_config).read_text())
    dataset = cfg.get("DATASET", "MOT17")
    split = "test"
    data_root = work / "data"
    # dataset layout the repo expects (submit_engine.submit): DanceTrack/<split>/<seq> or <ds>/images/<split>/<seq>
    if dataset in ("DanceTrack", "SportsMOT"):
        _symlink(seq_dir, data_root / dataset / split / seq)
    else:
        _symlink(seq_dir, data_root / dataset / "images" / split / seq)
    submit_dir = work / "submit"
    (submit_dir / "train").mkdir(parents=True, exist_ok=True)
    shutil.copy(train_config, submit_dir / "train" / "config.yaml")   # submit() reads the model config from here
    _symlink(ckpt, submit_dir / "model.pth")                           # and the checkpoint from SUBMIT_DIR/SUBMIT_MODEL
    rt = dict(cfg)
    rt.update({"MODE": "submit", "SUBMIT_DIR": str(submit_dir), "SUBMIT_MODEL": "model.pth", "SUBMIT_DATA_SPLIT": split,
               "DATA_ROOT": str(data_root), "DET_SCORE_THRESH": det_thresh, "TRACK_SCORE_THRESH": track_thresh,
               "RESULT_SCORE_THRESH": result_thresh, "MISS_TOLERANCE": int(miss_tolerance), "USE_MOTION": bool(use_motion),
               "AVAILABLE_GPUS": str(gpu), "USE_DISTRIBUTED": False, "VISUALIZE": False, "OUTPUTS_DIR": str(work / "outputs")})
    rt_path = work / "runtime.yaml"
    rt_path.write_text(yaml.safe_dump(rt))
    result = submit_dir / split / "tracker" / f"{seq}.txt"
    if result.exists():
        result.unlink()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    _run(["python3", "main.py", "--mode", "submit", "--config-path", rt_path, "--submit-dir", submit_dir,
          "--submit-model", "model.pth", "--submit-data-split", split, "--data-root", data_root,
          "--miss-tolerance", int(miss_tolerance)], repo, out_dir / "memotr.log", env)
    if not result.exists():
        raise SystemExit(f"MeMOTR produced no {result}; see {out_dir / 'memotr.log'}")
    return result


# ------------------------------------------------------------------------------------------------
def build_det_db(frames_meta: dict, det_db_path, key_prefix, yolo_weights="yolo11x.pt", imgsz=1280, conf=0.1,
                 device=0) -> dict:
    """Detector proposals in MOTRv2's det_db format: {"<key_prefix>/img1/<name>.txt": ["l,t,w,h,score\n", ...]}."""
    from ultralytics import YOLO
    model = YOLO(str(yolo_weights))
    img_dir = Path(frames_meta["img_dir"])
    files = sorted(p for p in img_dir.iterdir() if p.suffix == ".jpg")
    db = {}
    for i in range(0, len(files), 16):
        batch = files[i:i + 16]
        res = model.predict([str(p) for p in batch], imgsz=imgsz, conf=conf, classes=[0], device=device, verbose=False, half=True)
        for p, r in zip(batch, res):
            lines = []
            if r.boxes is not None and len(r.boxes):
                for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    lines.append(f"{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f}\n")
            db[f"{key_prefix}/img1/{p.stem}.txt"] = lines
        if (i // 16) % 50 == 0:
            print(f"[proposals] {i + len(batch)}/{len(files)}")
    Path(det_db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(det_db_path).write_text(json.dumps(db))
    return db


def run_motrv2(frames_meta: dict, out_dir, repo, ckpt, miss_tolerance=300, score_thresh=0.5, update_thresh=0.5,
               yolo_weights="yolo11x.pt", proposal_imgsz=1280, gpu="0") -> Path:
    repo, out_dir = Path(repo), Path(out_dir)
    seq_dir, seq = Path(frames_meta["seq_dir"]), frames_meta["seq_name"]
    work = out_dir / "motrv2_work"
    mot_path = work / "data"
    _symlink(seq_dir, mot_path / "DanceTrack" / "test" / seq)          # submit_dance.py hard-codes DanceTrack/test
    det_db = mot_path / "det_db_motrv2.json"
    if not det_db.exists():
        build_det_db(frames_meta, det_db, f"DanceTrack/test/{seq}", yolo_weights, proposal_imgsz, device=int(gpu))
    args = [a for a in (repo / "configs" / "motrv2.args").read_text().split() if a]
    # drop the training-only pretrained path and the det_db entry (we pass ours)
    cleaned = []
    skip = False
    for a in args:
        if skip:
            skip = False
            continue
        if a in ("--pretrained", "--det_db"):
            skip = True
            continue
        cleaned.append(a)
    out = work / "outputs"
    result = out / "tracker" / f"{seq}.txt"
    if result.exists():
        result.unlink()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    _run(["python3", "submit_dance.py", *cleaned, "--det_db", "det_db_motrv2.json", "--exp_name", "tracker", "--resume", ckpt,
          "--mot_path", mot_path, "--output_dir", out, "--score_threshold", score_thresh,
          "--update_score_threshold", update_thresh, "--miss_tolerance", int(miss_tolerance)], repo, out_dir / "motrv2.log", env)
    if not result.exists():
        raise SystemExit(f"MOTRv2 produced no {result}; see {out_dir / 'motrv2.log'}")
    return result


# ------------------------------------------------------------------------------------------------
def track(frames_meta: dict, out_dir, tracker="memotr", **kw) -> Path:
    out_dir = Path(out_dir)
    tracks = out_dir / "tracks.jsonl"
    if tracker == "memotr":
        txt = run_memotr(frames_meta, out_dir, **kw)
    elif tracker == "motrv2":
        txt = run_motrv2(frames_meta, out_dir, **kw)
    else:
        raise SystemExit(f"unknown tracker {tracker}")
    n = mot_txt_to_jsonl(txt, tracks, frames_meta["width"], frames_meta["height"])
    tids = set()
    with open(tracks) as fh:
        for line in fh:
            tids.add(json.loads(line)["tid"])
    write_json(out_dir / "tracks_meta.json", {"tracker": tracker, "n_obs": n, "n_tracks": len(tids), "source": str(txt), **{k: str(v) for k, v in kw.items()}})
    print(f"[tracker] {tracker}: {n} boxes, {len(tids)} track ids → {tracks}")
    return tracks


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True, help="frames_meta.json written by frames.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracker", default="memotr", choices=["memotr", "motrv2"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-config", default=None, help="memotr: repo train config (default configs/train_mot17.yaml)")
    ap.add_argument("--miss-tolerance", type=int, default=300, help="frames a hidden track query stays alive")
    ap.add_argument("--det-thresh", type=float, default=0.5)
    ap.add_argument("--track-thresh", type=float, default=0.5)
    ap.add_argument("--no-motion", action="store_true", help="memotr: disable the motion model for hidden tracks")
    ap.add_argument("--yolo-weights", default="yolo11x.pt", help="motrv2: proposal detector")
    ap.add_argument("--gpu", default="0")
    a = ap.parse_args()
    meta = json.loads(Path(a.frames_meta).read_text())
    if a.tracker == "memotr":
        track(meta, a.out, "memotr", repo=a.repo, ckpt=a.ckpt, train_config=a.train_config, miss_tolerance=a.miss_tolerance,
              det_thresh=a.det_thresh, track_thresh=a.track_thresh, result_thresh=a.track_thresh, use_motion=not a.no_motion, gpu=a.gpu)
    else:
        track(meta, a.out, "motrv2", repo=a.repo, ckpt=a.ckpt, miss_tolerance=a.miss_tolerance, score_thresh=a.det_thresh,
              update_thresh=a.track_thresh, yolo_weights=a.yolo_weights, gpu=a.gpu)
