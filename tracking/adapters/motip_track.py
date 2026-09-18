#!/usr/bin/env python3
"""Bake-off adapter A — MOTIP (end-to-end transformer, CVPR 2025), the runnable stand-in for MOTRv3.

Drives the official MCG-NJU/MOTIP repo unmodified through its `submit` mode on a DanceTrack-layout
image sequence, then normalises the MOT text to tracks.jsonl. MOTIP ships no MOT17/street checkpoint,
so we run its DanceTrack weights (trained on dancers) — a real domain gap for street crowds, and part
of what this bake-off is measuring.

    python3 motip_track.py --frames-meta <seq>/frames_meta.json --out <out_dir> --repo <MOTIP> \
        --config <cfg.yaml> --ckpt <model.pth> [--gpu 0] [--fp16]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import mot_txt_to_jsonl, read_json, write_json


def _symlink(src, dst):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(Path(src).resolve())


def run(frames_meta, out_dir, repo, config, ckpt, gpu="0", fp16=True):
    repo, out_dir = Path(repo), Path(out_dir)
    seq_dir, seq = Path(frames_meta["seq_dir"]), frames_meta["seq"]
    work = out_dir / "motip_work"
    data_root = work / "datasets"
    _symlink(seq_dir, data_root / "DanceTrack" / "test" / seq)  # submit reads DATA/DanceTrack/test/<seq>/img1
    outputs = work / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    # clear any previous seq result so the glob below is unambiguous
    for old in outputs.rglob(f"{seq}.txt"):
        old.unlink()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = ["accelerate", "launch", "--num_processes=1", "submit_and_evaluate.py",
           "--data-root", str(data_root), "--inference-mode", "submit", "--config-path", str(config),
           "--inference-model", str(ckpt), "--outputs-dir", str(outputs),
           "--inference-dataset", "DanceTrack", "--inference-split", "test"]
    if fp16:
        cmd += ["--inference-dtype", "FP16"]
    print("[motip] $", " ".join(cmd))
    log = out_dir / "motip.log"
    with open(log, "w") as fh:
        p = subprocess.run(cmd, cwd=str(repo), stdout=fh, stderr=subprocess.STDOUT, env=env)
    if p.returncode != 0:
        tail = "\n".join(Path(log).read_text().splitlines()[-40:])
        raise SystemExit(f"MOTIP failed (exit {p.returncode}); tail of {log}:\n{tail}")
    hits = list(outputs.rglob(f"{seq}.txt"))
    if not hits:
        raise SystemExit(f"MOTIP produced no {seq}.txt under {outputs}; see {log}")
    tracks = out_dir / "tracks.jsonl"
    n = mot_txt_to_jsonl(hits[0], tracks, frames_meta["width"], frames_meta["height"])
    tids = {json.loads(l)["tid"] for l in open(tracks)}
    write_json(out_dir / "tracks_meta.json", {"tracker": "motip", "n_obs": n, "n_tracks": len(tids),
               "source": str(hits[0]), "ckpt": str(ckpt), "config": str(config)})
    print(f"[motip] {n} boxes, {len(tids)} ids → {tracks}")
    return tracks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--fp16", action="store_true")
    a = ap.parse_args()
    run(read_json(a.frames_meta), a.out, a.repo, a.config, a.ckpt, a.gpu, a.fp16)
