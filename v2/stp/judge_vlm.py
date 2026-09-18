#!/usr/bin/env python3
"""V3 phase 3 — zero-shot demographic judge (no training).

Replaces the trained cross-attention head (demographics.py) with a vision-language model that already
knows what a person looks like from any angle. Per track it judges K crops spread over the track's life;
each crop yields one observation in the SAME attrs.jsonl schema demographics.py writes, so the Bayesian
aggregator, blur renderer, contact sheet and report all run downstream unchanged.

Why a VLM and not the trained head: the head was fine-tuned on face datasets, so a woman seen from behind
scores ~0.5. A general VLM (Qwen2.5-VL-7B) has seen whole people in every pose on the web, so it reads
back views, clothing and hair. It still misreads some partial/back crops as "man" (measured in v1), so we
judge many crops per identity, weight face-visible and large-sharp crops up, and let the aggregator pool.

    python3 judge_vlm.py --out out/clip [--judge Qwen/Qwen2.5-VL-7B-Instruct] [--k 14] [--min-side 40]

Writes  out/attrs.jsonl   {"f","tid","p_female","age_mean","age_std","quality","entropy","size_px","occ","sharp","raw"}
        out/attrs_meta.json    (backbone field = the VLM id, so the report and metadata read consistently)
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from common import box_coverage, iter_jsonl, read_json, sharpness, write_json
from frames import frame_path

PROMPT = (
    "You are labelling one person in a street video frame, cropped to a bounding box. "
    "Report ONLY compact JSON, no prose:\n"
    '{"is_person": true|false, "gender": "woman"|"man"|"unclear", "is_child": true|false, '
    '"face_visible": true|false, "view": "front"|"side"|"back", "age": <int years>, "confidence": <0..1>}\n'
    "Rules: gender is the adult gender presentation from hair, build, clothing and face if visible. "
    "Use \"unclear\" only when genuinely ambiguous. is_child true only for someone who looks 12 or younger. "
    "is_person false for a poster, statue, sign, reflection or a crop that is not a real person. "
    "confidence is how sure you are of gender."
)


def pick_spread(items, k, key):
    items = list(items)
    if k <= 0 or len(items) <= k:
        return items
    n, out = len(items), []
    for b in range(k):
        lo = int(b * n / k)
        hi = max(lo + 1, int((b + 1) * n / k))
        out.append(max(items[lo:hi], key=key))
    return out


def parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(m.group(0).replace("'", '"'))
        except Exception:
            return None


def to_attr(d, f, tid, size_px, occ, sharp):
    """Map one VLM answer + crop measures → an attrs.jsonl observation."""
    conf = float(d.get("confidence", 0.5) or 0.5)
    conf = min(1.0, max(0.0, conf))
    g = str(d.get("gender", "unclear")).lower()
    if not d.get("is_person", True):
        p_female, gender_w = 0.5, 0.0            # not a person → zero weight, cannot cause a blur
    elif g.startswith("woman") or g.startswith("female"):
        p_female = 0.5 + 0.5 * conf
        gender_w = 1.0
    elif g.startswith("man") or g.startswith("male"):
        p_female = 0.5 - 0.5 * conf
        gender_w = 1.0
    else:
        p_female, gender_w = 0.5, 0.3            # unclear → near-prior, low weight
    face = 1.0 if d.get("face_visible") else 0.4  # face-visible crops dominate (v1 finding)
    view = str(d.get("view", "")).lower()
    if view == "back":
        face *= 0.7
    age = d.get("age")
    try:
        age = float(age)
        if not (0 <= age <= 100):
            age = None
    except (TypeError, ValueError):
        age = None
    if d.get("is_child"):
        age = min(age if age is not None else 10.0, 12.0)
    # quality drives the aggregator weight; entropy encodes (un)confidence like the trained head's entropy
    quality = float(np.clip(conf * gender_w * face, 0.0, 1.0))
    entropy = float(np.clip(1.0 - conf, 0.0, 1.0))
    age_std = 6.0 if face >= 1.0 else 12.0
    return {"f": f, "tid": tid, "p_female": round(p_female, 4),
            "age_mean": None if age is None else round(age, 1), "age_std": age_std,
            "quality": round(quality, 4), "entropy": round(entropy, 4),
            "size_px": round(size_px, 1), "occ": round(occ, 3), "sharp": round(sharp, 3),
            "raw": {"g": g, "conf": round(conf, 2), "face": bool(d.get("face_visible")), "view": view,
                    "person": bool(d.get("is_person", True)), "child": bool(d.get("is_child", False))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--k", type=int, default=14, help="crops judged per track, spread over its life")
    ap.add_argument("--min-side", type=float, default=40, help="skip crops shorter than this many px")
    ap.add_argument("--pad", type=float, default=0.12, help="context padding added around each box")
    ap.add_argument("--max-px", type=int, default=448, help="longest crop side sent to the VLM")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = Path(a.out)
    fm = read_json(next((out / "frames").glob("*/frames_meta.json")))
    fps, W, H, img_dir = fm["fps"], fm["width"], fm["height"], fm["img_dir"]

    per_track, per_frame = defaultdict(list), defaultdict(list)
    for r in iter_jsonl(out / "tracks.jsonl"):
        per_track[r["tid"]].append(r)
        per_frame[r["f"]].append(r)

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    print(f"[judge] loading {a.judge}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(a.judge, torch_dtype=torch.bfloat16, device_map=a.device)
    proc = AutoProcessor.from_pretrained(a.judge)
    model.eval()

    # choose crops per track: spread over life, prefer big+sharp frames
    jobs = []
    grays = {}
    for tid, rows in per_track.items():
        rows = sorted(rows, key=lambda r: r["f"])
        rows = [r for r in rows if min(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) >= a.min_side]
        if not rows:
            continue
        def q(r):
            b = r["box"]
            return (b[3] - b[1])  # size proxy; sharpness added at read time
        for r in pick_spread(rows, a.k, key=q):
            jobs.append((tid, r))
    print(f"[judge] {len(per_track)} tracks → {len(jobs)} crops to judge")

    n, t0 = 0, time.time()
    with open(out / "attrs.jsonl", "w") as fh:
        for tid, r in jobs:
            f, b = r["f"], r["box"]
            im = cv2.imread(str(frame_path(img_dir, f)))
            if im is None:
                continue
            bw, bh = b[2] - b[0], b[3] - b[1]
            x0 = int(max(0, b[0] - a.pad * bw)); y0 = int(max(0, b[1] - a.pad * bh))
            x1 = int(min(W, b[2] + a.pad * bw)); y1 = int(min(H, b[3] + a.pad * bh))
            crop = im[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(im[int(max(0,b[1])):int(b[3]), int(max(0,b[0])):int(b[2])], cv2.COLOR_BGR2GRAY) if bw*bh > 0 else None
            sh = sharpness(gray) if gray is not None and gray.size else 0.0
            occ = max([box_coverage(b, o["box"]) for o in per_frame[f] if o is not r and o["box"][3] > b[3]] + [0.0])
            scale = a.max_px / max(crop.shape[0], crop.shape[1])
            if scale < 1:
                crop = cv2.resize(crop, (int(crop.shape[1]*scale), int(crop.shape[0]*scale)))
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            from PIL import Image
            msgs = [{"role": "user", "content": [{"type": "image", "image": Image.fromarray(rgb)}, {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[Image.fromarray(rgb)], return_tensors="pt").to(a.device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=120, do_sample=False)
            ans = proc.batch_decode([gen[0][inputs.input_ids.shape[1]:]], skip_special_tokens=True)[0]
            d = parse(ans)
            if d is None:
                continue
            fh.write(json.dumps(to_attr(d, f, tid, bh, occ, sh)) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"[judge] {n}/{len(jobs)} crops, {n/max(1e-6,time.time()-t0):.2f} crops/s")
    write_json(out / "attrs_meta.json", {"backbone": a.judge, "mode": "vlm-zeroshot", "k": a.k, "n_obs": n,
                                         "n_tracks": len(per_track), "seconds": round(time.time()-t0, 1)})
    print(f"[judge] {n} observations in {time.time()-t0:.0f} s → {out/'attrs.jsonl'}")


if __name__ == "__main__":
    main()
