#!/usr/bin/env python3
"""OfflineBlur stage 3 (v1.2) — ask the judge about every crop of every segment.

Judge: an open vision-language model (default Qwen2.5-VL-7B-Instruct; any transformers
image-text-to-text model works, e.g. Qwen/Qwen3-VL-8B-Instruct) shown the outlined crop and a
fixed prompt. It answers three things per crop: real person / depiction / not a person, man or
woman, child or adult (with an age estimate). Nothing is decided here — stage 4 turns these
per-crop answers into per-segment and per-identity decisions, so the linking can use the class
and the class can use the linking.

Each answer carries a weight for the later vote:
  0     if the crop is a sliver (mask fills < --min-fill of its box, or shorter than --min-height px)
        — a strip of coat behind an occluder says nothing about who wears it
  ×2    if a face was detected in the crop (gender from a visible face beats gender from a back view)
  ×0.5  if the mask fills < 0.5 of the box (partly hidden), and ×0.5 again if under 150 px tall

    python3 s3_classify.py --out out/clip [--judge Qwen/Qwen2.5-VL-7B-Instruct]

Writes  out/verdicts.jsonl        raw model text + parsed answer per crop
        out/segment_votes.json    per segment: the weighted answers
"""
import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from common import read_json, write_json

PROMPT = """One candidate in this image is outlined in green. Judge ONLY the outlined candidate, not anyone else.
Verdict:
- "real_person": a real, physically present human of any age, even if partly hidden, small, seen from the side or from behind.
- "depiction": a photo, poster, billboard, screen, painting or advertisement that shows a human.
- "not_person": a statue, mannequin, doll, toy, cartoon, animal, sign or any object that is not a human.
Judge gender and age from the face and body, never from clothing or head covering.
"child" means 12 years old or younger; teenagers are "adult".
Reply with ONLY this JSON and nothing else:
{"verdict": "real_person|depiction|not_person", "gender": "man|woman|unknown", "age_group": "child|adult|unknown", "estimated_age": <number or null>}
Use "unknown" only when it is genuinely not visible."""


class VLMJudge:
    def __init__(self, repo, device="cuda"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(repo)
        self.model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map=device).eval()
        self.repo = repo

    def ask(self, pil, prompt=PROMPT, max_new_tokens=80):
        msgs = [{"role": "user", "content": [{"type": "image", "image": pil}, {"type": "text", "text": prompt}]}]
        with self.torch.no_grad():
            inp = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                                return_dict=True, return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def parse(txt):
    m = re.search(r"\{.*?\}", txt, re.S)
    if not m:
        return None
    raw = m.group(0)
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        try:
            d = json.loads(raw.replace("'", '"'))
        except json.JSONDecodeError:
            return None
    if not isinstance(d, dict):
        return None
    v = {"verdict": str(d.get("verdict", "")).strip().lower(),
         "gender": str(d.get("gender", "unknown")).strip().lower(),
         "age_group": str(d.get("age_group", "unknown")).strip().lower(),
         "estimated_age": None}
    if v["verdict"] not in ("real_person", "depiction", "not_person"):
        return None
    try:
        if d.get("estimated_age") is not None:
            v["estimated_age"] = float(d["estimated_age"])
    except (TypeError, ValueError):
        pass
    return v


def crop_weight(c, a):
    if c.get("fill", 1.0) < a.min_fill or c.get("h", 1e9) < a.min_height:
        return 0.0
    w = 2.0 if c.get("face") else 1.0
    if c.get("fill", 1.0) < 0.5:
        w *= 0.5
    if c.get("h", 1e9) < 150:
        w *= 0.5
    return w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--min-fill", type=float, default=0.35)
    ap.add_argument("--min-height", type=float, default=0, help="crops shorter than this (px) do not vote; 0 = max(48, 10 %% of frame height)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from PIL import Image
    out = Path(a.out)
    crops_dir = out / "crops"
    segs = read_json(out / "segments.json")["segments"]
    if a.min_height <= 0:
        a.min_height = max(48.0, 0.10 * read_json(out / "s1_meta.json")["height"])
    print(f"[s3] crops shorter than {a.min_height:.0f} px or filling < {a.min_fill} of their box do not vote", flush=True)
    judge = VLMJudge(a.judge, a.device)
    t0 = time.time()
    n_calls = 0
    votes = {}
    with open(out / "verdicts.jsonl", "w") as vf:
        for n, (sid, s) in enumerate(sorted(segs.items(), key=lambda kv: int(kv[0])), 1):
            rows = []
            for c in s["crops"]:
                txt = judge.ask(Image.open(crops_dir / c["crop"]).convert("RGB"))
                v = parse(txt)
                w = crop_weight(c, a) if v else 0.0
                n_calls += 1
                rows.append({"f": c["f"], "crop": c["crop"], "v": v, "w": w, "face": bool(c.get("face")),
                             "fill": c.get("fill"), "h": c.get("h")})
                vf.write(json.dumps({"sid": int(sid), "tid": s["tid"], **rows[-1], "raw": txt}) + "\n")
            votes[sid] = rows
            g = Counter(r["v"]["gender"] for r in rows if r["v"] and r["w"] > 0)
            if n % 10 == 0 or n == len(segs):
                print(f"  [s3] {n}/{len(segs)} segments · {n_calls} calls · {(time.time()-t0)/max(n_calls,1):.2f} s/call "
                      f"· last: sid {sid} t{s['tid']} {dict(g)}", flush=True)
    write_json(out / "segment_votes.json", {"votes": votes, "judge": a.judge, "prompt": PROMPT, "settings": vars(a),
                                            "calls": n_calls, "seconds": round(time.time() - t0, 1)})
    print(f"[s3] {len(segs)} segments, {n_calls} judge calls in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
