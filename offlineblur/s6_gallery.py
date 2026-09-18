#!/usr/bin/env python3
"""OfflineBlur stage 6 — self-contained review page: one card per identity (dropped and uncertain
first), its crops across the video, the vote counts and every reason it was flagged. Open in any
browser; nothing else needed.

    python3 s6_gallery.py --out out/clip
"""
import argparse
import base64
import html
import json
from pathlib import Path

import cv2

from common import read_json


def thumb(path, max_side=220):
    im = cv2.imread(str(path))
    if im is None:
        return ""
    h, w = im.shape[:2]
    s = max_side / max(h, w)
    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-crops", type=int, default=12)
    a = ap.parse_args()
    out = Path(a.out)
    meta = read_json(out / "s1_meta.json")
    sf = out / "segments_final.json"
    feats = read_json(sf if sf.exists() else out / "segments.json")["segments"]
    classes = read_json(out / "classes.json")
    rs = out / "render" / "render_summary.json"
    rsum = read_json(rs) if rs.exists() else {}
    ids = classes["identities"]
    order = sorted(ids, key=lambda r: (0 if r["dropped_reason"] else (1 if r["uncertain"] else 2), -r["seconds"]))
    blur = set(rsum.get("blur_identities", []))
    H = [f"<html><head><meta charset='utf-8'><title>{html.escape(meta['video'])} review</title>",
         "<style>body{font-family:sans-serif;background:#111;color:#ddd} .card{border:2px solid #444;margin:10px;padding:8px;border-radius:6px}",
         ".Woman{border-color:#e0e} .Man{border-color:#38f} .Child{border-color:#fa0} .drop{border-color:#666;opacity:.6}",
         ".unc{background:#301} img{margin:2px;border-radius:3px} .tag{display:inline-block;padding:2px 6px;border-radius:4px;background:#333;margin-right:4px}",
         ".blur{background:#a0a}</style></head><body>",
         f"<h2>{html.escape(meta['video'])} — {meta['width']}x{meta['height']} @ {meta['fps']:.2f} fps, {meta['n_frames']} frames</h2>",
         f"<p>classes: {html.escape(json.dumps(classes['summary']))}</p>",
         f"<p>render: blurred identities {sorted(blur)}</p>" if rsum else "",
         "<p>Cards: dropped first, then uncertain, then by seconds on screen. Border: magenta Woman, blue Man, orange Child, grey dropped. "
         "Pink background = uncertain (check these). BLURRED tag = this identity is blurred in the output.</p>"]
    for r in order:
        cls = r["cls"]
        klass = "card " + ("drop" if r["dropped_reason"] else (cls or "")) + (" unc" if r["uncertain"] else "")
        tags = [f"<span class='tag'>{cls or '—'}</span>", f"<span class='tag'>{r['seconds']} s</span>",
                f"<span class='tag'>{len(r['segments'])} segments / {len(r['tracks'])} tracks</span>"]
        if r.get("n_weak"):
            tags.append(f"<span class='tag'>{r['n_weak']} sliver crops (no vote)</span>")
        if r["identity"] in blur:
            tags.append("<span class='tag blur'>BLURRED</span>")
        if r["dropped_reason"]:
            tags.append(f"<span class='tag'>DROPPED: {r['dropped_reason']}</span>")
        if r["uncertain"]:
            tags.append(f"<span class='tag'>UNCERTAIN: {html.escape('; '.join(r['reasons']) or '-')}</span>")
        for key in ("verdict_votes", "gender_votes", "gender_weight", "age_weight"):
            if r.get(key):
                tags.append(f"<span class='tag'>{key.replace('_', ' ')} {html.escape(json.dumps(r[key]))}</span>")
        if r.get("links"):
            tags.append(f"<span class='tag'>links: {', '.join(r['links'])}</span>")
        if r.get("median_age") is not None:
            tags.append(f"<span class='tag'>median age {r['median_age']}</span>")
        H.append(f"<div class='{klass}'><b>identity {r['identity']}</b> " + " ".join(tags) + "<br>")
        crops = sorted((c for sid in r["segments"] for c in feats[str(sid)]["crops"]), key=lambda c: c["f"])
        step = max(1, len(crops) // a.max_crops)
        for c in crops[::step][:a.max_crops]:
            H.append(f"<img src='{thumb(out / 'crops' / c['crop'])}' title='f{c['f']} {'face' if c['face'] else ''}'>")
        H.append("</div>")
    H.append("</body></html>")
    (out / "review.html").write_text("\n".join(H))
    print(f"[s6] review.html: {len(ids)} identities → {out / 'review.html'}")


if __name__ == "__main__":
    main()
