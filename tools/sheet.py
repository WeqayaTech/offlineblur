#!/usr/bin/env python3
"""Contact sheet of every identity with a given class: one row per identity, 6 crops spread over its life.
    python3 sheet.py <out_dir> Woman|Man|Child|None   -> <out_dir>/<class>_sheet.jpg"""
import json, sys, cv2, numpy as np
from pathlib import Path
out = Path(sys.argv[1]); want = sys.argv[2] if len(sys.argv) > 2 else "Woman"
C = json.load(open(out / "classes.json"))
sf = out / "segments_final.json"
F = json.load(open(sf if sf.exists() else out / "segments.json"))["segments"]
rows = []
for r in C["identities"]:
    if str(r["cls"]) != want:
        continue
    crops = sorted((c for s in r["segments"] for c in F[str(s)]["crops"]), key=lambda c: c["f"])
    step = max(1, len(crops) // 6); sel = crops[::step][:6]
    tiles = []
    for c in sel:
        im = cv2.imread(str(out / "crops" / c["crop"])); h, w = im.shape[:2]; s = 200 / max(h, w)
        im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
        pad = np.zeros((200, 200, 3), np.uint8); pad[:im.shape[0], :im.shape[1]] = im
        if c.get("face"): cv2.rectangle(pad, (0, 0), (199, 199), (0, 200, 255), 2)
        tiles.append(pad)
    while len(tiles) < 6:
        tiles.append(np.zeros((200, 200, 3), np.uint8))
    row = np.hstack(tiles)
    label = (f"id{r['identity']} {r['seconds']}s tracks{r['tracks']} {r.get('gender_votes')} "
             + ("UNCERTAIN " + ";".join(r["reasons"]) if r["uncertain"] else "") + " " + ",".join(r.get("links", [])))
    bar = np.zeros((28, row.shape[1], 3), np.uint8)
    cv2.putText(bar, label[:150], (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    rows.append(np.vstack([bar, row]))
if rows:
    cv2.imwrite(str(out / f"{want.lower()}_sheet.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("rows", len(rows))
