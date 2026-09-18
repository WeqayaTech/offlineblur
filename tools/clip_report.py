#!/usr/bin/env python3
"""Summarise one clip's outputs (fetched locally) as markdown + extract evenly spaced debug/blurred frames.
    python3 tools/clip_report.py <local_out_dir> [n_frames]
Needs in <local_out_dir>: s1_meta.json, segments.json, identities.json, classes.json, render_summary.json,
                          <stem>_debug.mp4 / <stem>_blurred.mp4 (frames are extracted from these), s*.log (timings)"""
import json, re, subprocess, sys
from collections import Counter
from pathlib import Path

d = Path(sys.argv[1]); n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 6
J = lambda n: json.load(open(d / n))
meta, segs, ident, cls, rs = J("s1_meta.json"), J("segments.json"), J("identities.json"), J("classes.json"), J("render_summary.json")
ids = cls["identities"]
print(f"# {meta['video']}\n")
print(f"{meta['width']}x{meta['height']} @ {meta['fps']:.2f} fps · {meta['n_frames']} frames processed ({meta['n_frames']/meta['fps']:.0f} s) · "
      f"{meta['n_tracks']} raw tracks · {segs['stats']['segments']} segments ({segs['stats']['gap_splits']} gap splits, "
      f"{segs['stats']['with_face']} with a face, {segs['stats']['duplicates']} duplicates)\n")
print(f"identities {cls['summary']['identities']} · by class {cls['summary']['by_class']} · uncertain {cls['summary']['uncertain']} · "
      f"links {cls['summary']['links']} · vote splits {cls['summary']['vote_splits']}\n")
print(f"blurred as {rs['target']}: {rs['blur_identities']} · skipped uncertain {rs.get('skipped_uncertain')} · frames with blur {rs['frames_with_blur']}/{rs['frames']}\n")
# timings
t = {}
for stage in ("s1", "s2", "s3", "s4", "s5"):
    p = d / f"{stage}.log"
    if p.exists():
        txt = p.read_text()
        m = re.findall(r"in ([\d.]+) min|\'seconds\': ([\d.]+)|in (\d+)s", txt)
        t[stage] = m[-1] if m else None
print("timings (from logs):", {k: [x for x in v if x][0] if v else None for k, v in t.items()}, "\n")
print("## Identities blurred (Woman)\n")
print("| id | seconds | tracks | votes W/M | weight W/M | uncertain | reasons | links |\n|---|---|---|---|---|---|---|---|")
for r in sorted(ids, key=lambda r: -r["seconds"]):
    if r["cls"] == rs["target"]:
        gv, gw = r.get("gender_votes", {}), r.get("gender_weight", {})
        print(f"| {r['identity']} | {r['seconds']} | {r['tracks'][:6]} | {gv.get('woman',0)}/{gv.get('man',0)} | {gw.get('woman',0)}/{gw.get('man',0)} | "
              f"{'yes' if r['uncertain'] else ''} | {'; '.join(r['reasons'])} | {','.join(r.get('links', []))} |")
print("\n## Long identities NOT blurred (>= 3 s), to check for escapes\n")
print("| id | cls | seconds | votes W/M | uncertain | reasons |\n|---|---|---|---|---|---|")
for r in sorted(ids, key=lambda r: -r["seconds"]):
    if r["cls"] != rs["target"] and r["seconds"] >= 3:
        gv = r.get("gender_votes", {})
        print(f"| {r['identity']} | {r['cls']} | {r['seconds']} | {gv.get('woman',0)}/{gv.get('man',0)} | {'yes' if r['uncertain'] else ''} | {'; '.join(r['reasons'])} |")
print("\nidentity seconds distribution:", sorted(Counter(min(int(r['seconds']), 10) for r in ids).items()))
# frames
stem = Path(meta["video"]).stem
fd = d / "frames"; fd.mkdir(exist_ok=True)
dur = meta["n_frames"] / meta["fps"]
for i in range(n_frames):
    ts = dur * (i + 0.5) / n_frames
    for kind in ("debug", "blurred"):
        src = d / f"{stem}_{kind}.mp4"
        if src.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{ts:.2f}", "-i", str(src), "-frames:v", "1", str(fd / f"{kind}_{i}_{ts:.0f}s.jpg")])
print(f"\nframes → {fd}")
