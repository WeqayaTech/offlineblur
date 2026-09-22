#!/bin/bash
# McByte (SAM+Cutie mask-conditioned ByteTrack, via Roboflow's `trackers[mask]`) vs BoT-SORT+CLIP-ReID.
#   bash run_mcbyte_bakeoff.sh /root/tracking/videos/clip.mp4 [out_root]
# env: GPU=0  IMGSZ=1280  CONF=0.25  REID=clip_market1501.pt  YOLO=yolo26x.pt
#      TRACK_THRESH=0.25  W=/root/tracking/weights
# Unlike SAMURAI/SAM3, McByte needs no separate repo clone or old CUDA toolchain: `pip install
# "trackers[mask]"` pulls modern torch + rf-segment-anything + rf-cutie into this same env, and SAM/Cutie
# checkpoints auto-download on first use.
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_mcbyte_bakeoff.sh <video> [out_root]"; exit 1; }
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1 HF_HOME=${HF_HOME:-/root/hf}
ROOT="${2:-/root/tracking/out}"; STEM=$(basename "${V%.*}"); O="$ROOT/$STEM"; mkdir -p "$O"
CODE="$(cd "$(dirname "$0")" && pwd)"
W=${W:-/root/tracking/weights}
YOLO=${YOLO:-yolo26x.pt}
GPU=${GPU:-0}
echo "=== bake-off $STEM → $O ($(date +%T))"

# 1) frames (shared by both trackers, DanceTrack layout)
SEQ="$O/seq/$STEM"
python3 -c "import sys; sys.path.insert(0,'$CODE'); from common import extract_frames; extract_frames('$V','$SEQ',${MAXS:-0})"
META="$SEQ/frames_meta.json"

# 2) YOLO detections (external, MOT-format text) — same file both McByte and, if desired, a
#    box-only sanity check can consume; McByte itself has no detector of its own.
DETS="$O/dets_$(basename "$YOLO" .pt).txt"
if [ ! -f "$DETS" ]; then
  python3 "$CODE/gen_yolo_dets.py" --frames-meta "$META" --out "$DETS" --yolo "$YOLO" --imgsz "${IMGSZ:-1280}" --conf "${CONF:-0.25}" --gpu "$GPU"
else echo "[gen-dets] done"; fi

# 3) McByte (mask-level: SAM-seeded, Cutie-propagated per-pixel masks as an association cue)
if [ ! -f "$O/mcbyte/tracks.jsonl" ]; then
  mkdir -p "$O/mcbyte"
  python3 "$CODE/adapters/mcbyte_track.py" --frames-meta "$META" --out "$O/mcbyte" --det-txt "$DETS" \
    --track-activation-threshold "${TRACK_THRESH:-0.25}" --high-conf-det-threshold "${TRACK_THRESH:-0.25}" --gpu "$GPU"
else echo "[mcbyte] done"; fi

# 4) BoT-SORT + CLIP-ReID
if [ ! -f "$O/botsort/tracks.jsonl" ]; then
  mkdir -p "$O/botsort"
  python3 "$CODE/adapters/botsort_track.py" --frames-meta "$META" --out "$O/botsort" \
    --yolo "$W/yolo11x.pt" --reid "${REID:-clip_market1501.pt}" --imgsz "${IMGSZ:-1280}" --conf "${CONF:-0.25}" --gpu "$GPU"
else echo "[botsort] done"; fi

# 5) compare
python3 "$CODE/compare.py" --seq "$STEM" --frames-meta "$META" \
  --a "name=mcbyte,tracks=$O/mcbyte/tracks.jsonl" --b "name=botsort,tracks=$O/botsort/tracks.jsonl" --out "$O/compare" --video "$V"

# 6) mask showcase (McByte's actual differentiator over a box-only tracker)
python3 "$CODE/render_masks.py" --frames-meta "$META" --masks "$O/mcbyte/masks.jsonl" --out "$O/mcbyte/mcbyte_masks.mp4" --label McByte

echo "=== $STEM done. side-by-side: $O/compare/${STEM}_compare.mp4  mask showcase: $O/mcbyte/mcbyte_masks.mp4  metrics: $O/compare/${STEM}_metrics.json"
