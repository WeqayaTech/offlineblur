#!/bin/bash
# SAMURAI vs BoT-SORT+CLIP-ReID on one clip.
#   bash run_samurai_bakeoff.sh /root/tracking/videos/clip.mp4 [out_root]
# env: GPU=0  IMGSZ=1280  CONF=0.25  REID=clip_market1501.pt  SAMURAI_CONF=0.4
#      SAMURAI_REPO=/root/tracking/samurai  SAMURAI_CKPT=<ckpt.pt>  SAMURAI_MODEL=large  W=/root/tracking/weights
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_samurai_bakeoff.sh <video> [out_root]"; exit 1; }
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1 HF_HOME=${HF_HOME:-/root/hf}
ROOT="${2:-/root/tracking/out}"; STEM=$(basename "${V%.*}"); O="$ROOT/$STEM"; mkdir -p "$O"
CODE="$(cd "$(dirname "$0")" && pwd)"
W=${W:-/root/tracking/weights}
SAMURAI_REPO=${SAMURAI_REPO:-/root/tracking/samurai}
SAMURAI_CKPT=${SAMURAI_CKPT:-$SAMURAI_REPO/sam2/checkpoints/sam2.1_hiera_large.pt}
SAMURAI_MODEL=${SAMURAI_MODEL:-large}
GPU=${GPU:-0}
echo "=== bake-off $STEM → $O ($(date +%T))"

# 1) frames (shared by both trackers, DanceTrack layout)
SEQ="$O/seq/$STEM"
python3 -c "import sys; sys.path.insert(0,'$CODE'); from common import extract_frames; extract_frames('$V','$SEQ',${MAXS:-0})"
META="$SEQ/frames_meta.json"

# 2) SAMURAI (frame-0 seeded, whole-clip propagation)
if [ ! -f "$O/samurai/tracks.jsonl" ]; then
  mkdir -p "$O/samurai"
  python3 "$CODE/adapters/samurai_track.py" --frames-meta "$META" --out "$O/samurai" \
    --samurai-repo "$SAMURAI_REPO" --checkpoint "$SAMURAI_CKPT" --model-size "$SAMURAI_MODEL" \
    --yolo "$W/yolo11x.pt" --conf "${SAMURAI_CONF:-0.4}" --gpu "$GPU"
else echo "[samurai] done"; fi

# 3) BoT-SORT + CLIP-ReID
if [ ! -f "$O/botsort/tracks.jsonl" ]; then
  mkdir -p "$O/botsort"
  python3 "$CODE/adapters/botsort_track.py" --frames-meta "$META" --out "$O/botsort" \
    --yolo "$W/yolo11x.pt" --reid "${REID:-clip_market1501.pt}" --imgsz "${IMGSZ:-1280}" --conf "${CONF:-0.25}" --gpu "$GPU"
else echo "[botsort] done"; fi

# 4) compare
python3 "$CODE/compare.py" --seq "$STEM" --frames-meta "$META" \
  --a "name=samurai,tracks=$O/samurai/tracks.jsonl" --b "name=botsort,tracks=$O/botsort/tracks.jsonl" --out "$O/compare" --video "$V"
echo "=== $STEM done. side-by-side: $O/compare/${STEM}_compare.mp4  metrics: $O/compare/${STEM}_metrics.json"
