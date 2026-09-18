#!/bin/bash
# Tracking bake-off on one clip: MOTIP (transformer) vs BoT-SORT+CLIP-ReID (tracking-by-detection).
#   bash run_bakeoff.sh /root/tracking/videos/clip.mp4 [out_root]
# env: MAXS=0  GPU=0  IMGSZ=1280  CONF=0.25  REID=clip_market1501.pt
#      MOTIP_REPO=/root/tracking/MOTIP  MOTIP_CFG=<cfg.yaml>  MOTIP_CKPT=<model.pth>  W=/root/tracking/weights
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_bakeoff.sh <video> [out_root]"; exit 1; }
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1 HF_HOME=${HF_HOME:-/root/hf}
ROOT="${2:-/root/tracking/out}"; STEM=$(basename "${V%.*}"); O="$ROOT/$STEM"; mkdir -p "$O"
CODE="$(cd "$(dirname "$0")" && pwd)"
W=${W:-/root/tracking/weights}
MOTIP_REPO=${MOTIP_REPO:-/root/tracking/MOTIP}
MOTIP_CFG=${MOTIP_CFG:-$MOTIP_REPO/configs/r50_deformable_detr_motip_dancetrack.yaml}
MOTIP_CKPT=${MOTIP_CKPT:-$W/r50_deformable_detr_motip_dancetrack.pth}
GPU=${GPU:-0}
echo "=== bake-off $STEM → $O ($(date +%T))"

# 1) frames (shared by both trackers, DanceTrack layout)
SEQ="$O/seq/$STEM"
python3 -c "import sys; sys.path.insert(0,'$CODE'); from common import extract_frames; extract_frames('$V','$SEQ',${MAXS:-0})"
META="$SEQ/frames_meta.json"

# 2) MOTIP
if [ ! -f "$O/motip/tracks.jsonl" ]; then
  mkdir -p "$O/motip"
  python3 "$CODE/adapters/motip_track.py" --frames-meta "$META" --out "$O/motip" --repo "$MOTIP_REPO" \
    --config "$MOTIP_CFG" --ckpt "$MOTIP_CKPT" --gpu "$GPU" --fp16
else echo "[motip] done"; fi

# 3) BoT-SORT + CLIP-ReID
if [ ! -f "$O/botsort/tracks.jsonl" ]; then
  mkdir -p "$O/botsort"
  python3 "$CODE/adapters/botsort_track.py" --frames-meta "$META" --out "$O/botsort" \
    --yolo "$W/yolo11x.pt" --reid "${REID:-clip_market1501.pt}" --imgsz "${IMGSZ:-1280}" --conf "${CONF:-0.25}" --gpu "$GPU"
else echo "[botsort] done"; fi

# 4) compare
python3 "$CODE/compare.py" --seq "$STEM" --frames-meta "$META" \
  --a "name=motip,tracks=$O/motip/tracks.jsonl" --b "name=botsort,tracks=$O/botsort/tracks.jsonl" --out "$O/compare" --video "$V"
echo "=== $STEM done. side-by-side: $O/compare/${STEM}_compare.mp4  metrics: $O/compare/${STEM}_metrics.json"
