#!/bin/bash
# OfflineBlur v2 — run the four phases on one video (resumable: a finished phase is skipped).
#   bash run_v2.sh /root/offlineblur_v2/videos/clip.mp4 [out_root]
# env knobs: MAXS=120 (seconds, 0 = all)  TRACKER=motrv2|memotr (motrv2 default: cleanest on street crowds)  CKPT=memotr_mot17.pth|memotr_dancetrack.pth
#            MISS=300 (frames a hidden track stays alive)  BACKBONE=dinov2-giant  STRIDE=8  EVERY=1  RELINK=0 (cosine, 0 = off)
#            HEAD=weights/demo_head.pt   TRACK_ARGS / DEMO_ARGS / AGG_ARGS = extra flags per phase
#   re-run a phase by deleting its output (tracks.jsonl / attrs.jsonl / identities.json / render/)
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_v2.sh <video> [out_root]"; exit 1; }
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1 HF_HOME=${HF_HOME:-/root/hf}
V2_ROOT=${V2_ROOT:-/root/offlineblur_v2}; W=${V2_WEIGHTS:-$V2_ROOT/weights}
ROOT="${2:-$V2_ROOT/out}"; STEM=$(basename "${V%.*}"); O="$ROOT/$STEM"; mkdir -p "$O"
CODE="$(cd "$(dirname "$0")" && pwd)/stp"
TRACKER=${TRACKER:-motrv2}
echo "=== $STEM → $O  ($(date +%T))"

python3 "$CODE/frames.py" --video "$V" --seq-root "$O/frames" --seq "$STEM" --max-seconds "${MAXS:-0}" 2>&1 | tee "$O/frames.log"
META="$O/frames/$STEM/frames_meta.json"

if [ ! -f "$O/tracks.jsonl" ]; then
  if [ "$TRACKER" = "memotr" ]; then
    CK="$W/${CKPT:-memotr_mot17.pth}"; CFG=train_mot17.yaml; [[ "$CK" == *dancetrack* ]] && CFG=train_dancetrack.yaml
    python3 "$CODE/tracker.py" --frames-meta "$META" --out "$O" --tracker memotr --repo "$V2_ROOT/third_party/MeMOTR" --ckpt "$CK" \
      --train-config "$V2_ROOT/third_party/MeMOTR/configs/$CFG" --miss-tolerance "${MISS:-300}" $TRACK_ARGS 2>&1 | tee "$O/track.log"
  else
    PYTHONPATH="$V2_ROOT/third_party/MOTRv2/models/ops" python3 "$CODE/tracker.py" --frames-meta "$META" --out "$O" --tracker motrv2 \
      --repo "$V2_ROOT/third_party/MOTRv2" --ckpt "$W/${CKPT:-motrv2_dancetrack.pth}" --miss-tolerance "${MISS:-300}" \
      --yolo-weights "$W/yolo11x.pt" $TRACK_ARGS 2>&1 | tee "$O/track.log"
  fi
else echo "[track] done already"; fi

if [ ! -f "$O/attrs.jsonl" ]; then
  python3 "$CODE/demographics.py" --out "$O" --head "${HEAD:-$W/demo_head.pt}" --backbone "${BACKBONE:-dinov2-giant}" \
    --feat-stride "${STRIDE:-8}" --every "${EVERY:-1}" $DEMO_ARGS 2>&1 | tee "$O/demo.log"
else echo "[demo] done already"; fi

if [ ! -f "$O/identities.json" ]; then
  python3 "$CODE/aggregator.py" --out "$O" --relink-sim "${RELINK:-0}" $AGG_ARGS 2>&1 | tee "$O/aggregate.log"
else echo "[aggregate] done already"; fi

python3 "$CODE/render.py" --out "$O" --video "$V" 2>&1 | tee "$O/render.log"
echo "=== $STEM done ($(date +%T)). identities: $O/identities.json  debug video: $O/render/${STEM}_debug.mp4"
