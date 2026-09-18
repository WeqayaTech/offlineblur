#!/bin/bash
# OfflineBlur — run every stage on one video (resumable: a finished stage is skipped).
#   bash run.sh /workspace/offlineblur/videos/clip.mp4 [out_root]
# env knobs: MAXS=120 (seconds to process, 0 = all)  IMGSZ=1280  VIEWER=male
#            JUDGE=Qwen/Qwen2.5-VL-7B-Instruct   S1_ARGS…S5_ARGS = extra flags per stage
#   force a stage to re-run by deleting its output (s1_meta.json / segments.json / segment_votes.json / classes.json / render/)
set -eo pipefail
V="$1"; ROOT="${2:-/workspace/offlineblur/out}"
[ -f "$V" ] || { echo "usage: bash run.sh <video> [out_root]"; exit 1; }
STEM=$(basename "${V%.*}"); O="$ROOT/$STEM"
CODE="$(cd "$(dirname "$0")" && pwd)/offlineblur"
export HF_HOME=${HF_HOME:-/workspace/hf}
export OFFLINEBLUR_WEIGHTS=${OFFLINEBLUR_WEIGHTS:-/workspace/offlineblur/weights}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
mkdir -p "$O"
echo "=== $STEM → $O  ($(date +%T))"
if [ ! -f "$O/s1_meta.json" ]; then
  python3 "$CODE/s1_detect_track.py" --video "$V" --out "$O" --imgsz "${IMGSZ:-1280}" --max-seconds "${MAXS:-0}" $S1_ARGS 2>&1 | tee "$O/s1.log"
else echo "[s1] done already"; fi
if [ ! -f "$O/segments.json" ]; then
  python3 "$CODE/s2_segments.py" --video "$V" --out "$O" $S2_ARGS 2>&1 | tee "$O/s2.log"
else echo "[s2] done already"; fi
if [ ! -f "$O/segment_votes.json" ]; then
  python3 "$CODE/s3_classify.py" --out "$O" --judge "${JUDGE:-Qwen/Qwen2.5-VL-7B-Instruct}" $S3_ARGS 2>&1 | tee "$O/s3.log"
else echo "[s3] done already"; fi
if [ ! -f "$O/classes.json" ]; then
  python3 "$CODE/s4_identities.py" --out "$O" $S4_ARGS 2>&1 | tee "$O/s4.log"
else echo "[s4] done already"; fi
if [ ! -f "$O/render/render_summary.json" ]; then
  python3 "$CODE/s5_render.py" --video "$V" --out "$O" --viewer "${VIEWER:-male}" $S5_ARGS 2>&1 | tee "$O/s5.log"
else echo "[s5] done already"; fi
python3 "$CODE/s6_gallery.py" --out "$O"
echo "=== $STEM done ($(date +%T)). Outputs: $O/render/  review: $O/review.html"
