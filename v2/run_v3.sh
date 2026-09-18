#!/bin/bash
# OfflineBlur v3 — transformer tracking + ZERO-SHOT VLM judge (no training) + Bayesian pooling + blur.
#   bash run_v3.sh /root/offlineblur/videos/clip.mp4 [out_root]
# Same as v2 but phase 3 is judge_vlm.py (Qwen2.5-VL) instead of the trained head, so no demo_head.pt is needed.
# env knobs: MAXS=0  TRACKER=motrv2|memotr  CKPT=...  MISS=300  K=14 (crops/track)  JUDGE=Qwen/Qwen2.5-VL-7B-Instruct
#            TRACK_ARGS / JUDGE_ARGS / AGG_ARGS / BLUR_ARGS   VIEWER=male
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_v3.sh <video> [out_root]"; exit 1; }
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-/root/hf}
# Qwen2.5-VL is already cached on the volume from v1; read it there (read-only) instead of a 16 GB re-download.
[ -d /workspace/hf-cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct ] && export HF_HOME=/workspace/hf-cache
V2_ROOT=${V2_ROOT:-/root/offlineblur_v2}; W=${V2_WEIGHTS:-$V2_ROOT/weights}
ROOT="${2:-$V2_ROOT/out}"; STEM=$(basename "${V%.*}"); O="$ROOT/${STEM}_v3"; mkdir -p "$O"
CODE="$(cd "$(dirname "$0")" && pwd)/stp"
TRACKER=${TRACKER:-motrv2}
echo "=== v3 $STEM → $O  ($(date +%T))"

python3 "$CODE/frames.py" --video "$V" --seq-root "$O/frames" --seq "$STEM" --max-seconds "${MAXS:-0}" 2>&1 | tee "$O/frames.log"
META="$O/frames/$STEM/frames_meta.json"

if [ ! -f "$O/tracks.jsonl" ]; then
  if [ "$TRACKER" = "memotr" ]; then
    CK="$W/${CKPT:-memotr_mot17.pth}"; CFG=train_mot17.yaml; [[ "$CK" == *dancetrack* ]] && CFG=train_dancetrack.yaml
    HF_HOME=/root/hf python3 "$CODE/tracker.py" --frames-meta "$META" --out "$O" --tracker memotr --repo "$V2_ROOT/third_party/MeMOTR" --ckpt "$CK" \
      --train-config "$V2_ROOT/third_party/MeMOTR/configs/$CFG" --miss-tolerance "${MISS:-300}" $TRACK_ARGS 2>&1 | tee "$O/track.log"
  else
    PYTHONPATH="$V2_ROOT/third_party/MOTRv2/models/ops" HF_HOME=/root/hf python3 "$CODE/tracker.py" --frames-meta "$META" --out "$O" --tracker motrv2 \
      --repo "$V2_ROOT/third_party/MOTRv2" --ckpt "$W/${CKPT:-motrv2_dancetrack.pth}" --miss-tolerance "${MISS:-300}" \
      --yolo-weights "$W/yolo11x.pt" $TRACK_ARGS 2>&1 | tee "$O/track.log"
  fi
else echo "[track] done already"; fi

if [ ! -f "$O/attrs.jsonl" ]; then
  python3 "$CODE/judge_vlm.py" --out "$O" --judge "${JUDGE:-Qwen/Qwen2.5-VL-7B-Instruct}" --k "${K:-14}" $JUDGE_ARGS 2>&1 | tee "$O/judge.log"
else echo "[judge] done already"; fi

if [ ! -f "$O/identities.json" ]; then
  python3 "$CODE/aggregator.py" --out "$O" --relink-sim 0 $AGG_ARGS 2>&1 | tee "$O/aggregate.log"
else echo "[aggregate] done already"; fi

python3 "$CODE/render.py" --out "$O" --video "$V" 2>&1 | tee "$O/render.log"
python3 "$CODE/sheet.py" --out "$O"
if [ ! -f "$O/render/blur_summary.json" ]; then
  python3 "$CODE/blur.py" --out "$O" --video "$V" --viewer "${VIEWER:-male}" --seg-weights "$W/yolo11x-seg.pt" $BLUR_ARGS 2>&1 | tee "$O/blur.log"
fi
echo "=== v3 $STEM done ($(date +%T)). blurred: $O/render/${STEM}_blurred.mp4"
