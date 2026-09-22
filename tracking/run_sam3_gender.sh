#!/bin/bash
# SAM 3 with gender as the prompt — the production-pipeline candidate, end to end on one clip.
#   bash run_sam3_gender.sh /root/tracking/videos/clip.mp4 [out_root]
# env: GPU=0  PROMPTS=woman,man,person  BLUR_PROMPT=woman  DTYPE=bfloat16  MAXS=0 (seconds, 0=all)
#      MIN_SCORE=0  NEW_DET_THRESH=  SCORE_THRESH=  MAX_OBJECTS=  MODEL_ID=facebook/sam3
#      STATE_DEVICE=cpu  EXTRA_PROMPTS=child
# MODEL_ID may be a local directory holding the transformers snapshot, which avoids the gated
# download entirely when the weights are already on the machine.
#
# One SAM 3 session runs all three concepts in a single propagation pass (shared vision features):
#   woman  -> the blur set      man -> contrastive, exposes undecided people
#   person -> recall control, exposes people no gender concept fired on (the escapes that ship unblurred)
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_sam3_gender.sh <video> [out_root]"; exit 1; }
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1 HF_HOME=${HF_HOME:-/root/hf}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
ROOT="${2:-/root/tracking/out}"; STEM=$(basename "${V%.*}"); O="$ROOT/$STEM/sam3_gender"; mkdir -p "$O"
CODE="$(cd "$(dirname "$0")" && pwd)"
GPU=${GPU:-0}; PROMPTS=${PROMPTS:-woman,man,person}; BLUR_PROMPT=${BLUR_PROMPT:-woman}
MODEL_ID=${MODEL_ID:-facebook/sam3}
echo "=== SAM 3 gender prompt: $STEM  prompts='$PROMPTS'  ->  $O  ($(date +%T))"

# 1) frames (same DanceTrack layout every other adapter in this module uses)
SEQ="$ROOT/$STEM/seq/$STEM"
python3 -c "import sys; sys.path.insert(0,'$CODE'); from common import extract_frames; extract_frames('$V','$SEQ',${MAXS:-0})"
META="$SEQ/frames_meta.json"

# 2) SAM 3 — detects, segments and tracks every instance of every prompt, no external detector
if [ ! -f "$O/tracks.jsonl" ]; then
  EXTRA=""
  [ -n "$NEW_DET_THRESH" ] && EXTRA="$EXTRA --new-det-thresh $NEW_DET_THRESH"
  [ -n "$SCORE_THRESH" ]   && EXTRA="$EXTRA --score-threshold-detection $SCORE_THRESH"
  [ -n "$MAX_OBJECTS" ]    && EXTRA="$EXTRA --max-num-objects $MAX_OBJECTS"
  python3 "$CODE/adapters/sam3_track.py" --frames-meta "$META" --out "$O" --text "$PROMPTS" \
    --model-id "$MODEL_ID" --gpu "$GPU" --dtype "${DTYPE:-bfloat16}" --min-score "${MIN_SCORE:-0}" \
    --state-device "${STATE_DEVICE:-cpu}" $EXTRA
else echo "[sam3] done (delete $O/tracks.jsonl to rerun)"; fi

# 3) escape / conflict numbers between the three concepts (GT-free: SAM 3 vs itself)
python3 "$CODE/sam3_gender_report.py" --masks "$O/masks.jsonl" --frames-meta "$META" --out "$O/gender" \
  --blur-prompt "$BLUR_PROMPT" --extra-prompts "${EXTRA_PROMPTS:-child}"

# 4) videos: labels (who got which concept), the blur itself, and original|blurred
python3 "$CODE/render_gender.py" --frames-meta "$META" --masks "$O/masks.jsonl" \
  --out "$O/${STEM}_sam3_labels.mp4" --blur-out "$O/${STEM}_sam3_blur.mp4" \
  --sbs-out "$O/${STEM}_sam3_sbs.mp4" --blur-prompt "$BLUR_PROMPT" \
  --verdicts "$O/gender/person_identities.json"

echo "=== $STEM done"
echo "  labels : $O/${STEM}_sam3_labels.mp4"
echo "  blur   : $O/${STEM}_sam3_blur.mp4"
echo "  sbs    : $O/${STEM}_sam3_sbs.mp4"
echo "  metrics: $O/gender/metrics.json"
