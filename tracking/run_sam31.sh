#!/bin/bash
# SAM 3.1 (Object Multiplex) on one clip: person tracking video + gender-prompt videos, with profiling.
#   bash run_sam31.sh /root/tracking/videos/clip.mp4 [out_root]
# env: PROMPTS=woman,man,person  BLUR_PROMPT=woman  MAXS=0 (seconds, 0=all)  MAX_OBJECTS=128
#      GROUNDING_BATCH=4  MEMORY_KEEP=20  CKPT="/workspace/SAM 3.1/sam3.1_multiplex.pt"
#      VENV=/root/venv31  EXTRA_PROMPTS=child  SKIP_PERSON=0
# Every prompt is its own SAM 3.1 session (the sam3 repo API takes one text prompt per session), so the
# gender run costs ~3x the person run. Outputs sit next to the SAM 3 ones: <out_root>/<stem>/sam31_*.
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_sam31.sh <video> [out_root]"; exit 1; }
. "${VENV:-/root/venv31}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
ROOT="${2:-/root/tracking/out}"; STEM=$(basename "${V%.*}")
CODE="$(cd "$(dirname "$0")" && pwd)"
CKPT=${CKPT:-/workspace/SAM 3.1/sam3.1_multiplex.pt}
PROMPTS=${PROMPTS:-woman,man,person}; BLUR_PROMPT=${BLUR_PROMPT:-woman}
ARGS=(--checkpoint "$CKPT" --max-num-objects "${MAX_OBJECTS:-128}"
      --grounding-batch "${GROUNDING_BATCH:-4}" --memory-keep-frames "${MEMORY_KEEP:-20}")

SEQ="$ROOT/$STEM/seq/$STEM"
python3 -c "import sys; sys.path.insert(0,'$CODE'); from common import extract_frames; extract_frames('$V','$SEQ',${MAXS:-0})"
META="$SEQ/frames_meta.json"

# 1) person only — the tracking showcase and the like-for-like speed/memory number vs SAM 3
if [ "${SKIP_PERSON:-0}" != "1" ]; then
  P="$ROOT/$STEM/sam31_person"
  [ -f "$P/tracks_meta.json" ] || python3 -W ignore "$CODE/adapters/sam31_track.py" --frames-meta "$META" \
      --out "$P" --text person "${ARGS[@]}" 2>&1 | tr '\r' '\n' | grep --line-buffered -E '^\[sam31\]|Error'
  python3 "$CODE/render_masks.py" --frames-meta "$META" --masks "$P/masks.jsonl" \
      --out "$P/${STEM}_sam31_person.mp4" --label "SAM 3.1"
fi

# 2) gender as the prompt — same report + renders as run_sam3_gender.sh
O="$ROOT/$STEM/sam31_gender"
[ -f "$O/tracks_meta.json" ] || python3 -W ignore "$CODE/adapters/sam31_track.py" --frames-meta "$META" \
    --out "$O" --text "$PROMPTS" "${ARGS[@]}" 2>&1 | tr '\r' '\n' | grep --line-buffered -E '^\[sam31\]|Error'
python3 "$CODE/sam3_gender_report.py" --masks "$O/masks.jsonl" --frames-meta "$META" --out "$O/gender" \
  --blur-prompt "$BLUR_PROMPT" --extra-prompts "${EXTRA_PROMPTS:-child}"
python3 "$CODE/render_gender.py" --frames-meta "$META" --masks "$O/masks.jsonl" \
  --out "$O/${STEM}_sam31_labels.mp4" --blur-out "$O/${STEM}_sam31_blur.mp4" \
  --sbs-out "$O/${STEM}_sam31_sbs.mp4" --blur-prompt "$BLUR_PROMPT" \
  --verdicts "$O/gender/person_identities.json" --label "SAM 3.1"

echo "=== $STEM done"
echo "  person : $ROOT/$STEM/sam31_person/${STEM}_sam31_person.mp4"
echo "  labels : $O/${STEM}_sam31_labels.mp4"
echo "  sbs    : $O/${STEM}_sam31_sbs.mp4"
echo "  metrics: $O/gender/metrics.json"
