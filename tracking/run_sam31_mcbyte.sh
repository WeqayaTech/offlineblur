#!/bin/bash
# SAM 3.1 detector (woman,man,child per frame, one shared backbone pass) -> McByte identities -> per-track
# gender vote. Two envs, because trackers[mask] needs numpy>=2 and the sam3 repo pins numpy<2.
#   bash run_sam31_mcbyte.sh /root/tracking/videos/clip.mp4 [out_root]
# env: PROMPTS=woman,man,child  BLUR_PROMPT=woman  DET_THRESH=0.25 (detector; ByteTrack uses 0.25-0.4 as its
#      low-score second association)  HIGH_CONF=0.4  MERGE_IOU=0.5  LOST_BUFFER=30  MASK_MANAGER=1
#      MIN_SHARE= (blur a track once woman is >= this share of its vote; default argmax)
#      CONTROL=<masks.jsonl with a `person` prompt, e.g. a sam31_track.py run> (evaluation + outlines only)
#      VENV_DET=/root/venv31  VENV_MCB=/root/venv_mcb  CKPT="/workspace/SAM 3.1/sam3.1_multiplex.pt"  MAXS=0
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_sam31_mcbyte.sh <video> [out_root]"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
ROOT="${2:-/root/tracking/out}"; STEM=$(basename "${V%.*}")
CODE="$(cd "$(dirname "$0")" && pwd)"
PROMPTS=${PROMPTS:-woman,man,child}; BLUR_PROMPT=${BLUR_PROMPT:-woman}
CKPT=${CKPT:-/workspace/SAM 3.1/sam3.1_multiplex.pt}
TAG=${TAG:-sam31_mcbyte}
D="$ROOT/$STEM/sam31_det_${PROMPTS//,/_}"; O="$ROOT/$STEM/$TAG"; mkdir -p "$O"

. "${VENV_DET:-/root/venv31}/bin/activate"
SEQ="$ROOT/$STEM/seq/$STEM"
python3 -c "import sys; sys.path.insert(0,'$CODE'); from common import extract_frames; extract_frames('$V','$SEQ',${MAXS:-0})"
META="$SEQ/frames_meta.json"

# 1) per-frame detections, no tracking (reused if present: the detector is deterministic per setting)
[ -f "$D/detect_meta.json" ] || python3 -W ignore "$CODE/adapters/sam31_detect.py" --frames-meta "$META" \
    --out "$D" --text "$PROMPTS" --checkpoint "$CKPT" --score-threshold-detection "${DET_THRESH:-0.25}" \
    2>&1 | tr '\r' '\n' | grep --line-buffered -E '^\[sam31-det\]|Error'

# 2) McByte identities + per-track gender vote
. "${VENV_MCB:-/root/venv_mcb}/bin/activate"
MM=""; [ "${MASK_MANAGER:-1}" = "0" ] && MM="--no-mask-manager"
[ -n "$MIN_SHARE" ] && MM="$MM --min-share $MIN_SHARE --blur-label $BLUR_PROMPT"
python3 -W ignore "$CODE/adapters/sam31_mcbyte.py" --frames-meta "$META" --dets "$D/masks.jsonl" --out "$O" \
    --merge-iou "${MERGE_IOU:-0.5}" --high-conf-det-threshold "${HIGH_CONF:-0.4}" \
    --track-activation-threshold "${HIGH_CONF:-0.4}" --lost-track-buffer "${LOST_BUFFER:-30}" $MM \
    2>&1 | tr '\r' '\n' | grep --line-buffered -E '^\[sam31-mcbyte\]|Error'

# 3) escape/conflict numbers against a tracked `person` control, and the videos (outlines = control)
. "${VENV_DET:-/root/venv31}/bin/activate"
EVAL="$O/masks.jsonl"
if [ -n "$CONTROL" ] && [ -f "$CONTROL" ]; then
  EVAL="$O/masks_with_control.jsonl"
  { grep '"prompt": "person"' "$CONTROL"; cat "$O/masks.jsonl"; } > "$EVAL"
  python3 "$CODE/sam3_gender_report.py" --masks "$EVAL" --frames-meta "$META" --out "$O/gender" \
    --blur-prompt "$BLUR_PROMPT" --extra-prompts child
fi
python3 "$CODE/render_gender.py" --frames-meta "$META" --masks "$EVAL" --out "$O/${STEM}_${TAG}_labels.mp4" \
  --blur-out "$O/${STEM}_${TAG}_blur.mp4" --sbs-out "$O/${STEM}_${TAG}_sbs.mp4" --blur-prompt "$BLUR_PROMPT" \
  --label "SAM 3.1 det + McByte"
echo "=== $STEM done -> $O"
