#!/bin/bash
# RF-DETR-Seg person masks -> McByte identities -> per-track OpenCLIP gender (best-K crops, batched).
# Each stage answers one question: where is every person (pixel-exact) / who is who / is this identity a woman.
#   bash run_rfdetr_mcbyte_clip.sh /root/tracking/videos/clip.mp4 [out_root]
# env: RF_MODEL=2XLarge  RF_THRESH=0.15 (low on purpose: McByte's second association)  HIGH_CONF=0.5
#      MASK_MANAGER=0 (1 = McByte SAM/Cutie masks; +82 s/300 frames on an L4 for a small coverage gain)
#      CLIP_MODEL=ViT-L-14-336  CLIP_PRETRAINED=openai  K=10  BLUR_MIN=0.25 (P(woman)+P(girl) per track)
#      CONTROL=<masks.jsonl with a `person` prompt> (outlines in the labels video + GT-free report only)
#      VENV=/root/venv_mcb  MAXS=0
# Measured on the 12 s trial clip, L4: detector 42 ms/frame, McByte (IoU) <1 s total, CLIP 11 s for 418 crops.
# BLUR_MIN 0.25 matched oracle labels on the 21 hand-labelled people of that clip — tuned on it, so
# re-check on new footage with hand labels + gender_gt_eval.py (labels are kept locally, not versioned).
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_rfdetr_mcbyte_clip.sh <video> [out_root]"; exit 1; }
. "${VENV:-/root/venv_mcb}/bin/activate"
export HF_HUB_DISABLE_XET=1
ROOT="${2:-/root/tracking/out}"; STEM=$(basename "${V%.*}")
CODE="$(cd "$(dirname "$0")" && pwd)"
RF_MODEL=${RF_MODEL:-2XLarge}
D="$ROOT/$STEM/rfdetr_${RF_MODEL}"; T="$ROOT/$STEM/rfdetr_mcbyte"; O="$ROOT/$STEM/rfdetr_mcbyte_clip"
mkdir -p "$O"
SEQ="$ROOT/$STEM/seq/$STEM"
python3 -c "import sys; sys.path.insert(0,'$CODE'); from common import extract_frames; extract_frames('$V','$SEQ',${MAXS:-0})"
META="$SEQ/frames_meta.json"

# 1) person masks per frame (rfdetr writes its weights to the working directory's .roboflow cache)
[ -f "$D/detect_meta.json" ] || (cd /root && python3 -W ignore "$CODE/adapters/rfdetr_seg_detect.py" \
    --frames-meta "$META" --out "$D" --model "$RF_MODEL" --threshold "${RF_THRESH:-0.15}" \
    2>&1 | grep --line-buffered -E '^\[rfdetr\]|Error')

# 2) identities
MM="--no-mask-manager"; [ "${MASK_MANAGER:-0}" = "1" ] && MM=""
python3 -W ignore "$CODE/adapters/sam31_mcbyte.py" --frames-meta "$META" --dets "$D/masks.jsonl" --out "$T" \
    --high-conf-det-threshold "${HIGH_CONF:-0.5}" --track-activation-threshold "${HIGH_CONF:-0.5}" $MM \
    2>&1 | tr '\r' '\n' | grep --line-buffered -E '^\[sam31-mcbyte\]|Error'

# 3) gender per identity
python3 -W ignore "$CODE/clip_track_gender.py" --frames-meta "$META" --tracks "$T/tracks.jsonl" --masks "$T/masks.jsonl" \
    --out "$O" --model "${CLIP_MODEL:-ViT-L-14-336}" --pretrained "${CLIP_PRETRAINED:-openai}" --k "${K:-10}" \
    --masked --blur-min "${BLUR_MIN:-0.25}" --save-crops 2>&1 | grep --line-buffered -E '^\[clip\]|Error'

# 4) videos (+ GT-free report when a person control is given)
EVAL="$O/masks.jsonl"
if [ -n "$CONTROL" ] && [ -f "$CONTROL" ]; then
  EVAL="$O/masks_with_control.jsonl"
  { grep '"prompt": "person"' "$CONTROL"; cat "$O/masks.jsonl"; } > "$EVAL"
  python3 "$CODE/sam3_gender_report.py" --masks "$EVAL" --frames-meta "$META" --out "$O/gender" --blur-prompt woman
fi
python3 "$CODE/render_gender.py" --frames-meta "$META" --masks "$EVAL" --out "$O/${STEM}_rfdetr_clip_labels.mp4" \
  --blur-out "$O/${STEM}_rfdetr_clip_blur.mp4" --sbs-out "$O/${STEM}_rfdetr_clip_sbs.mp4" --blur-prompt woman \
  --label "RF-DETR + McByte + CLIP"
echo "=== $STEM done -> $O"
