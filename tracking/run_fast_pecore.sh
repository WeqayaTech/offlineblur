#!/bin/bash
# Pipeline 11: the latest fast structure (fast_blur.py: RF-DETR-Seg 2XL -> McByte buffered IoU 0.5 / 2nd round 0.3,
# lost tracks kept 4 s -> per-track zero-shot gender) with the classifier swapped from OpenAI CLIP ViT-L/14-336 to
# Meta's Perception Encoder PE-Core-L/14-336 (open_clip `PE-Core-L-14-336`, pretrained `meta`). Detector and tracker
# are identical, so both runs produce the same tracks and every difference is the classifier's.
#   bash run_fast_pecore.sh /root/tracking/videos/clip.mp4 [out_root]
# env: PE_TAU=0.25 (P(woman)+P(girl) threshold for PE-Core)  CLIP_TAU=0.25  BATCH=8  ENCODER=libx264
#      GT=<hand-label json on the script RF-DETR+McByte track ids>  (optional: GT scoring + classifier_compare.py)
#      VENV=/root/venv_fast
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_fast_pecore.sh <video> [out_root]"; exit 1; }
. "${VENV:-/root/venv_fast}/bin/activate"
export HF_HUB_DISABLE_XET=1
ROOT="${2:-/root/tracking/out}"; STEM=$(basename "${V%.*}")
CODE="$(cd "$(dirname "$0")" && pwd)"
O="$ROOT/$STEM/fast_pecore"; mkdir -p "$O"
LATEST="--rf-model 2XLarge --batch ${BATCH:-8} --encoder ${ENCODER:-libx264} --lost-seconds 4 --iou biou --biou-buffer 0.5 --assoc2 0.3"

run() {   # name model pretrained tau title
  (cd /root && python3 -W ignore "$CODE/fast_blur.py" --video "$V" --out "$O/$1_blur.mp4" $LATEST \
      --clip-model "$2" --clip-pretrained "$3" --blur-min "$4" --debug-title "$5" \
      --dump-tracks "$O/$1.tracks.jsonl" --dump-masks "$O/$1.masks.jsonl" --debug-out "$O/$1_labels_masks_ids.mp4" \
      2>&1 | grep -vE "^(Warning|  warnings)|UserWarning" | tail -45)
}
run clip_vitl ViT-L-14-336 openai "${CLIP_TAU:-0.25}" "RF-DETR-Seg + McByte (BIoU) + CLIP ViT-L/14-336"
run pecore_l PE-Core-L-14-336 meta "${PE_TAU:-0.25}" "RF-DETR-Seg + McByte (BIoU) + PE-Core-L/14-336"

# product video: original | CLIP blur | PE-Core blur
ffmpeg -y -loglevel error -i "$V" -i "$O/clip_vitl_blur.mp4" -i "$O/pecore_l_blur.mp4" -filter_complex \
  "[0]scale=640:-2,drawtext=text='original':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black[a];\
   [1]scale=640:-2,drawtext=text='CLIP ViT-L':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black[b];\
   [2]scale=640:-2,drawtext=text='PE-Core-L':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black[c];\
   [a][b][c]hstack=3" -c:v libx264 -crf 20 -preset veryfast "$O/blur_original_clip_pecore.mp4" || true
# labels video: CLIP left | PE-Core right (same tracks, so only the tag/box colours can differ)
ffmpeg -y -loglevel error -i "$O/clip_vitl_labels_masks_ids.mp4" -i "$O/pecore_l_labels_masks_ids.mp4" \
  -filter_complex "[0][1]hstack=2" -c:v libx264 -crf 22 -preset veryfast "$O/labels_clip_left_pecore_right.mp4" || true

if [ -n "$GT" ] && [ -f "$GT" ]; then
  # GT ids belong to the SCRIPT pipeline's RF-DETR + McByte (IoU) tracks: rebuild that layer, then carry the labels
  # onto this run's tracks (reference = the CLIP run's own person layer; the PE run has identical tracks)
  S="$ROOT/$STEM/gt_layer"; SEQ="$ROOT/$STEM/seq/$STEM"
  python3 -c "import sys; sys.path.insert(0,'$CODE'); from common import extract_frames; extract_frames('$V','$SEQ',0)"
  [ -f "$S/rf/detect_meta.json" ] || (cd /root && python3 -W ignore "$CODE/adapters/rfdetr_seg_detect.py" \
      --frames-meta "$SEQ/frames_meta.json" --out "$S/rf" --model 2XLarge --threshold 0.15 2>&1 | grep -E '^\[rfdetr\]|Error')
  [ -f "$S/mcb/tracks.jsonl" ] || python3 -W ignore "$CODE/adapters/sam31_mcbyte.py" --frames-meta "$SEQ/frames_meta.json" \
      --dets "$S/rf/masks.jsonl" --out "$S/mcb" --high-conf-det-threshold 0.5 --track-activation-threshold 0.5 \
      --no-mask-manager 2>&1 | tr '\r' '\n' | grep -E '^\[sam31-mcbyte\]|Error'
  python3 "$CODE/gender_gt_eval.py" --gt "$GT" --gt-masks "$S/mcb/masks.jsonl" --reference "$O/clip_vitl.masks.jsonl" \
      --reference-prompt any --run "CLIP ViT-L=$O/clip_vitl.masks.jsonl" --run "PE-Core-L=$O/pecore_l.masks.jsonl" \
      --out "$O/gt_eval.json"
  python3 "$CODE/classifier_compare.py" --gt-eval "$O/gt_eval.json" --tracks "$O/clip_vitl.tracks.jsonl" \
      --tracks-b "$O/pecore_l.tracks.jsonl" --run "CLIP ViT-L=$O/clip_vitl_blur.json" \
      --run "PE-Core-L=$O/pecore_l_blur.json" --out "$O/classifier_compare.json"
fi
echo "=== $STEM done -> $O"
