#!/bin/bash
# Pipeline 12: pipeline 11 (RF-DETR-Seg 2XL -> McByte -> PE-Core-L) with McByte replaced by EdgeTAM
# (facebookresearch/EdgeTAM, on-device SAM 2) driven online by fast_blur_edgetam.py.
#   bash run_fast_edgetam.sh /root/tracking/videos/clip.mp4 [out_root]
# env: MCBYTE_RUN=<pipeline 11 pecore_l_blur.mp4 / _labels_masks_ids.mp4 dir> for the side-by-side videos
#      EXTRA="--new-track 0.5 --match-iou 0.3 ..."  VENV=/root/venv_fast
# EdgeTAM install (once): git clone https://github.com/facebookresearch/EdgeTAM /root/EdgeTAM &&
#   SAM2_BUILD_CUDA=0 pip install --no-deps -e /root/EdgeTAM && pip install hydra-core iopath
set -eo pipefail
V="$1"; [ -f "$V" ] || { echo "usage: bash run_fast_edgetam.sh <video> [out_root]"; exit 1; }
. "${VENV:-/root/venv_fast}/bin/activate"
export HF_HUB_DISABLE_XET=1
ROOT="${2:-/root/tracking/out}"; STEM=$(basename "${V%.*}")
CODE="$(cd "$(dirname "$0")" && pwd)"
O="$ROOT/$STEM/fast_edgetam"; mkdir -p "$O"
M="${MCBYTE_RUN:-$ROOT/$STEM/fast_pecore}"

(cd /root && python3 -W ignore "$CODE/fast_blur_edgetam.py" --video "$V" --out "$O/edgetam_pecore_blur.mp4" $EXTRA \
    --dump-tracks "$O/edgetam_pecore.tracks.jsonl" --debug-out "$O/edgetam_pecore_labels_masks_ids.mp4" \
    2>&1 | grep -vE "Warning|warn" | tail -45)

if [ -f "$M/pecore_l_blur.mp4" ]; then
  ffmpeg -y -loglevel error -i "$V" -i "$M/pecore_l_blur.mp4" -i "$O/edgetam_pecore_blur.mp4" -filter_complex \
    "[0]scale=640:-2,drawtext=text='original':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black[a];\
     [1]scale=640:-2,drawtext=text='McByte + PE-Core':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black[b];\
     [2]scale=640:-2,drawtext=text='EdgeTAM + PE-Core':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black[c];\
     [a][b][c]hstack=3" -c:v libx264 -crf 20 -preset veryfast "$O/blur_original_mcbyte_edgetam.mp4"
  ffmpeg -y -loglevel error -i "$M/pecore_l_labels_masks_ids.mp4" -i "$O/edgetam_pecore_labels_masks_ids.mp4" \
    -filter_complex "[0][1]hstack=2" -c:v libx264 -crf 22 -preset veryfast "$O/labels_mcbyte_left_edgetam_right.mp4"
fi
echo "=== $STEM done -> $O"
