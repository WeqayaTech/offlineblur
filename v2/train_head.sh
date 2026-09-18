#!/bin/bash
# Fine-tune the demographic head. Put the datasets under $V2_ROOT/data first (UTKFace/, adience/, celeba/ - see stp/manifest_builders.py).
set -eo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1 HF_HOME=${HF_HOME:-/root/hf}
V2_ROOT=${V2_ROOT:-/root/offlineblur_v2}; W=${V2_WEIGHTS:-$V2_ROOT/weights}; D=$V2_ROOT/data; M=$V2_ROOT/manifests
CODE="$(cd "$(dirname "$0")" && pwd)/stp"
mkdir -p "$M"
# datasets: either Kaggle copies under $D (see stp/manifest_builders.py) or the Hugging Face export below (default)
[ -f "$M/all.csv" ] || python3 "$CODE/hf_export.py" --root "$D" --manifests "$M" $HF_ARGS
python3 -u "$CODE/train_demographics.py" --manifest "$M/all.csv" --out "$W/demo_head.pt" --backbone "${BACKBONE:-dinov2-giant}" \
  --epochs "${EPOCHS:-8}" --batch "${BATCH:-48}" $TRAIN_ARGS 2>&1 | tee "$W/train_head.log"
