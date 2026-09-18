#!/bin/bash
# Tracking bake-off pod bootstrap. 1x A40/L40S 48GB or A100 80GB, RunPod PyTorch 2.x/CUDA 12.x, Ampere/Ada
# (NOT Blackwell — the deformable-attention op fails on stock torch there). Everything on container disk.
#   bash pod_setup_tracking.sh
set -eo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-/root/hf}
ROOT=${TRACK_ROOT:-/root/tracking}; W=$ROOT/weights
mkdir -p "$HF_HOME" "$W" "$ROOT/out" "$ROOT/videos"
cd "$(dirname "$0")"

echo "== GPU / torch"
python3 - <<'PY'
import torch
cc = torch.cuda.get_device_capability(0)
print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "sm", cc)
assert cc[0] < 12, "Blackwell GPU: the MOTIP deformable-attention op needs a stock Hopper/Ampere/Ada torch - use A40/L40S/A100"
PY

echo "== system + python packages"
apt-get update -qq && apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 build-essential ninja-build git > /dev/null
pip install -q --upgrade pip
pip install -q "transformers<4.57" accelerate huggingface_hub gdown einops pyyaml scipy tqdm opencv-python-headless \
  numpy pillow ultralytics boxmot yt-dlp lap cython pycocotools
python3 -c "import torch; assert torch.cuda.is_available()"

echo "== MOTIP (end-to-end transformer, the MOTRv3 stand-in)"
cd "$ROOT"
[ -d MOTIP ] || git clone -q https://github.com/MCG-NJU/MOTIP.git
cd MOTIP/models/ops 2>/dev/null && { sed -i 's/AT_ASSERTM/TORCH_CHECK/g; s/\.type()\.is_cuda()/.is_cuda()/g' src/cuda/*.cu src/cpu/*.cpp src/*.h 2>/dev/null || true; python3 setup.py build install > "$ROOT/motip_ops.log" 2>&1 || { tail -25 "$ROOT/motip_ops.log"; echo "MOTIP op build failed"; exit 1; }; } || echo "(no models/ops in MOTIP layout - check repo)"
cd "$W"
for f in r50_deformable_detr_coco.pth r50_deformable_detr_motip_dancetrack.pth; do
  [ -f "$f" ] || wget -q "https://github.com/MCG-NJU/MOTIP/releases/download/v0.1/$f" -O "$f"
done
mkdir -p "$ROOT/MOTIP/pretrains" && ln -sfn "$W/r50_deformable_detr_coco.pth" "$ROOT/MOTIP/pretrains/r50_deformable_detr_coco.pth"
ls -la r50_deformable_detr_motip_dancetrack.pth

echo "== detector + ReID for BoT-SORT"
python3 - <<'PY'
from ultralytics import YOLO
YOLO("yolo11x.pt")
PY
mv -f yolo11x.pt "$W/" 2>/dev/null || true
# boxmot auto-downloads the CLIP ReID weight on first use; warm it
python3 - <<'PY'
try:
    from boxmot import BotSort
    print("boxmot BotSort import ok")
except Exception as e:
    print("boxmot import note:", e)
PY
echo "== done. Next: bash get_clips.sh ; then: bash run_bakeoff.sh $ROOT/videos/<clip>.mp4"
