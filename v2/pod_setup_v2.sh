#!/bin/bash
# OfflineBlur v2 — one-shot pod bootstrap (1x H100 SXM 80 GB or A100 SXM 80 GB, RunPod PyTorch image).
# Clones + compiles MeMOTR and MOTRv2, downloads their checkpoints and the giant backbone.  Usage: bash pod_setup_v2.sh
set -eo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1   # xet downloads fail with "disk quota exceeded" on RunPod volumes
export HF_HOME=${HF_HOME:-/root/hf}
export V2_ROOT=${V2_ROOT:-/root/offlineblur_v2}
export V2_WEIGHTS=${V2_WEIGHTS:-$V2_ROOT/weights}
mkdir -p "$HF_HOME" "$V2_WEIGHTS" "$V2_ROOT/out" "$V2_ROOT/videos" "$V2_ROOT/third_party" "$V2_ROOT/data" "$V2_ROOT/manifests"
cd "$(dirname "$0")"

echo "== GPU / torch"
python3 - <<'PY'
import torch
cc = torch.cuda.get_device_capability(0)
print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "sm", cc)
assert cc[0] < 12, "Blackwell GPU: the Deformable-Attention CUDA op of MeMOTR/MOTRv2 needs a stock Hopper/Ampere torch — use an H100/A100 pod"
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda"); print("cuda kernel ok", torch.isfinite(x).all().item())
PY

echo "== system packages"
apt-get update -qq && apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 build-essential ninja-build git > /dev/null
ffmpeg -version | head -1

echo "== python packages"
pip install -q --upgrade pip
pip install -q -r requirements_v2.txt
python3 -c "import torch; assert torch.cuda.is_available(), 'pip replaced torch with a CPU build - reinstall the image torch'"

echo "== MeMOTR (tracking branch, long-term memory)"
cd "$V2_ROOT/third_party"
[ -d MeMOTR ] || git clone -q https://github.com/MCG-NJU/MeMOTR.git
cd MeMOTR/models/ops
# old Deformable-DETR op source: modernise the two deprecated macros so it compiles against torch 2.x
sed -i 's/AT_ASSERTM/TORCH_CHECK/g; s/\.type()\.is_cuda()/.is_cuda()/g' src/cuda/*.cu src/cpu/*.cpp src/*.h 2>/dev/null || true
if ! python3 -c "import torch, MultiScaleDeformableAttention" 2>/dev/null; then
  python3 setup.py build install > "$V2_ROOT/memotr_ops_build.log" 2>&1 || { tail -30 "$V2_ROOT/memotr_ops_build.log"; echo "MeMOTR op build failed (see log)"; exit 1; }
fi
python3 -c "import torch, MultiScaleDeformableAttention; print('MeMOTR deformable-attention op ok')"
cd "$V2_WEIGHTS"
[ -f memotr_mot17.pth ] || gdown -q "https://drive.google.com/uc?id=1MPZJfP91Pb1ThnX5dvxZ7tcjDH8t9hew" -O memotr_mot17.pth
[ -f memotr_dancetrack.pth ] || gdown -q "https://drive.google.com/uc?id=1_Xh-TDwwDIeacVEywwlYNvyRmhTKB5K2" -O memotr_dancetrack.pth
ls -la memotr_*.pth

echo "== MOTRv2 (tracking branch, proposal anchors)"
cd "$V2_ROOT/third_party"
[ -d MOTRv2 ] || git clone -q https://github.com/megvii-research/MOTRv2.git
cd MOTRv2/models/ops
sed -i 's/AT_ASSERTM/TORCH_CHECK/g; s/\.type()\.is_cuda()/.is_cuda()/g' src/cuda/*.cu src/cpu/*.cpp src/*.h 2>/dev/null || true
# both repos install a module of the same name; MOTRv2's build is kept in-tree (PYTHONPATH) so they don't collide
python3 setup.py build_ext --inplace > "$V2_ROOT/motrv2_ops_build.log" 2>&1 || { tail -30 "$V2_ROOT/motrv2_ops_build.log"; echo "MOTRv2 op build failed (see log) - MeMOTR still works"; }
cd "$V2_WEIGHTS"
[ -f motrv2_dancetrack.pth ] || gdown -q "https://drive.google.com/uc?id=1EA4lndu2yQcVgBKR09KfMe5efbf631Th" -O motrv2_dancetrack.pth || echo "motrv2 checkpoint download failed (Google Drive quota?) - download manually to $V2_WEIGHTS/motrv2_dancetrack.pth"
python3 - <<'PY'
from ultralytics import YOLO
YOLO("yolo11x.pt")   # proposal detector for --tracker motrv2
YOLO("yolo11x-seg.pt")   # instance masks for the blur renderer (blur.py)
PY

echo "== giant backbone (cached in HF_HOME)"
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("facebook/dinov2-giant", allow_patterns=["*.safetensors", "*.json", "*.txt"])
import torch, transformers
from transformers import Dinov2Model
m = Dinov2Model.from_pretrained("facebook/dinov2-giant", torch_dtype=torch.float16).cuda().eval()
x = torch.randn(1, 3, 14 * 40, 14 * 64, device="cuda", dtype=torch.float16)
with torch.no_grad(): y = m(pixel_values=x).last_hidden_state
print("dinov2-giant ok", tuple(y.shape), "=", 1 + 40 * 64, "tokens")
PY

echo "== sanity"
cd "$(dirname "$0")"
python3 -c "import sys; sys.path.insert(0,'stp'); import backbone, roi_attention, aggregator, tracker, demographics, frames, render; print('stp modules import ok')"
echo "== done. Next: (1) build manifests + train the head:  bash train_head.sh   (2) run a clip:  bash run_v2.sh /root/offlineblur_v2/videos/clip.mp4"
