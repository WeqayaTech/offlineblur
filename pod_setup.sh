#!/bin/bash
# OfflineBlur — one-shot pod bootstrap. Run once on a fresh pod; weights and outputs live on the
# network volume (/workspace) so they survive the pod. Usage: bash pod_setup.sh
set -eo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1      # RunPod images use Debian's externally-managed system Python; torch lives there, so install beside it
export HF_HOME=${HF_HOME:-/workspace/hf}
export OFFLINEBLUR_WEIGHTS=${OFFLINEBLUR_WEIGHTS:-/workspace/offlineblur/weights}
mkdir -p "$HF_HOME" "$OFFLINEBLUR_WEIGHTS" /workspace/offlineblur/out /workspace/offlineblur/videos
cd "$(dirname "$0")"

echo "== GPU / torch"
# Blackwell GPUs (RTX PRO 6000 / 5090 / B200, compute capability 12.x) need the cu128 torch build:
# the stock RunPod torch stops at sm_90 and every kernel fails. Hopper/Ampere (H100/A100) keep the stock torch.
NEED_CU128=$(python3 -c "import torch; print(int(torch.cuda.get_device_capability(0)[0] >= 12 and 'sm_120' not in torch.cuda.get_arch_list()))")
if [ "$NEED_CU128" = "1" ]; then
  echo "Blackwell GPU without sm_120 kernels — installing torch/torchvision from the cu128 index"
  pip install -q -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
fi
python3 - <<'PY'
import torch
cc = torch.cuda.get_device_capability(0)
print(torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "sm", cc, "arch list", torch.cuda.get_arch_list()[-3:])
x = torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")   # a real kernel, not just is_available()
print("cuda kernel ok", torch.isfinite(x).all().item())
PY

echo "== system packages"
apt-get update -qq && apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 build-essential > /dev/null
ffmpeg -version | head -1

echo "== python packages"
pip install -q --upgrade pip
pip install -q cython numpy
pip install -q -r requirements.txt
# onnxruntime: the default PyPI wheel (>=1.23) is built for CUDA 13; torch here ships CUDA 12 libs. Pin the CUDA 12
# build and drop the CPU-only package that insightface pulls in (it would shadow the GPU one).
pip uninstall -y -q onnxruntime onnxruntime-gpu 2>/dev/null || true
pip install -q "onnxruntime-gpu==1.22.0"
# pip may have swapped torch for a CPU/cu12x build while resolving deps; make sure the GPU build survived
python3 -c "import torch; assert torch.cuda.is_available(); cc=torch.cuda.get_device_capability(0); assert cc[0] < 12 or 'sm_120' in torch.cuda.get_arch_list(), 'torch lost its sm_120 kernels - rerun: pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128'"

echo "== weights (cached on the volume)"
cd "$OFFLINEBLUR_WEIGHTS"
python3 - <<'PY'
from ultralytics import YOLO
YOLO("yolo11x-seg.pt")                                   # downloads into the weights dir (cwd)
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-VL-7B-Instruct")
snapshot_download("facebook/dinov2-base")
import onnxruntime as ort
ort.preload_dlls()
from insightface.app import FaceAnalysis
fa = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
fa.prepare(ctx_id=0, det_size=(640, 640))
print("face providers:", fa.models["detection"].session.get_providers())
import torch
from boxmot.reid.core.runtime import ReID
from pathlib import Path
ReID(weights=Path("osnet_x1_0_msmt17.pt"), device=torch.device("cuda"))          # OSNet re-id weights (Google Drive via gdown)
print("weights ready")
PY

echo "== sanity"
python3 - <<'PY'
import cv2, ultralytics, transformers, insightface, onnxruntime as ort, pycocotools, boxmot
print("cv2", cv2.__version__, "ultralytics", ultralytics.__version__, "transformers", transformers.__version__,
      "insightface", insightface.__version__, "boxmot", boxmot.__version__, "onnxruntime", ort.__version__)
PY
echo "== setup done. Put videos in /workspace/offlineblur/videos and run: bash run.sh /workspace/offlineblur/videos/<clip>.mp4"
