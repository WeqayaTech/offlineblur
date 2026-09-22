#!/bin/bash
# SAM 3 pod bootstrap — standalone, deliberately NOT the same env as pod_setup_tracking.sh.
#
# Why separate: the bake-off env pins `transformers<4.57` because MOTIP's compiled deformable-attention
# op is built against that image's torch. SAM 3 needs a much newer transformers (Sam3VideoModel) and
# therefore a newer torch. SAM 3 itself compiles nothing — it is pure PyTorch — so unlike MOTIP it is
# fine on Blackwell (sm_120) as long as the image's torch actually ships sm_120 kernels, i.e. a
# CUDA 12.8+ build. That is checked below before anything else is installed.
#
# Requires gated access to facebook/sam3: accept the licence at https://huggingface.co/facebook/sam3
# with the same account as your token, then `export HF_TOKEN=hf_...` before running this.
#
#   export HF_TOKEN=hf_...
#   bash pod_setup_sam3.sh
set -eo pipefail
export PIP_BREAK_SYSTEM_PACKAGES=1 HF_HUB_DISABLE_XET=1
export HF_HOME=${HF_HOME:-/root/hf}
ROOT=${TRACK_ROOT:-/root/tracking}
mkdir -p "$HF_HOME" "$ROOT/out" "$ROOT/videos" "$ROOT/weights"

echo "== GPU / torch"
python3 - <<'PY'
import torch
cc = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"{name}  sm_{cc[0]}{cc[1]}  torch {torch.__version__}  cuda {torch.version.cuda}")
archs = torch.cuda.get_arch_list()
print("torch arch list:", archs)
tag = f"sm_{cc[0]}{cc[1]}"
if tag not in archs:
    raise SystemExit(
        f"FATAL: this torch has no {tag} kernels ({archs}).\n"
        f"On a Blackwell card (RTX PRO 4500/6000, 5090, B200) pick a RunPod template with "
        f"PyTorch 2.7+/CUDA 12.8, or: pip install --force-reinstall torch torchvision "
        f"--index-url https://download.pytorch.org/whl/cu128")
# prove kernels actually run, not just that the arch is listed
x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().abs().mean()))
PY

echo "== system packages"
apt-get update -qq && apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 git > /dev/null

echo "== python packages"
pip install -q --upgrade pip
pip install -q --upgrade "transformers>=4.58" accelerate huggingface_hub safetensors
pip install -q numpy pillow opencv-python-headless pycocotools tqdm yt-dlp ultralytics

echo "== transformers has SAM 3?"
python3 - <<'PY' || NEED_MAIN=1
from transformers import Sam3VideoModel, Sam3VideoProcessor, Sam3VideoConfig  # noqa: F401
import transformers
print("Sam3Video* present in transformers", transformers.__version__)
PY
if [ "${NEED_MAIN:-0}" = "1" ]; then
  echo "== released transformers lacks Sam3Video*, installing from git main"
  pip install -q --upgrade "git+https://github.com/huggingface/transformers.git"
  python3 -c "from transformers import Sam3VideoModel; import transformers; print('ok', transformers.__version__)"
fi

echo "== hugging face auth (facebook/sam3 is gated)"
if [ -n "$HF_TOKEN" ]; then
  python3 - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("logged in")
PY
else
  echo "HF_TOKEN not set — if facebook/sam3 download 401/403s, set it and rerun"
fi

echo "== fetch facebook/sam3 weights"
python3 - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("facebook/sam3")
print("sam3 cached at", p)
PY

echo "== done. Next: bash get_clips.sh ; then bash run_sam3_gender.sh $ROOT/videos/<clip>.mp4"
