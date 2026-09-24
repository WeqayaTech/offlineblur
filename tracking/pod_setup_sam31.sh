#!/bin/bash
# SAM 3.1 (Object Multiplex) pod bootstrap. Separate venv from SAM 3 (transformers) and from the
# bake-off env: SAM 3.1 has NO transformers integration, it runs from github.com/facebookresearch/sam3.
#
# Weights live on the network volume and are loaded from there in place — nothing is downloaded:
#   /workspace/SAM 3.1/sam3.1_multiplex.pt   (3.5 GB, the facebook/sam3.1 HF checkpoint)
# The config/tokenizer json files next to it are the SAM 3 *transformers* configs and are not used by
# the sam3 repo (it ships its own BPE vocab).
#
#   bash pod_setup_sam31.sh          # then: bash run_sam31.sh <clip.mp4>
set -eo pipefail
VENV=${VENV:-/root/venv31}
REPO=${SAM3_REPO:-/root/sam3_repo}
REF=${SAM3_REF:-2345a4a}   # the commit these adapters were verified against (2026-09-23)
CKPT=${CKPT:-/workspace/SAM 3.1/sam3.1_multiplex.pt}

[ -f "$CKPT" ] || { echo "FATAL: no checkpoint at $CKPT"; exit 1; }
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null; }

# the image python is PEP 668 "externally managed"; a venv that sees the image torch avoids both that
# and any chance of pip replacing the image's torch/torchvision pair
[ -d "$VENV" ] || python3 -m venv --system-site-packages "$VENV"
. "$VENV/bin/activate"
[ -d "$REPO" ] || git clone -q https://github.com/facebookresearch/sam3.git "$REPO"
git -C "$REPO" fetch -q origin && git -C "$REPO" checkout -q "$REF"

python3 - <<'PY' > /tmp/sam31_constraints.txt
import torch, torchvision
print(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}")
PY
echo "numpy<2" >> /tmp/sam31_constraints.txt
pip install -q -c /tmp/sam31_constraints.txt -e "$REPO"
# installed one by one on purpose: a combined install failed metadata generation on py3.12;
# opencv 4.10 because >=4.11 wheels require numpy 2, which sam3 pins below
for p in "opencv-python-headless<4.11" einops psutil pycocotools; do
  pip install -q -c /tmp/sam31_constraints.txt "$p"
done

python3 -W ignore - <<'PY'
import torch, numpy, sam3
cc = torch.cuda.get_device_capability(0)
print(f"{torch.cuda.get_device_name(0)} sm_{cc[0]}{cc[1]}  torch {torch.__version__}  numpy {numpy.__version__}")
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok:", float((x @ x).float().abs().mean()) > 0)
if cc[0] != 9:
    print("not Hopper: FlashAttention 3 unavailable, sam31_track.py runs with fa3 off (its default)")
PY
echo "== ready: $VENV  repo $REPO@$REF  ckpt $CKPT"
