#!/bin/bash
# One-time setup of a GPU pod for the RF-DETR + EdgeTAM + PE-Core pipeline (fast_blur_edgetam.py, dist_edgetam.py).
# Everything on the container disk; copy the tracking/ folder to /root/tracking first.
#   bash /root/tracking/pod_setup_edgetam.sh
set -e
export HF_HUB_DISABLE_XET=1
python3 -m venv /root/venv_fast
. /root/venv_fast/bin/activate
pip install -q -U pip
pip install -q torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -q "rfdetr==1.10.1" "trackers==2.6.0" "open_clip_torch==3.3.0" pycocotools opencv-python-headless \
    "numpy<2.3" timm psutil "hydra-core>=1.3.2" "iopath>=0.1.10"
[ -d /root/EdgeTAM ] || git clone -q https://github.com/facebookresearch/EdgeTAM.git /root/EdgeTAM
SAM2_BUILD_CUDA=0 pip install -q --no-deps -e /root/EdgeTAM          # the CUDA extension only fills mask holes
# upstream bug: the memory encoder's perceiver breaks with more than one object per batch (.view on an expanded
# tensor); .reshape gives the same result
sed -i "s/expand(B, -1, -1).view(/expand(B, -1, -1).reshape(/" /root/EdgeTAM/sam2/modeling/perceiver.py
# download and warm every model once (RF-DETR 2XL, EdgeTAM, PE-Core-L)
cd /root && python -W ignore -c "
import sys; sys.path.insert(0, '/root/tracking')
import argparse; from fast_blur_edgetam import EdgeTAMEngine, add_args
ap = argparse.ArgumentParser(); add_args(ap); e = EdgeTAMEngine(ap.parse_args([]))
import torch; print('ready:', torch.cuda.device_count(), 'x', torch.cuda.get_device_name(0), f'load {e.load_s:.1f} s')"
