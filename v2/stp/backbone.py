#!/usr/bin/env python3
"""Phase 1 — shared giant-backbone feature extraction on the full frame.

One >1B-parameter vision transformer sees the whole frame; nothing is cropped at pixel level. To keep
a 60 px background person legible the token grid is made *dense*: the frame is resampled so that one
14 px patch covers `feat_stride` source pixels (default 8 → a 60 px person is ~8x15 tokens), and the
resampled frame is processed in overlapping tiles (attention cost stays bounded at 4K). Tiles that
contain no tracked person are skipped.

Backbones (all public weights, downloaded once into HF_HOME):
  dinov2-giant   facebook/dinov2-giant           1.1 B params, ViT-g/14, self-supervised dense features (default)
  eva02-giant    timm eva_giant_patch14_336      1.0 B params, EVA-CLIP-g/14 fine-tuned on IN-1k
  internimage-h  OpenGVLab/internimage_h_22kto1k_640 — hierarchical CNN-transformer hybrid; its finest public
                 output is stride-32 pooled features, i.e. not dense enough for 60 px people. Not wired in;
                 use one of the ViT-g variants (they *are* the "ViT-Giant" option of the blueprint).

    feat = GiantBackbone("dinov2-giant").feature_map(bgr, boxes)   # torch.HalfTensor [C, Hf, Wf] on GPU
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OPENAI_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
OPENAI_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

BACKBONES = {
    "dinov2-giant": {"hf": "facebook/dinov2-giant", "patch": 14, "dim": 1536, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "dinov2-large": {"hf": "facebook/dinov2-large", "patch": 14, "dim": 1024, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "eva02-giant": {"timm": "eva_giant_patch14_336.clip_ft_in1k", "patch": 14, "dim": 1408,
                    "mean": OPENAI_CLIP_MEAN, "std": OPENAI_CLIP_STD},
}


class GiantBackbone:
    def __init__(self, name="dinov2-giant", device="cuda", dtype=torch.float16, feat_stride=8,
                 tile_tokens=64, overlap_tokens=8, batch_tiles=8):
        if name == "internimage-h":
            raise SystemExit("internimage-h: hierarchical backbone without a dense stride-8 output; use dinov2-giant "
                             "or eva02-giant (see backbone.py docstring)")
        if name not in BACKBONES:
            raise SystemExit(f"unknown backbone {name}; choose from {list(BACKBONES)}")
        self.cfg = BACKBONES[name]
        self.name, self.device, self.dtype = name, device, dtype
        self.patch, self.dim = self.cfg["patch"], self.cfg["dim"]
        self.feat_stride = float(feat_stride)
        self.tile_tokens, self.overlap_tokens, self.batch_tiles = tile_tokens, overlap_tokens, batch_tiles
        if "hf" in self.cfg:
            from transformers import Dinov2Model
            self.model = Dinov2Model.from_pretrained(self.cfg["hf"], torch_dtype=dtype, attn_implementation="sdpa")
            self._kind = "hf"
        else:
            import timm
            self.model = timm.create_model(self.cfg["timm"], pretrained=True, num_classes=0, dynamic_img_size=True)
            self.model = self.model.to(dtype)
            self._kind = "timm"
        self.model.eval().to(device)
        self.mean = torch.tensor(self.cfg["mean"], device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(self.cfg["std"], device=device, dtype=dtype).view(1, 3, 1, 1)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[backbone] {name}: {n_params:.2f} B params, patch {self.patch}, feature stride {feat_stride} px, dim {self.dim}")

    # ---- raw token forward ---------------------------------------------------------------------
    @torch.no_grad()
    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B,3,H,W] float in 0..1, H and W multiples of patch → [B, H/p, W/p, C]."""
        x = (x.to(self.dtype) - self.mean) / self.std
        B, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        if self._kind == "hf":
            out = self.model(pixel_values=x).last_hidden_state
            n_prefix = out.shape[1] - h * w  # CLS (+ register tokens on -reg variants)
            tok = out[:, n_prefix:, :]
        else:
            out = self.model.forward_features(x)
            n_prefix = out.shape[1] - h * w
            tok = out[:, n_prefix:, :]
        return tok.reshape(B, h, w, -1)

    # ---- dense tiled feature map ---------------------------------------------------------------
    def grid_size(self, w_src: int, h_src: int):
        """Token grid (wt, ht) and the exact resample scale used so the grid is integral."""
        wt = max(1, int(round(w_src / self.feat_stride)))
        ht = max(1, int(round(h_src / self.feat_stride)))
        return wt, ht

    def to_tokens(self, box_xyxy, w_src, h_src):
        """Source-pixel box → token-unit box (float) on the grid of `grid_size`."""
        wt, ht = self.grid_size(w_src, h_src)
        sx, sy = wt / w_src, ht / h_src
        return [box_xyxy[0] * sx, box_xyxy[1] * sy, box_xyxy[2] * sx, box_xyxy[3] * sy]

    @torch.no_grad()
    def feature_map(self, bgr: np.ndarray, boxes=None, margin=0.25) -> torch.Tensor:
        """Full-frame dense features [C, ht, wt] (half precision, on device). With `boxes` (source xyxy),
        only tiles that intersect an (expanded) box are computed; the rest stay zero."""
        h_src, w_src = bgr.shape[:2]
        wt, ht = self.grid_size(w_src, h_src)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        big = cv2.resize(rgb, (wt * self.patch, ht * self.patch), interpolation=cv2.INTER_CUBIC if wt * self.patch > w_src else cv2.INTER_AREA)
        img = torch.from_numpy(big).to(self.device).permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0

        feat = torch.zeros(self.dim, ht, wt, device=self.device, dtype=torch.float32)
        wsum = torch.zeros(1, ht, wt, device=self.device, dtype=torch.float32)
        T, O = self.tile_tokens, self.overlap_tokens
        step = max(1, T - O)
        ys = list(range(0, max(1, ht - T + 1), step)) + ([max(0, ht - T)] if ht > T else [])
        xs = list(range(0, max(1, wt - T + 1), step)) + ([max(0, wt - T)] if wt > T else [])
        ys, xs = sorted(set(ys)), sorted(set(xs))

        need = None
        if boxes is not None:
            need = np.zeros((ht, wt), dtype=bool)
            for b in boxes:
                tb = self.to_tokens(b, w_src, h_src)
                mw, mh = (tb[2] - tb[0]) * margin, (tb[3] - tb[1]) * margin
                x0, y0 = int(max(0, math.floor(tb[0] - mw))), int(max(0, math.floor(tb[1] - mh)))
                x1, y1 = int(min(wt, math.ceil(tb[2] + mw))), int(min(ht, math.ceil(tb[3] + mh)))
                need[y0:y1, x0:x1] = True

        tiles = []
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(ht, y0 + T), min(wt, x0 + T)
                if need is not None and not need[y0:y1, x0:x1].any():
                    continue
                tiles.append((y0, x0, y1, x1))
        if not tiles:
            return feat.to(self.dtype)

        win_cache = {}
        for i in range(0, len(tiles), self.batch_tiles):
            chunk = tiles[i:i + self.batch_tiles]
            # tiles at the border can be smaller; group by size to batch
            by_size = {}
            for t in chunk:
                by_size.setdefault((t[2] - t[0], t[3] - t[1]), []).append(t)
            for (th, tw), ts in by_size.items():
                x = torch.cat([img[:, :, t[0] * self.patch:t[2] * self.patch, t[1] * self.patch:t[3] * self.patch] for t in ts], 0)
                tok = self.tokens(x).float()  # [n, th, tw, C]
                if (th, tw) not in win_cache:
                    wy = torch.hann_window(th + 2, periodic=False, device=self.device)[1:-1] if th > 1 else torch.ones(1, device=self.device)
                    wx = torch.hann_window(tw + 2, periodic=False, device=self.device)[1:-1] if tw > 1 else torch.ones(1, device=self.device)
                    win_cache[(th, tw)] = (wy[:, None] * wx[None, :]).clamp_min(1e-3)[None]  # [1, th, tw]
                win = win_cache[(th, tw)]
                for t, tk in zip(ts, tok):
                    feat[:, t[0]:t[2], t[1]:t[3]] += tk.permute(2, 0, 1) * win
                    wsum[:, t[0]:t[2], t[1]:t[3]] += win
        feat = feat / wsum.clamp_min(1e-6)
        return feat.to(self.dtype)

    @torch.no_grad()
    def batch_feature_maps(self, rgb_uint8_list):
        """Training helper: list of RGB uint8 images (each side a multiple of patch) → list of [C,h,w] maps.
        Images are padded to the batch max and un-padded afterwards (padding tokens never enter the ROI)."""
        H = max(im.shape[0] for im in rgb_uint8_list)
        W = max(im.shape[1] for im in rgb_uint8_list)
        x = torch.zeros(len(rgb_uint8_list), 3, H, W, device=self.device, dtype=self.dtype)
        for i, im in enumerate(rgb_uint8_list):
            x[i, :, :im.shape[0], :im.shape[1]] = torch.from_numpy(im).to(self.device).permute(2, 0, 1).to(self.dtype) / 255.0
        tok = self.tokens(x)  # [B, H/p, W/p, C]
        outs = []
        for i, im in enumerate(rgb_uint8_list):
            h, w = im.shape[0] // self.patch, im.shape[1] // self.patch
            outs.append(tok[i, :h, :w, :].permute(2, 0, 1).contiguous())
        return outs
