#!/usr/bin/env python3
"""Phase 3 — demographic head: cross-attention ROI over the shared backbone feature map.

No pixel crop. For each tracked box the head gathers the backbone tokens that fall inside the box
(plus a margin), gives them a box-relative 2-D sine position, and lets a small set of learned queries
cross-attend to them (self-attention between queries, cross-attention to the ROI tokens, FFN; a few
layers). Pooled queries feed three heads:
  gender   2-way logits
  age      101-bin distribution (0..100, DLDL) → mean and std
  quality  1 logit trained to predict "this frame's answer is right" — the per-frame confidence the
           temporal aggregator weights by (together with the cross-attention entropy, which is low when
           the queries found something to focus on and high on a blank/occluded box)

    head = DemographicHead(in_dim=1536)
    out = head([feat_map, ...], [boxes_tok, ...])      # boxes in token units of that map
    out["gender_logits"] [N,2]  out["age_logits"] [N,101]  out["quality_logit"] [N]  out["entropy"] [N]  out["pooled"] [N,dim]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

AGE_BINS = 101


def sine_pos_2d(yx: torch.Tensor, dim: int, temperature=10000.0) -> torch.Tensor:
    """yx: [..., 2] coordinates in 0..1 → [..., dim] sine/cosine embedding (half for y, half for x)."""
    d = dim // 2
    freq = torch.arange(d // 2, device=yx.device, dtype=torch.float32)
    freq = temperature ** (2 * freq / d)
    out = []
    for k in range(2):
        v = yx[..., k:k + 1] * 2 * math.pi / freq
        out.append(torch.cat([v.sin(), v.cos()], -1))
    return torch.cat(out, -1)


class ROILayer(nn.Module):
    def __init__(self, dim, heads, ffn=4, dropout=0.0):
        super().__init__()
        self.sa = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ca = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ffn, dim))

    def forward(self, q, kv, kv_mask):
        h = self.n1(q)
        q = q + self.sa(h, h, h, need_weights=False)[0]
        h = self.n2(q)
        a, w = self.ca(h, kv, kv, key_padding_mask=kv_mask, need_weights=True, average_attn_weights=True)
        q = q + a
        q = q + self.ffn(self.n3(q))
        return q, w  # w: [B, Q, K]


class DemographicHead(nn.Module):
    def __init__(self, in_dim=1536, dim=512, n_queries=8, n_layers=3, heads=8, max_keys=4096, margin=0.15,
                 dropout=0.0):
        super().__init__()
        self.in_dim, self.dim, self.max_keys, self.margin = in_dim, dim, max_keys, margin
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim))
        self.queries = nn.Parameter(torch.randn(n_queries, dim) * 0.02)
        self.content = nn.Linear(dim, dim)  # ROI mean token → added to every query (content-aware init)
        self.layers = nn.ModuleList([ROILayer(dim, heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)
        self.gender = nn.Linear(dim, 2)
        self.age = nn.Linear(dim, AGE_BINS)
        self.quality = nn.Sequential(nn.Linear(dim + 1, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1))
        self.register_buffer("age_values", torch.arange(AGE_BINS, dtype=torch.float32), persistent=False)

    # ---- token gathering -----------------------------------------------------------------------
    def gather(self, feats, boxes_list):
        """feats: list of [C,h,w]; boxes_list: list of [n,4] token-unit xyxy → padded keys [N,K,C_in],
        positions [N,K,2] in 0..1 (box-relative), mask [N,K] (True = pad)."""
        keys, poss = [], []
        for feat, boxes in zip(feats, boxes_list):
            C, h, w = feat.shape
            for b in boxes.tolist():
                bw, bh = max(1e-3, b[2] - b[0]), max(1e-3, b[3] - b[1])
                x0 = int(max(0, math.floor(b[0] - self.margin * bw)))
                y0 = int(max(0, math.floor(b[1] - self.margin * bh)))
                x1 = int(min(w, math.ceil(b[2] + self.margin * bw)))
                y1 = int(min(h, math.ceil(b[3] + self.margin * bh)))
                if x1 <= x0:
                    x0, x1 = max(0, min(w - 1, x0)), max(1, min(w, x0 + 1))
                if y1 <= y0:
                    y0, y1 = max(0, min(h - 1, y0)), max(1, min(h, y0 + 1))
                ys = torch.arange(y0, y1, device=feat.device)
                xs = torch.arange(x0, x1, device=feat.device)
                n = len(ys) * len(xs)
                if n > self.max_keys:  # uniform stride subsample keeps the ROI shape
                    s = math.ceil(math.sqrt(n / self.max_keys))
                    ys, xs = ys[::s], xs[::s]
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                tok = feat[:, gy.reshape(-1), gx.reshape(-1)].t()  # [k, C]
                pos = torch.stack([(gy.reshape(-1).float() + 0.5 - b[1]) / bh,
                                   (gx.reshape(-1).float() + 0.5 - b[0]) / bw], -1)  # box-relative, margin → <0 or >1
                pos = (pos + self.margin) / (1 + 2 * self.margin)
                keys.append(tok)
                poss.append(pos)
        K = max(k.shape[0] for k in keys)
        N = len(keys)
        C = keys[0].shape[1]
        kv = torch.zeros(N, K, C, device=keys[0].device, dtype=keys[0].dtype)
        pp = torch.zeros(N, K, 2, device=keys[0].device, dtype=torch.float32)
        mask = torch.ones(N, K, device=keys[0].device, dtype=torch.bool)
        for i, (k, p) in enumerate(zip(keys, poss)):
            kv[i, :k.shape[0]] = k
            pp[i, :p.shape[0]] = p
            mask[i, :k.shape[0]] = False
        return kv, pp, mask

    def forward(self, feats, boxes_list):
        kv, pos, mask = self.gather(feats, boxes_list)
        kv = self.proj(kv.float()) + sine_pos_2d(pos, self.dim)
        N = kv.shape[0]
        valid = (~mask).float().unsqueeze(-1)
        content = (kv * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        q = self.queries.unsqueeze(0).expand(N, -1, -1) + self.content(content).unsqueeze(1)
        w = None
        for layer in self.layers:
            q, w = layer(q, kv, mask)
        pooled = self.norm(q).mean(1)
        # attention entropy of the last cross-attention, normalised by log(#valid keys): 0 = focused, 1 = flat
        n_valid = (~mask).sum(1).float().clamp_min(2.0)
        ent = -(w.clamp_min(1e-9) * w.clamp_min(1e-9).log()).sum(-1).mean(1) / n_valid.log()
        age_logits = self.age(pooled)
        gender_logits = self.gender(pooled)
        quality_logit = self.quality(torch.cat([pooled.detach(), ent.detach().unsqueeze(-1)], -1)).squeeze(-1)
        return {"gender_logits": gender_logits, "age_logits": age_logits, "quality_logit": quality_logit,
                "entropy": ent, "pooled": pooled}

    # ---- decoding --------------------------------------------------------------------------------
    def decode(self, out):
        p_gender = out["gender_logits"].softmax(-1)
        p_age = out["age_logits"].softmax(-1)
        mean = (p_age * self.age_values).sum(-1)
        var = (p_age * (self.age_values - mean.unsqueeze(-1)) ** 2).sum(-1)
        return {"p_female": p_gender[:, 1], "age_mean": mean, "age_std": var.clamp_min(1.0).sqrt(),
                "quality": out["quality_logit"].sigmoid(), "entropy": out["entropy"]}


def dldl_target(age: torch.Tensor, sigma=2.0) -> torch.Tensor:
    """Gaussian label distribution over 0..100 for each age (DLDL)."""
    v = torch.arange(AGE_BINS, device=age.device, dtype=torch.float32)
    d = torch.exp(-0.5 * ((v[None] - age[:, None]) / sigma) ** 2)
    return d / d.sum(-1, keepdim=True)


def range_target(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """Uniform label distribution over an age *range* (Adience-style groups)."""
    v = torch.arange(AGE_BINS, device=lo.device, dtype=torch.float32)
    d = ((v[None] >= lo[:, None]) & (v[None] <= hi[:, None])).float()
    return d / d.sum(-1, keepdim=True).clamp_min(1.0)
