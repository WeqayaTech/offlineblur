# OfflineBlur v2 — transformer-driven spatio-temporal pipeline

Continuous background-person tracking + stable age/gender per person, built as the four-phase
attention pipeline (shared giant backbone → track queries → cross-attention ROI demographics →
Bayesian temporal pooling). Runs on a RunPod H100/A100 pod; nothing runs locally.

```
                 [video]  →  frames.py (jpg sequence, MOT layout)
                     │
      ┌──────────────┴──────────────────┐
      ▼                                 ▼
 Phase 2  tracker.py                Phase 1  backbone.py
 MeMOTR (track queries + long-term  DINOv2-giant / EVA-02-giant, 1.1 B params,
 memory, MOT17 or DanceTrack ckpt)  dense stride-8 token map, tiled, full frame
 or MOTRv2 (proposal anchors)                   │
      │  tracks.jsonl                           ▼
      │                             Phase 3  roi_attention.py + demographics.py
      └────────────────────────────▶ cross-attention ROI head on the shared map:
                                    gender, age (101-bin DLDL), quality, attn entropy
                                                │  attrs.jsonl (one row per frame × track)
                                                ▼
                                    Phase 4  aggregator.py
                                    Beta posterior (gender) + Gaussian product of experts (age),
                                    per-frame weights = quality × size × occlusion × sharpness × focus,
                                    clear-look overwrite, identity lock, optional id relink
                                                │  identities.json
                                                ▼
                                    render.py → debug video, MOT text with identity ids, summary
```

## What is real and what is substituted (read this first)

The blueprint names components that have no public weights. This package uses the closest
public equivalents and keeps the *mechanics* of each phase:

| blueprint | here | why |
|---|---|---|
| InternImage-H / ViT-G shared backbone, tiny patch | **DINOv2-giant** (ViT-g/14, 1.1 B) or EVA-02-giant, run on the full frame at a chosen feature stride (default 8 px per token via resampling + tiling) | both are public ViT-Giants with dense features; InternImage-H's public output is stride-32 pooled, not dense |
| MOTR-v2 / MeMOTR with the giant backbone swapped in | **MeMOTR** (default, long-term memory) or **MOTRv2**, run *as published* (ResNet-50 inside) via their own submit scripts | no MOTR checkpoint exists for a giant backbone; swapping it means retraining the tracker (multi-day, 8×H100 — script layout for that is in "Backbone swap" below). The tracker still is the end-to-end track-query transformer of the blueprint |
| Swin-V2-G / EVA-02-G demographic branch with cross-attention ROI | **cross-attention ROI head** (learned queries attend to the backbone tokens inside each tracked box; no pixel crop) on the shared giant map, trained by `train_demographics.py` | Swin-V2-G weights are not public; the ROI-attention mechanism is implemented exactly, on the shared backbone |
| Bayesian temporal pooling | implemented as specified: confidence-weighted Beta/Gaussian posteriors, clear-frame overwrite, lock | — |

Consequence to remember: the trackers resample every frame to 800×1536 internally, so a 4K
background person is tracked at ~40 % scale. The demographic branch has no such limit.

## Pod

**1× H100 SXM 80 GB** (or A100 80 GB, ~2.5× slower), RunPod PyTorch 2.x / CUDA 12.x image, Hopper or
Ampere only — the trackers compile a Deformable-Attention CUDA op against the stock torch; Blackwell
pods (RTX PRO 6000 / 5090 / B200) need a cu128 torch rebuild that breaks that op. 100 GB container
disk, network volume ≥ 300 GB (datasets for the head: UTKFace 0.3 GB, Adience 2 GB, CelebA 1.5 GB;
frames of a 1-hour 4K clip ≈ 60 GB). VRAM in use: ~25 GB inference, ~40 GB training the head.

```bash
# on the pod, once (everything lives on the container disk: /root/offlineblur_v2 and /root/hf — re-run the setup after a pod restart)
mkdir -p /root/offlineblur && cd /root/offlineblur    # copy the v2/ folder here
bash v2/pod_setup_v2.sh          # clones + compiles MeMOTR and MOTRv2, downloads checkpoints and dinov2-giant
# demographic head: exports UTKFace + FairFace + CelebA(40k) from Hugging Face (no Kaggle needed), then trains
EPOCHS=4 bash v2/train_head.sh   # → /root/offlineblur_v2/weights/demo_head.pt  (~95 min on an A100 for 4 epochs)
# a clip
MAXS=120 bash v2/run_v2.sh /root/offlineblur_v2/videos/clip.mp4
```

Run long jobs with `nohup ... > log 2>&1 &`.

Training data (`stp/hf_export.py`): UTKFace 23.7 k (exact age + gender), FairFace 97.7 k (age group + gender,
stands in for Adience, which has no public mirror), CelebA 40 k (gender only). Kaggle copies of
UTKFace / Adience / CelebA work too through `stp/manifest_builders.py`. Every image is rendered at a
random short side between 56 and 448 px, with blur / JPEG / context-margin augmentation, so the head
meets the token densities of 60 px background people as well as the reporter filling the frame.

Measured on the A100 80 GB PCIe pod (2026-09-17): DINOv2-giant fp16, 8 tiles of 64×64 tokens in 0.9 s;
MeMOTR ~2 frames/s at 1280×720; demographic branch at stride 8 ~1 frame/s on a crowd frame (16 boxes);
head training 2.3 steps/s at batch 48 (GPU 95 %).

## Outputs (`/root/offlineblur_v2/out/<clip>/`)

- `tracks.jsonl` — `{"f", "tid", "box":[x1,y1,x2,y2], "score"}` straight from the transformer tracker.
- `attrs.jsonl` — one row per frame × track: `p_female, age_mean, age_std, quality, entropy, size_px, occ, sharp`.
- `identities.json` — per identity: `gender, p_female, gender_ci95, age_mean, age_std, class (Woman/Man/child), locked, evidence, n_obs, tracks, first_f, last_f, span_s, best_frame`.
- `render/<clip>_debug.mp4` — every box with identity id, gender + probability, age ± std, `L` = locked, `q` = that frame's quality.
- `render/<clip>_identities_mot.txt` — MOT text keyed by identity id (for the v1 blur renderer or an editor).
- `memotr.log` / `motrv2.log`, `tracks_meta.json`, `attrs_meta.json`, `render/summary.json`.

## Knobs

- Occlusion length: `MISS=300` (frames a hidden track query stays alive; 300 = 10 s at 30 fps). Longer = fewer id breaks, more risk of a query re-attaching to the wrong person after a long absence.
- Tracker: `TRACKER=motrv2` (default: MOTRv2 with YOLO11x proposal anchors, cleanest on the trial crowd) or `TRACKER=memotr` with `CKPT=memotr_mot17.pth` (street pedestrians) or `memotr_dancetrack.pth` (dense, similar-looking people; under-detects on streets).
- Token density: `STRIDE=8` (source px per token). 6 for very small people at 4K; 14 = native, fastest.
- Demographic sampling: `EVERY=1` scores every frame; `EVERY=3` is 3× faster with little loss after pooling.
- Pooling gates (`AGG_ARGS`): `--size-lo/--size-hi` (px), `--occ-lo/--occ-hi`, `--sharp-lo/--sharp-hi`, `--sharp-thresh --sharp-boost` (clear-look overwrite), `--lock-n --lock-p` (lock), `--post-lock-damp`.
- Identity relink across tracker id breaks: `RELINK=0.92` (cosine of quality-weighted ROI embeddings; off by default — measure on your footage first, as v1 showed body embeddings of strangers can score high in night crowds).
- Detection / track thresholds: `TRACK_ARGS="--det-thresh 0.4 --track-thresh 0.4"`.

## Backbone swap (the blueprint's step 3) — not run by default

MeMOTR is trained end-to-end; replacing its ResNet-50 by the giant ViT means retraining on
MOT17 + CrowdHuman (+ DanceTrack) with `BACKBONE` in its config pointing at a new backbone module
that returns the 4 feature levels the deformable encoder expects (project DINOv2 tokens to 256-d at
strides 8/16/32/64). Budget: 8× H100 SXM, ~3–5 days, 1 TB volume for the datasets. The package does
not need it to run; the shared backbone already serves the demographic branch, which is where the
60 px-person legibility matters.

## Files

`stp/frames.py` `stp/tracker.py` `stp/backbone.py` `stp/roi_attention.py` `stp/demographics.py`
`stp/aggregator.py` `stp/render.py` `stp/train_demographics.py` `stp/manifest_builders.py` `stp/common.py`
`pod_setup_v2.sh` `train_head.sh` `run_v2.sh` `requirements_v2.txt`
