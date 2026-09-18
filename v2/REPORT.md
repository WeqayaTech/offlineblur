# OfflineBlur v2 — Transformer-driven spatio-temporal pipeline: design report

Date: 2026-09-17/18. Pod: RunPod A100 80 GB PCIe (torch 2.4.1 + cu124). Code: `v2/` in this repository.

## 1. What this approach is

v1 finds people frame by frame (YOLO-seg), links them with a geometric tracker (BoT-SORT), crops each
person and asks a vision-language model to vote on gender. Every step works on its own crop, so a
background person 60 px tall is judged from 60 px of pixels, and the identity chain breaks each time a
detector miss or an occlusion splits a track.

v2 replaces both the frame-by-frame detection and the crop-and-classify loop with a single
attention-based design in four phases:

```
                         [broadcast video]
                                │
                                ▼
                   ┌───────────────────────────────────┐
                   │ Phase 1  Shared giant backbone    │  DINOv2-giant (ViT-g/14, 1.1 B) on the FULL frame,
                   │          dense feature map        │  resampled + tiled so one token = 8 source pixels
                   └───────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌─────────────────────────────┐   ┌─────────────────────────────────┐
│ Phase 2  Tracking branch    │   │ Phase 3  Demographics branch     │
│ MeMOTR / MOTRv2             │   │ cross-attention ROI head         │
│ track queries + long-term   │──▶│ learned queries attend to the    │
│ memory (300-frame miss      │   │ backbone tokens inside each box  │
│ tolerance)                  │   │ → gender, age dist., quality     │
└─────────────────────────────┘   └─────────────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
                   ┌───────────────────────────────────┐
                   │ Phase 4  Bayesian temporal pooling │  Beta posterior (gender), Gaussian product of
                   │          identity lock             │  experts (age); per-frame weights from quality,
                   └───────────────────────────────────┘  size, occlusion, sharpness, attention focus
                                │
                                ▼
                [continuous track id + stable gender / age per person]
```

### Phase 1 — shared feature extraction (`stp/backbone.py`)
One >1 B-parameter vision transformer sees the whole frame. To keep a small person legible the frame is
resampled so that a 14 px patch covers 8 source pixels ("feature stride 8"); a 60 px person is then an
8×15-token region rather than 4×8. The resampled frame is processed as overlapping 64×64-token tiles
with Hann-window blending, so attention cost stays bounded at any input resolution, and tiles that
contain no tracked person are skipped. Backbones: `dinov2-giant` (default) or `eva02-giant`; both are
public. InternImage-H was evaluated and rejected: its public head only exposes stride-32 pooled
features, which is the opposite of what a 60 px person needs.

### Phase 2 — tracking with track queries (`stp/tracker.py`)
End-to-end transformer trackers. Every person is a learned track query that the decoder carries across
frames; when the person disappears behind the reporter the query lives on in memory (`--miss-tolerance
300` = 10 s at 30 fps) and re-attaches on reappearance without a new id. Two adapters drive the
official research code unchanged, through their own submit scripts:

- **MeMOTR** (long-term memory; MOT17 or DanceTrack checkpoints)
- **MOTRv2** (MOTR with detector proposals as anchor queries; proposals from a YOLO11x person detector)

Both resample frames to 800×1536 internally; that is the resolution their queries were trained at.

### Phase 3 — demographics by cross-attention on the shared map (`stp/roi_attention.py`, `stp/demographics.py`)
No pixel crop. For each tracked box the head gathers the backbone tokens inside the box (+15 % margin),
gives them a box-relative 2-D sine position, and lets 8 learned queries run 3 layers of self-attention /
cross-attention / FFN over them. Pooled queries feed three heads: gender (2-way), age (101-bin label
distribution → mean and std) and a quality logit trained to predict "this frame's answer is right". The
normalised entropy of the last cross-attention is exported too: flat attention means the queries found
nothing to focus on (back view, occlusion, blur).

The head is the only trained component. `stp/train_demographics.py` fine-tunes it on the frozen backbone
with UTKFace (exact age + gender), FairFace (age group + gender; the public stand-in for Adience) and
CelebA (gender), 161 k images exported from Hugging Face by `stp/hf_export.py`. Images are rendered at
random short sides of 56–448 px with blur, JPEG and context-margin augmentation, so training token
densities match what the head meets on a 4K frame.

### Phase 4 — Bayesian temporal pooling (`stp/aggregator.py`)
Every frame contributes with weight `w = quality × g_size × g_occlusion × g_sharpness × (1 − entropy)`.
Gender accumulates a Beta posterior (α += w·p, β += w·(1−p)); age accumulates a Gaussian product of
experts (precision += w/σ², mean = Σ w·μ/σ² / precision). A clear look (w ≥ 0.8) counts three times, so
one sharp frontal frame dominates a long blurry history instead of averaging into it. Once the evidence
exceeds a threshold and the posterior is ≥ 0.9 on one side the profile is locked; later low-weight frames
are damped and cannot flip it. Optionally, tracks whose quality-weighted ROI embeddings agree, do not
overlap in time and are less than 10 s apart are merged before pooling (off by default; calibrate first).

## 2. What is substituted and why

| blueprint component | implemented as | reason |
|---|---|---|
| InternImage-H / ViT-G backbone | DINOv2-giant (or EVA-02-giant) | public ViT-Giants with dense features; no dense InternImage-H head is public |
| MOTR-v2 / MeMOTR with the giant backbone inside | MeMOTR / MOTRv2 as published (ResNet-50 inside) | no MOTR checkpoint exists for a giant backbone; swapping it means retraining the tracker end to end (8× H100, days). The shared backbone therefore serves the demographic branch, which is where 60 px legibility matters |
| Swin-V2-G / EVA-02-G demographic branch | cross-attention ROI head on the shared map | Swin-V2-G weights are not public; the ROI-attention mechanism is implemented as specified |
| UTKFace + Adience + CelebA | UTKFace + FairFace + CelebA | Adience has no public mirror; FairFace carries the same age-group + gender labels with better demographic balance |

## 3. Results so far (12 s trial clip, 1280×720, 25 fps, night street, ~25 people)

Trackers, 300 frames:

| tracker | track ids | boxes | verdict |
|---|---|---|---|
| MeMOTR MOT17, det 0.5 | 41 | 3291 | good; misses two partially visible people at the far left |
| MeMOTR MOT17, det 0.4 | 50 | 6226 | worse: duplicate track queries stacked on the same people |
| MeMOTR DanceTrack ckpt | 17 | 2208 | under-detects on a street crowd |
| **MOTRv2 + YOLO11x proposals** | 41 | 2527 | cleanest: no duplicates, catches the far-left people → new default |

Demographic head, validation on a held-out 5 % of the training mix (8069 gender-labelled, 1180
exact-age faces), 4 epochs at 2.3 steps/s on the A100 (108 min total):

| epoch | gender accuracy | age MAE (years) |
|---|---|---|
| 1 | 0.968 | 5.78 |
| 2 | 0.968 | 4.74 |
| 3 | 0.969 | 4.47 |
| 4 (final) | **0.970** | **4.43** |

Pipeline on the clip with the final head (MOTRv2 tracks, every frame scored): 41 identities,
11 female / 30 male, 7 locked. Reporter locked male P(f)=0.01 (CI 0.00–0.02), interviewee locked
female P(f)=0.94 (CI 0.90–0.99), both as one id across the whole 12 s. People seen only from behind
stay near 0.5 with a wide interval instead of locking wrongly (identity 16: 0.52, CI 0.29–0.75).
Known weakness: ages compress toward the mid-30s (an older man comes out at 43 ± 3); exact-age data
for older faces would fix this. The trained head is `demo_head.pt` (55 MB; a copy is kept outside the
repo in `v2/weights/`).

Speeds (A100 80 GB PCIe): frames → jpg 300 frames in a few seconds; MeMOTR ~2 fps; MOTRv2 ~2 fps plus
proposals; demographic branch at stride 8 ~1.1 fps on a 16-person frame (every frame scored);
pooling and render seconds.

## 4. Pod

1× H100 SXM 80 GB (or A100 80 GB, ~2.5× slower), RunPod PyTorch 2.x / CUDA 12.x image, Hopper or Ampere
only (the trackers compile a Deformable-Attention CUDA op against the image torch; Blackwell pods need a
cu128 torch that breaks it). Everything is kept on the container disk (`/root/offlineblur_v2`, `/root/hf`);
~15 GB for repos, checkpoints and the backbone, ~4 GB for the training images. Backbone swap training
(blueprint step 3, not required to run the pipeline): 8× H100 SXM, 1 TB volume, several days.

## 5. How to run

```bash
bash v2/pod_setup_v2.sh                     # once per pod
EPOCHS=4 bash v2/train_head.sh              # once: exports data from Hugging Face, trains demo_head.pt
TRACKER=motrv2 bash v2/run_v2.sh /root/offlineblur/videos/clip.mp4
```

Outputs per clip: `tracks.jsonl`, `attrs.jsonl`, `identities.json`, `render/<clip>_debug.mp4`,
`render/<clip>_identities_mot.txt`, `render/summary.json` (see `v2/README.md`).

## 5b. Blur deliverable and clip findings (added 2026-09-18)

`stp/blur.py` renders the blurred video: YOLO11x-seg instance mask matched to each target box, clipped to the
box, dilated, feathered, pixelated per person. Default target: P(female) ≥ 0.6 or locked female; 0.5–0.6 is a
review band (`review_sheet.jpg`). On the trial clip: 3 identities blurred (interviewee + 2 background women),
0 false blurs, 4–5 escapes — women seen from behind scored 0.3–0.55 because the head's training data is faces
only. v1 on the same clip: 12 women blurred, 0 false blurs, 1 escape. Pooled ROI embeddings turned out to be
attribute features (cosine 0.96 between strangers), so identity relinking stays off. The full HTML report with
frames is `v2/reports/v2_report.html`.

## 5c. V3 — zero-shot VLM judge, no training (added 2026-09-18)

The trained head's weakness (faces only → back-view women escape) is a data gap, not an architecture gap.
V3 keeps every phase except the demographic branch, which becomes a zero-shot Qwen2.5-VL-7B judge
(`stp/judge_vlm.py`): per track, K crops spread over its life are each labelled woman/man/child + face
visible + view + confidence, mapped into the same `attrs.jsonl` schema, and pooled by the same Bayesian
aggregator. No training, no `demo_head.pt`.

Trial clip, MOTRv2 tracks, K=14:

| pipeline | phase-3 model | women blurred | escapes | note |
|---|---|---|---|---|
| v2 | trained cross-attention head | 3 | 4–5 back-view women | faces-only training domain |
| **v3** | **Qwen2.5-VL-7B zero-shot** | **11** | back-view escapes closed | overconfident on occluded fragments |
| v1 | Qwen per-crop vote (older tracker) | 12 | 1 | reference |

V3 catches the grey-coat and glasses women V2 missed. Its cost is over-aggression on ambiguous/occluded
fragments: the VLM answers with high confidence (0.9+) even on partial crops, so a few thin-evidence
identities (e.g. a couple embracing, a red-and-checkered fragment near the reporter) are blurred on wide
confidence intervals. Tune with `K` (more crops = steadier), `BLUR_ARGS="--min-p 0.65"` (stricter), or
`--locked-only`. Speed: ~0.8 crops/s on the A100, ~12 min for the clip.

Recommendation: V3 for street/crowd footage where people are often seen from behind; the trained head for
face-forward footage where it is faster. The two can also be combined as a two-vote ensemble (both write
`attrs.jsonl`; concatenate before the aggregator).

## 6. Next steps

1. Retrain the head with body-level labels (PA-100K / PETA pedestrian attributes, or Qwen-distilled labels
   from the v1 corpus) so back views stop escaping; this is the single biggest gap versus v1.
2. Calibrate the pooling gates and the relink threshold on the six-clip test corpus from `reports/`.
3. Age: add IMDB-WIKI-style exact-age data or re-weight FairFace groups to fix the compression toward 30.
4. Backbone swap (giant backbone inside MeMOTR) only if tracker misses, not the head, turn out to be the
   accuracy limit on 4K footage.
