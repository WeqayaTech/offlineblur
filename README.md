# OfflineBlur

Two pipelines live here. **v1** (this file, folder `offlineblur/`) is the production blur pipeline.
**v2** (folder [`v2/`](v2/README.md), design report in [`v2/REPORT.md`](v2/REPORT.md)) is the transformer-driven
spatio-temporal pipeline: end-to-end track-query tracking (MeMOTR / MOTRv2), a frozen 1.1 B-parameter ViT-giant
backbone shared with a cross-attention demographic head, and Bayesian temporal pooling of age and gender per track.
v2 outputs tracks + stable per-person profiles; it does not render the blur yet (that is the v1 stage 5 renderer).

# OfflineBlur v1

Offline video pipeline that blurs every adult woman in a recording at pixel level, for video
editors. Runs on a RunPod GPU pod; nothing runs locally. Output: the blurred video (pixelated,
feathered, audio kept), a white-on-black matte, per-frame masks with identity ids, a debug
video and a review page.

Rules it is built around, in priority order:
1. No false blur — never a non-person (poster, statue, sign) and never a man.
2. No escape — a woman is blurred in every frame she is visible, through short occlusions.
3. Pixel-accurate per-person masks that do not flicker or bleed onto the person next to her.
Children (12 or younger) and men are never blurred.

## The pipeline (v1.4)

| stage | what | model | output |
|---|---|---|---|
| 1 `s1_detect_track.py` | detect + segment + track every person, every frame | YOLO11x-seg + BoT-SORT with ReID | `tracks.jsonl` (per frame: track id, box, conf, RLE mask) |
| 2 `s2_segments.py` | cut raw tracks into single-person segments (gap splits, appearance splits); per segment: crops, OSNet re-id embedding, face embedding (on upscaled crops), sliver + duplicate measures | OSNet (MSMT17, via boxmot), InsightFace SCRFD + ArcFace | `segments.json`, `crops/` |
| 3 `s3_classify.py` | judge every crop: real person? woman/man? child? (face crops weigh ×3, slivers 0) | Qwen2.5-VL-7B-Instruct | `segment_votes.json`, `verdicts.jsonl` |
| 4 `s4_identities.py` | split segments whose votes flip (id switch); decide each segment's class; link segments into identities (duplicate → host, face ≥ 0.45, body ≥ 0.85 + motion + same class, geometric handoff); decide each identity from all its crops; fill short holes | — | `identities.json`, `classes.json` |
| 5 `s5_render.py` | dilate + temporal union + feather + pixelate per identity; matte; masks export | — | `render/` |
| 6 `s6_gallery.py` | review page, uncertain identities first | — | `review.html` |

Classes are decided **before** linking so a link can never turn a woman into a man, and the
identity vote pools every crop afterwards so a person seen mostly from behind still benefits from
the frames where the face was visible. Every stage writes plain files, so a later stage can be
re-run with different settings without re-running the GPU-heavy detection.

Thresholds were calibrated on the trial clip (night street, crowd): OSNet body similarity between
*different* people reaches 0.83 (top 1 %: 0.76), so body links need 0.85 plus motion continuity;
ArcFace similarity between different people never exceeded 0.16, so face links at 0.45 are safe.

## Pod

Pick **H100 SXM 80 GB** or **A100 SXM 80 GB** (stock torch works), or an **RTX PRO 6000 / other
Blackwell** card (works too: `pod_setup.sh` detects a missing sm_120 kernel set and installs the
cu128 torch build; the trial ran on an RTX PRO 6000). v1 needs about 30 GB of VRAM. Storage: 50 GB
container disk is enough for installs + weights (~20 GB) + a few short clips; long videos need a
volume (per 3-hour 1080p video: ~30 GB of outputs; 4K: ~100 GB).

```bash
# on the pod, once
mkdir -p /workspace/offlineblur && cd /workspace/offlineblur
# copy this folder here (scp / upload the tarball), then (paths default to /workspace; override with
# HF_HOME / OFFLINEBLUR_WEIGHTS and the out_root argument when there is no volume):
bash pod_setup.sh
# put a video in /workspace/offlineblur/videos/, then:
MAXS=120 bash run.sh /workspace/offlineblur/videos/clip.mp4      # first 2 minutes
bash run.sh /workspace/offlineblur/videos/clip.mp4 [out_root]    # whole video
```

Run long jobs with `nohup bash run.sh ... > run.log 2>&1 &` and keep the SSH session open
(RunPod can kill detached jobs on a disconnect).

## Outputs (`/workspace/offlineblur/out/<clip>/`)

- `render/<clip>_blurred.mp4` — deliverable, original audio.
- `render/<clip>_matte.mp4` — feathered white-on-black matte, same frame count and fps.
- `render/masks_rle.jsonl` — one line per (frame, identity): `{"f", "identity", "cls", "blurred", "rle"}`.
  Every human identity is exported (men too, `blurred: false`) so an editor can key anyone.
  Decode in Python: `pycocotools.mask.decode({"size": rle["size"], "counts": rle["counts"].encode()})`.
- `render/<clip>_debug.mp4` — every tracked person outlined, identity id, class, `?` = uncertain.
- `review.html` — one card per identity with its crops and votes; dropped / uncertain first.
- `s1.log … s5.log`, `s1_meta.json`, `segments.json`, `classes.json`, `render/render_summary.json`.
- `tools/sheet.py <out_dir> Woman` builds a contact sheet of every identity of one class for a quick eyeball check.

## What to check on the first run

1. `debug.mp4`: does every person carry one id from entry to exit? Where ids change on the same
   person → stage 4 thresholds (`--face-link`, `--body-link`, `--body-max-gap-s`, `--handoff-*`).
2. `review.html`: any dropped identity that is actually a person? any woman flagged `Man`, any
   man `Woman`? → look at the raw answers in `verdicts.jsonl` (bad crops vs bad model).
3. `blurred.mp4`: edge quality, bleed, flicker → `--dilate-frac`, `--temporal`, `--feather`.
4. `s1_meta.json` → `n_box_fallback_masks` should be ~0; `fps_processing` gives the run-time estimate.

## Knobs that matter

- Small / far people: `IMGSZ=1920` (or higher) in `run.sh`, at native resolution nothing is downscaled below that.
- Missed people at low confidence: `S1_ARGS="--new-track 0.4"` (more tracks start; stage 3 filters junk).
- Occlusion longer than 3 s without a visible face: `S1_ARGS="--track-buffer-s 10"`, `S4_ARGS="--body-max-gap-s 10"` (body links are the risky kind; face links already cross any gap).
- Vote more frames: `S2_ARGS="--k 16"` (crops per segment; every saved crop is judged).
- Stricter "no false blur": `S5_ARGS="--no-uncertain-blur"` (uncertain identities are then review-only).
- Posters of real people are NOT blurred by default; `S4_ARGS="--blur-depictions"` flips that.
- Female viewer (blur men): `VIEWER=female`.

## Known limits of v1 (and the v2 candidates)

- Mask edges come from the YOLO segmentation head; good, not matting-grade. v2: refine each
  mask with SAM 2 (box-prompted) if edges are the bottleneck.
- One detector. Very small (< ~40 px) or heavily occluded people can be missed. v2: add a second
  detector (RF-DETR / Grounding DINO) as a union, or tiling for 4K.
- One judge. v2: add MiVOLO V2 (face + body, no language) as an independent second vote.
- Body re-identification is weak on dark night-time crowds (same person across a gap scores ~0.6, strangers up to 0.83), so fragments without a visible face stay separate identities; each is still judged and blurred on its own.
- Licences: ultralytics is AGPL; the InsightFace `buffalo_l` pack is non-commercial (research);
  Qwen2.5-VL is Apache 2.0; DINOv2 is Apache 2.0. Replace ArcFace/SCRFD before a commercial release.
- Identity linking stores embeddings only inside the run folder; nothing is matched across videos.
