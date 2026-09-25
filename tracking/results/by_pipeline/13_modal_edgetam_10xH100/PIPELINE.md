# 13 · RF-DETR-Seg + EdgeTAM + PE-Core-L on Modal, one video split over 10 H100s

Same pipeline as run 12 (`fast_blur_edgetam.py`), EdgeTAM people batched per frame, run in parallel chunks
(`dist_edgetam.py` steps, launcher `modal_edgetam.py`).

**Input:** `dm_dutchman_converts.mp4`, 60 s continuous stretch from 0:30 (1,803 frames, 1280×720, 30 fps).

## Pipeline, per chunk (one H100 each)
1. Decode own frames plus a 32-frame warm-up before them (ffmpeg, own thread).
2. RF-DETR-Seg 2XL person boxes + masks (fp16, batch 8, score ≥ 0.15).
3. EdgeTAM image encoder once per 8 frames; one batched EdgeTAM memory step for all live people per frame.
4. Match detections to EdgeTAM masks (IoU ≥ 0.3); start a person from an unmatched detection ≥ 0.5; drop duplicates (≥ 0.7); end after 4 s unconfirmed.
5. PE-Core-L/14-336 scores each person's views.

Then the coordinator links people across chunk boundaries (mask IoU ≥ 0.5 over the warm-up frames). It pools each person's views and takes the vote: 10 views, P(woman)+P(girl) ≥ 0.25. Each GPU then blurs its own frames in memory, and the coordinator joins the segments without re-encoding.

## Metrics (models loaded and warm; container start and model load, 25 s, excluded)
| | 10 × H100, split | 1 × H100, whole video |
|---|---|---|
| Wall time, 60 s video | **22.7 s** | 89.0 s |
| End-to-end rate | **79.4 Hz** | 20.3 Hz |
| Stages | tracking 16.1 s · vote 2.0 s · blur 3.3 s · join 1.4 s | tracking 81.1 s |
| People / women | 123 / 50 | 111 / 42 |

Repeat of the same configuration: 82.4 Hz (21.9 s). Per frame on each GPU: EdgeTAM steps ~32 ms · EdgeTAM image ~14 ·
RF-DETR 4.2 · rest ≈ 5.

## Videos
- `compare_ids_single_vs_10xH100.mp4`: masks (colour = person id), box/tag = gender `#id label P(female)`;
  1 × H100 left, 10 × H100 split right; Hz on each panel
- `compare_blur_original_single_10xH100.mp4`: original | 1 × H100 | 10 × H100 split, Hz on each panel
- `blur_10xH100_split.mp4`: the split output, full size

Run: `modal run tracking/modal_edgetam.py --video dm_dutchman_converts.mp4 --clip-start 30 --clip-seconds 60 --chunks 10 --warm 32 --overlay`
