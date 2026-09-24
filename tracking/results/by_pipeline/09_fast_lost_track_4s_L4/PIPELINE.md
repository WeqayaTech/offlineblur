# 09 · Fast pipeline with lost tracks kept 4 s (occlusion up to 4 s), L4

Pipeline as 07 (`tracking/fast_blur.py`): RF-DETR-Seg 2XLarge (batch 8, threshold 0.15) → McByte box overlap
(new-track threshold 0.5) → OpenCLIP ViT-L/14-336 (10 crops per person, blur at P(woman)+P(girl) ≥ 0.25).
Only change: `--lost-seconds 4.0` → McByte `lost_track_buffer` 120 (McByte counts 30 fps frames and rescales:
120 × 25/30 = 100 frames = 4.0 s at 25 fps). GPU: NVIDIA L4 22 GB (so slower than 07's RTX PRO 6000).

| Lost tracks kept | McByte buffer → frames | Tracks | Likely fragments | Frames with no box | End to end (L4) |
|---|---|---|---|---|---|
| 1.0 s (previous default) | 30 → 25 | 60 | 35 | 63 | 20.0 Hz |
| 2.5 s | 75 → 63 | 60 | 34 | 89 | 19.7 Hz |
| **4.0 s** | 120 → 100 | 60 | 34 | 89 | 19.5 Hz |

Why 4 s barely helps: of the 35 identity breaks at 1.0 s, 24 happen after a gap of ≤ 25 frames (≤ 1 s, while the
old track was still alive) and in 27 the person's reappearing box does not overlap their last box (IoU < 0.1;
median shift half a body height). McByte (box mode) re-associates only by box overlap with a Kalman-predicted
position, so a waiting track is not matched and a new id is started. Keeping tracks longer cannot fix that;
matching by appearance (a ReID embedding) or by position tolerance can.

Videos: `ids_1s_left_vs_4s_right.mp4` (same frames, 1.0 s left, 4.0 s right), `labels_masks_ids_lost_4.0s.mp4`,
`labels_masks_ids_lost_1.0s.mp4` (legend as folder 07: fill = track, box/tag = gender), `blur_side_by_side.mp4`
(4.0 s run; original left, blurred right).
Metrics: `metrics/` (per-run JSON with `lost_buffer`, `lost_frames`, `lost_seconds`, and `.tracks.jsonl` dumps).
