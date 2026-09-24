# 10 · Fast pipeline — McByte association tuning for partial occlusion (lost tracks kept 4 s), L4

Pipeline as 07/09 (`tracking/fast_blur.py`, RF-DETR-Seg 2XL → McByte box mode → CLIP ViT-L/14-336, K=10, ≥ 0.25),
lost tracks kept 4 s (`--lost-seconds 4`). What changes is how McByte matches a returning person to their track:

- `--iou biou --biou-buffer R`: **buffered IoU** (C-BIoU) — both boxes enlarged by R × their size before overlap is
  measured, so a person who moved or whose box shrank behind an occluder still overlaps their track.
- `--assoc2 T`: minimum similarity in McByte's 2nd association round, where LOW-score detections (0.15–0.5 — typical of
  a partly hidden person) are matched. McByte's default is 0.5, which a shrunken box rarely reaches.
- 1st round (0.1) and unconfirmed (0.3) kept at McByte defaults — raising the 1st round made every setting worse.

| Setting | Tracks | Identity breaks | Suspected id swaps | Median track life | End to end (L4) |
|---|---|---|---|---|---|
| default (plain IoU, 2nd round 0.5) | 60 | 34 | 0 | 40 frames | 19.9 Hz |
| **buffered 0.5, 2nd round 0.3** | 48 | 18 | **0** | 61 | 20.1 Hz |
| buffered 0.7, 2nd round 0.4 | 44 | 15 | 1 (track #35, frame 263, 10.5 s) | 61 | 20.6 Hz |
| buffered 1.0, 2nd round 0.4 | 42 | 12 | 2 (#32 f244 9.8 s; #33 f263 10.5 s) | 66 | 21.2 Hz |
| buffered 1.5 (sweep only) | 36 | 6 | 5–7 | 83 | — |

"Identity breaks" = a track starting within 4 s where another ended (one person, two ids). "Suspected swaps" = a box
jumping ≥ 0.75 body heights with no overlap inside one track within 3 frames (id moved to another person) — a proxy;
check the listed frames in the videos. Full 50-setting sweep (same detections replayed, `tracking/mcbyte_sweep.py`):
`metrics/sweep/sweep.json`, `metrics/sweep2/sweep.json`. With buffered IoU, 4 s vs 1 s lost time differs by one track.

Videos: `grid_2x2_default_biou05_biou07_biou10.mp4` (top-left default, top-right buffered 0.5 / 0.3, bottom-left
0.7 / 0.4, bottom-right 1.0 / 0.4 — same frames); `labels_masks_ids_<setting>.mp4` full size each (legend as 07:
fill = track, box/tag = gender, tag `#id gender P(female)`); `blur_only_biou05_a203.mp4` = the blurred product with
the recommended setting.
