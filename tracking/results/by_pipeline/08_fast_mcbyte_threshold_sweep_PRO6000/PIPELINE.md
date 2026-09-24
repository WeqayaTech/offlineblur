# 08 · Fast pipeline — McByte new-track threshold sweep (crowd recall), RTX PRO 6000
Pipeline as 07; only McByte's high-confidence threshold (the score a person needs to START a track) changes.
Lost-track buffer setting 30 at 0.5 and 75 for the others. McByte counts it in 30 fps frames and rescales to the
video's 25 fps: 30 → 25 frames (1.0 s), 75 → 63 frames (2.5 s).
| Threshold | End to end | People tracked / frame | Tracks | Tracks labelled woman | Video |
|---|---|---|---|---|---|
| 0.50 (current) | 86 Hz | 13 | 61 | 24 | `labels_masks_ids_thresh_0.50.mp4` |
| 0.40 | 82 Hz | 15 | 71 | 30 | `labels_masks_ids_thresh_0.40.mp4` |
| 0.30 | 78 Hz | 16 | 81 | 40 | `labels_masks_ids_thresh_0.30.mp4` |
| 0.25 | 74 Hz | 17 | 91 | 50 | `labels_masks_ids_thresh_0.25.mp4` |
| 0.20 | 72 Hz | 18 | 96 | 52 | `labels_masks_ids_thresh_0.20.mp4` |
| SAM 3.1 / SAM 3 tracked (reference, L4) | — | 22 / 24 | 66 / 58 | — | folders 01–02 |
`grid_2x2_thresh_050_040_030_025.mp4` = same frame four ways: top-left 0.50, top-right 0.40, bottom-left 0.30,
bottom-right 0.25. `blur_only_thresh_0.50.mp4` / `_0.30.mp4` = the blurred product at two thresholds.
Watch for: background people only appearing at lower thresholds; tag numbers changing on one person (fragments);
magenta tags on men at low thresholds (over-blur from short tracks). Camera-motion compensation (`--cmc`) changed
nothing and cost 2.5 ms/frame (see `metrics/trk/B_cmc*.json`).

Clip: `ali-dawah-street-interview-source.mp4` — 300 frames, 1280x720, 25 fps, 12 s, ~60 people, night street.
Accuracy uses the 21 hand-labelled people (11 women, 10 men; `../gt_gender_ali_rfdetr_tracks.json`, scorer
`tracking/gender_gt_eval.py`): **women blurred** = average share of each woman's frames where the blur covers at
least half of her; **men false-blur** = the same over the men.

**How to read `labels_masks_ids*.mp4`** (renderer `fast_blur.py --debug-out`): mask FILL colour = track identity
(same colour = same id), box and tag colour = gender decision (magenta woman, cyan man), tag = `#id  gender  P(female)`
(blurred when P(female) >= 0.25). Nothing is blurred in these. Top bar: frame, tracked / women / men counts.
`blur_side_by_side.mp4` = original left, blurred right.
Metrics: `metrics/vis` (per-threshold runs behind these videos), `metrics/trk` (tracker variants + track dumps;
`tracking/track_quality.py` reads the `.tracks.jsonl` files).
