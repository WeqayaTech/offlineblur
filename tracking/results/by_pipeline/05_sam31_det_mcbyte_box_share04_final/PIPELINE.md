# 05 · SAM 3.1 detector + McByte (box overlap) + woman-share ≥ 0.4 — final SAM 3.1 two-stage setup
| Role | Component |
|---|---|
| Detector + mask | same as 04 (woman/man/child, one backbone pass) |
| Tracker | McByte, box overlap only (no SAM/Cutie) — 2–4 ms/frame |
| Classifier | per-track vote: woman if woman ≥ 40% of the track's summed score |
| GPU | NVIDIA L4 22 GB |
| Speed | 373 ms/frame · **2.7 Hz** (2.5 Hz as run) |
| Peak GPU | 8.3 GB |
| Accuracy | women blurred **0.731**, men false-blur **0.039**; 108 tracks |

Clip: `ali-dawah-street-interview-source.mp4` — 300 frames, 1280x720, 25 fps, 12 s, ~60 people, night street.
Accuracy uses the 21 hand-labelled people (11 women, 10 men; `../gt_gender_ali_rfdetr_tracks.json`, scorer
`tracking/gender_gt_eval.py`): **women blurred** = average share of each woman's frames where the blur covers at
least half of her; **men false-blur** = the same over the men.

Only `blur_side_by_side.mp4` was rendered for this run (original left, blurred right).
Metrics: `metrics/sam31_mcbyte_final`.
