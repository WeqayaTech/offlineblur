# 03 · SAM 3.1 detector only, `woman` prompt, no tracking
| Role | Component |
|---|---|
| Detector + mask | SAM 3.1 detection stage only (backbone → grounding → mask head → NMS → threshold 0.4 → edge filter) |
| Tracker | none — every frame independent |
| Classifier | the prompt itself |
| GPU | NVIDIA L4 22 GB |
| Speed | **259 ms/frame · 3.9 Hz** (batch 1 fastest; 2/4/8 slower; 16 OOM). ViT backbone = 194 ms of it |
| Peak GPU | 8.1 GB, flat |
| Accuracy | women blurred **0.743**, men false-blur **0.099** — flickers frame to frame |

Clip: `ali-dawah-street-interview-source.mp4` — 300 frames, 1280x720, 25 fps, 12 s, ~60 people, night street.
Accuracy uses the 21 hand-labelled people (11 women, 10 men; `../gt_gender_ali_rfdetr_tracks.json`, scorer
`tracking/gender_gt_eval.py`): **women blurred** = average share of each woman's frames where the blur covers at
least half of her; **men false-blur** = the same over the men.

**How to read `labels_masks_ids.mp4`** (renderer `render_gender.py`): masks are FILLED by gender concept —
magenta = woman, blue = man — with a `prompt:id` tag. Thin grey outlines are the reference `person` layer (SAM 3.1
tracked); a grey outline with no fill is a person the pipeline did not gender in that frame (a potential miss).
`blur_side_by_side.mp4` = original left, blurred right. Here the `woman:id` numbers are per-DETECTION ids (frame*1000+k), not identities — expect
the tag to change every frame.
Metrics: `metrics/sam31_detect_woman`, `metrics/det_sweep` (batch sweep).
