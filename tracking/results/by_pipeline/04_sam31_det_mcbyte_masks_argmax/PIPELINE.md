# 04 · SAM 3.1 detector (woman/man/child) + McByte with masks, argmax vote
| Role | Component |
|---|---|
| Detector + mask | SAM 3.1 detector, `woman,man,child` in ONE shared backbone pass, threshold 0.25 |
| Merge | per frame, same-person detections across prompts merged (mask IoU ≥ 0.5) |
| Tracker | McByte (Roboflow trackers 2.6.0) + SAM ViT-B / Cutie mask propagation, high/activation 0.4 |
| Classifier | per-track vote: gender with the largest summed score (argmax) |
| GPU | NVIDIA L4 22 GB |
| Speed | detector 369 ms + McByte 371 ms = 742 ms/frame · **1.35 Hz** |
| Peak GPU | 8.3 GB (detector) then 3.3 GB (McByte) |
| Accuracy | women blurred **0.728**, men false-blur **0.026**; 110 tracks for ~66 people (fragmented) |

Clip: `ali-dawah-street-interview-source.mp4` — 300 frames, 1280x720, 25 fps, 12 s, ~60 people, night street.
Accuracy uses the 21 hand-labelled people (11 women, 10 men; `../gt_gender_ali_rfdetr_tracks.json`, scorer
`tracking/gender_gt_eval.py`): **women blurred** = average share of each woman's frames where the blur covers at
least half of her; **men false-blur** = the same over the men.

**How to read `labels_masks_ids.mp4`** (renderer `render_gender.py`): masks are FILLED by gender concept —
magenta = woman, blue = man — with a `prompt:id` tag. Thin grey outlines are the reference `person` layer (SAM 3.1
tracked); a grey outline with no fill is a person the pipeline did not gender in that frame (a potential miss).
`blur_side_by_side.mp4` = original left, blurred right. Fill colour here is the track's VOTED gender, so a person does not flicker between colours.
Metrics: `metrics/sam31_det_woman_man_child` (detector), `metrics/sam31_mcbyte`, `metrics/sam31_bytetrack` (box-only
variant: 432 Hz tracker, same accuracy).
