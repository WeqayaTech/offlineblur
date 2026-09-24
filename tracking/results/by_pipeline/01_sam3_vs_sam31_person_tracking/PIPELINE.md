# 01 · SAM 3 vs SAM 3.1 — person tracking only (no gender)
| Role | SAM 3 (left) | SAM 3.1 (right) |
|---|---|---|
| Detector + mask + tracker | SAM 3, transformers 5.17, prompt `person`, one session, memory pruning, state on CPU | SAM 3.1 Object Multiplex (github sam3 @2345a4a), prompt `person`, pruning + crash guard |
| GPU | NVIDIA L4 22 GB | NVIDIA L4 22 GB |
| Speed | 2.59 s/frame · **0.39 Hz** | 0.54 s/frame model (0.60 as run) · **1.84 Hz** |
| Peak GPU | 8.4 GB | 15.1 GB |
| Identities | 61 | 66 |

Clip: `ali-dawah-street-interview-source.mp4` — 300 frames, 1280x720, 25 fps, 12 s, ~60 people, night street.
Accuracy uses the 21 hand-labelled people (11 women, 10 men; `../gt_gender_ali_rfdetr_tracks.json`, scorer
`tracking/gender_gt_eval.py`): **women blurred** = average share of each woman's frames where the blur covers at
least half of her; **men false-blur** = the same over the men.

Videos: `person_tracking_sam3_left_vs_sam31_right.mp4` (each colour = one track id);
`labels_masks_ids_sam31_person.mp4` = SAM 3.1 alone, full size.
Metrics: `metrics/sam3_person`, `metrics/sam31_person`.
