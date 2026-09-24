# 02 · SAM 3.1 tracked, gender as the prompt
| Role | Component |
|---|---|
| Detector + mask + tracker | SAM 3.1 Object Multiplex, one session per prompt: `woman`, `man`, `person` |
| Classifier | the prompt itself: a `woman` mask is blurred |
| GPU | NVIDIA L4 22 GB |
| Speed | 1.46 s/frame model (436 + 480 + 542 ms per prompt session) · **0.69 Hz** (0.62 Hz as run) · 8.1 min for the clip |
| Peak GPU | 16.2 GB |
| Accuracy | women blurred **0.737**, men false-blur **0.168**; 2 women never blurred; woman/man masks overlap on 49% of woman masks |

Clip: `ali-dawah-street-interview-source.mp4` — 300 frames, 1280x720, 25 fps, 12 s, ~60 people, night street.
Accuracy uses the 21 hand-labelled people (11 women, 10 men; `../gt_gender_ali_rfdetr_tracks.json`, scorer
`tracking/gender_gt_eval.py`): **women blurred** = average share of each woman's frames where the blur covers at
least half of her; **men false-blur** = the same over the men.

**How to read `labels_masks_ids.mp4`** (renderer `render_gender.py`): masks are FILLED by gender concept —
magenta = woman, blue = man — with a `prompt:id` tag. Thin grey outlines are the reference `person` layer (SAM 3.1
tracked); a grey outline with no fill is a person the pipeline did not gender in that frame (a potential miss).
`blur_side_by_side.mp4` = original left, blurred right. Red outline = person whose verdict was escape (never confidently gendered).

For reference (no local video): SAM 3 tracked (transformers, woman/man/person in one session) on the L4 ran at
5.63 s/frame (0.18 Hz), 16.4 GB, women blurred 0.799, men false-blur 0.214 — see `metrics/sam3_gender`.
Metrics: `metrics/sam31_gender`, `metrics/sam31_gender_t57` (SAM 3's thresholds 0.5/0.7 — no better).
